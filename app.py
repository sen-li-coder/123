# -*- coding: utf-8 -*-
"""
多对多协同防空任务筹划与推演系统 v5.0
========================================
模块化架构 · 真正多对多分配 · 统计检验 · LLM校验 · CSV导出
运行: streamlit run app.py
"""
# Load pandas completely before Plotly/Streamlit. Plotly checks for
# pandas.Series while validating traces; importing it first avoids a transient
# circular-import state in some Streamlit reloads.
import pandas as pd
if not hasattr(pd, "Series"):
    raise RuntimeError("pandas 导入不完整，请关闭旧的 Streamlit 进程后重新启动 Python 环境")
import streamlit as st, plotly.graph_objects as go, numpy as np
import time, random, math, os, json, csv, io, copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from scipy import stats as sc_stats
try: from dotenv import load_dotenv; load_dotenv()
except ImportError: pass

# ---- core imports ----
from core.constants import *
from core.models import (Missile, Position, Interceptor, Assignment,
                          SimulationStats, SimulationConfig, SimulationResult, IVStatus)
from core.math_utils import d3, norm3, predict_3d
from core.guidance import pn_guidance
from core.threat import compute_threat
from core.simulation import step_simulation, snapshot_state, run_full_simulation
from core.scenario import gen_scenario, SCENARIOS, get_scenario
from core.validation import validate_assignments, validate_llm_plan
# ---- algorithms ----
from algorithms.random_strategy import RandomAllocation
from algorithms.nearest import NearestAllocation
from algorithms.greedy import GreedyAllocation
from algorithms.optimized import OptimizedAllocation
STRATEGIES = {"random": RandomAllocation(), "nearest": NearestAllocation(),
              "greedy": GreedyAllocation(), "optimized": OptimizedAllocation()}
# ---- experiments ----
from experiments.statistics import mean_ci, paired_ttest
from experiments.monte_carlo import run_monte_carlo as mc_run, export_csv
from experiments.validation_cases import run_validation_suite
# ---- services ----
from services.llm_service import llm_reallocate, MockLLM
from integrations.afsim_adapter import (
    AFSIMConfig, AFSIMRunResult, build_exchange_payload, compare_plan_execution, compare_stats,
    generate_afsim_scenario, open_mystic, open_warlock, parse_afsim_output,
    run_batch, run_warlock, scenario_id_for_payload, write_exchange_json,
)

# ============================================================
# 展示层常量(仅影响可视化, 不改变仿真/分配/导引逻辑)
# ============================================================
# 目标配色: 经 dataviz 校验器验证(亮度带/色度/CVD ΔE≥8/正常视觉ΔE≥15 全部通过)。
# 固定顺序分配, 对象颜色只跟随实体ID, 不随排序/筛选变化。
MSL_COLORS = ["#D7263D","#00897B","#EF6C00","#1565C0",
              "#9E9D24","#8E24AA","#C9B400","#43A047"]
# 画质档位: 展示帧目标数 / 轨迹显示点数 / 动态模型形式 / 基准帧间隔
QUALITY_PRESETS = {
    "流畅": {"target_frames": 200, "trail_len": 100, "model": "marker", "frame_ms": 60},
    "清晰": {"target_frames": 400, "trail_len": 200, "model": "simple", "frame_ms": 70},
}
MAX_SLIDER_STEPS = 80  # 时间轴控制点上限(原始实现每帧一个step, 上千帧时时间轴自身也很重)
MIN_FRAME_MS, MAX_FRAME_MS = 30, 200  # 播放帧间隔的下限/上限(倍速映射后)
_ANIM_BUILD_COUNT = 0  # 动画图构建次数(验证session_state缓存命中: 每次构建+1)

# 浏览器帧率探针: 挂在父文档的 plotly 图上监听 plotly_animatingframe 事件,
# 在组件内直接显示, 不写回 session_state(避免触发 rerun 循环)。
FPS_PROBE_HTML = """
<div id="cafps" style="font:13px -apple-system,'Segoe UI',sans-serif;color:#4b5563;padding:2px 0">
⏳ 浏览器帧率: 播放动画后自动测量</div>
<script>
(function(){
  var el=document.getElementById('cafps');
  var wired=new WeakSet();
  function attach(gd){
    var n=0,t0=0;
    gd.addEventListener('plotly_animatingframe',function(){
      var now=performance.now();
      if(!t0){t0=now;}
      n++;
      if(now-t0>=1000){
        el.textContent='🎞 浏览器帧率: '+(n*1000/(now-t0)).toFixed(0)+' FPS';
        n=0;t0=now;
      }
    });
    gd.addEventListener('plotly_animated',function(){n=0;t0=0;});
  }
  function scan(){
    var gds=window.parent.document.querySelectorAll('.js-plotly-plot');
    for(var i=0;i<gds.length;i++){
      if(!wired.has(gds[i])){wired.add(gds[i]);attach(gds[i]);}
    }
  }
  setInterval(scan,1500);scan();
})();
</script>
"""

# ============================================================
# 3D模型 + 可视化
# ============================================================
def _rodrigues(vx,vy,vz,ax,ay,az,a):
    c,s=math.cos(a),math.sin(a)
    cx,cy,cz=ay*vz-az*vy,az*vx-ax*vz,ax*vy-ay*vx
    d=ax*vx+ay*vy+az*vz
    return (vx*c+cx*s+ax*d*(1-c),vy*c+cy*s+ay*d*(1-c),vz*c+cz*s+az*d*(1-c))

def _missile_mesh(x,y,z,vx,vy,vz,body_len=30,radius=5):
    n=8;h=body_len/2;bt=h-body_len*0.35;bb=-h
    v=[(0,bb,0)]; br=[]; tr=[]
    for i in range(n):
        a=2*math.pi*i/n; br.append(len(v))
        v.append((radius*math.cos(a),bb,radius*math.sin(a)))
    for i in range(n):
        a=2*math.pi*i/n; tr.append(len(v))
        v.append((radius*math.cos(a),bt,radius*math.sin(a)))
    tip=len(v); v.append((0,h,0))
    i_f,j_f,k_f=[],[],[]
    for i in range(n): i_f.append(0);j_f.append(br[i]);k_f.append(br[(i+1)%n])
    for i in range(n):
        nx=(i+1)%n; i_f.append(br[i]);j_f.append(br[nx]);k_f.append(tr[i])
        i_f.append(tr[i]);j_f.append(br[nx]);k_f.append(tr[nx])
    for i in range(n): nx=(i+1)%n; i_f.append(tr[i]);j_f.append(tr[nx]);k_f.append(tip)
    sp=math.sqrt(vx**2+vy**2+vz**2)
    nx,ny,nz=(vx/sp,vy/sp,vz/sp) if sp>1e-6 else (0,1,0)
    ra=math.acos(max(-1,min(1,ny)))
    if abs(ra)>1e-6 and abs(ra-math.pi)>1e-6:
        rx,rz=-nz,nx; rm=math.sqrt(rx**2+rz**2)
        if rm>1e-6: rx/=rm; rz/=rm
        v=[_rodrigues(vx,vy,vz,rx,0,rz,ra) for (vx,vy,vz) in v]
    return [vv[0]+x for vv in v],[vv[1]+y for vv in v],[vv[2]+z for vv in v],i_f,j_f,k_f

def _house_mesh(x,y,z,size=25):
    h=size; rh=size*0.7; hf=h/2
    b0=(-hf,-hf,0);b1=(hf,-hf,0);b2=(hf,hf,0);b3=(-hf,hf,0)
    t0=(-hf,-hf,h);t1=(hf,-hf,h);t2=(hf,hf,h);t3=(-hf,hf,h)
    apex=(0,0,h+rh)
    v=[b0,b1,b2,b3,t0,t1,t2,t3,apex]; i,j,k=[],[],[]
    def q(a,b,c,d): i.append(a);j.append(b);k.append(c);i.append(a);j.append(c);k.append(d)
    q(0,3,2,1)
    for s in[(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]: q(*s)
    i.append(4);j.append(5);k.append(8);i.append(5);j.append(6);k.append(8)
    i.append(6);j.append(7);k.append(8);i.append(7);j.append(4);k.append(8)
    return [vv[0]+x for vv in v],[vv[1]+y for vv in v],[vv[2]+z for vv in v],i,j,k

def _launcher_mesh(x, y, z, size=26):
    """Compact launcher vehicle model for the research visualization."""
    w, d, h = size * 0.8, size * 1.35, size * 0.28
    verts=[(x-w/2,y-d/2,z),(x+w/2,y-d/2,z),(x+w/2,y+d/2,z),(x-w/2,y+d/2,z),
           (x-w/2,y-d/2,z+h),(x+w/2,y-d/2,z+h),(x+w/2,y+d/2,z+h),(x-w/2,y+d/2,z+h)]
    ii=[0,0,4,4,0,1,2,3]; jj=[1,2,5,6,1,2,3,0]; kk=[2,3,6,7,5,6,7,4]
    return ([v[0] for v in verts],[v[1] for v in verts],[v[2] for v in verts],ii,jj,kk)

def _radar_mesh(x, y, z, size=22):
    """Dish-and-mast silhouette to make each site visually identifiable."""
    theta=np.linspace(0,2*np.pi,18)
    dish_x=x+size*0.55*np.cos(theta); dish_y=y+size*0.55*np.sin(theta)
    dish_z=np.full_like(theta,z+size*0.75)
    return dish_x,dish_y,dish_z

def _simple_missile_verts(x, y, z, vx, vy, vz, L=16):
    """清晰模式下的简化弹体: 6顶点8面箭头, 拓扑固定, 比完整弹体轻一个数量级。"""
    sp=math.sqrt(vx**2+vy**2+vz**2)
    nx,ny,nz=(vx/sp,vy/sp,vz/sp) if sp>1e-6 else (0,1,0)
    ra=math.acos(max(-1,min(1,ny)))
    r=L*0.35; b=-L*0.3
    local=[(0,L,0),(0,-L*0.45,0)]
    for a in (0,math.pi/2,math.pi,3*math.pi/2):
        local.append((r*math.cos(a),b,r*math.sin(a)))
    if abs(ra)>1e-6 and abs(ra-math.pi)>1e-6:
        rx,rz=-nz,nx; rm=math.sqrt(rx**2+rz**2)
        if rm>1e-6: rx/=rm; rz/=rm
        local=[_rodrigues(vx,vy,vz,rx,0,rz,ra) for (vx,vy,vz) in local]
    return ([p[0]+x for p in local],[p[1]+y for p in local],[p[2]+z for p in local],
            [0,0,0,0,1,1,1,1],[2,3,4,5,2,3,4,5],[3,4,5,2,3,4,5,2])

def _combined_msl_msh(msls):
    ax,ay,az,ai,aj,ak=[],[],[],[],[],[]
    for m in msls[:8]:
        mx,my,mz,mi,mj,mk=_missile_mesh(m.x,m.y,m.z,m.vx,m.vy,m.vz)
        o=len(ax); ax+=mx;ay+=my;az+=mz
        ai+=[x+o for x in mi];aj+=[x+o for x in mj];ak+=[x+o for x in mk]
    return ax,ay,az,ai,aj,ak

def _combined_house_msh(poss):
    ax,ay,az,ai,aj,ak=[],[],[],[],[],[]
    for p in poss:
        hx,hy,hz,hi,hj,hk=_house_mesh(p.x,p.y,p.z)
        o=len(ax); ax+=hx;ay+=hy;az+=hz
        ai+=[x+o for x in hi];aj+=[x+o for x in hj];ak+=[x+o for x in hk]
    return ax,ay,az,ai,aj,ak

def _terrain_surface(nx=28, ny=24):
    """Deterministic low-relief terrain for the command-view preview.

    This is a visualization layer only.  The authoritative AFSIM terrain and
    entity dynamics remain in the generated AER replay.
    """
    x=np.linspace(0,MW,nx); y=np.linspace(0,MH,ny)
    xx,yy=np.meshgrid(x,y)
    z=(8*np.exp(-(((xx-0.25*MW)/(0.22*MW))**2+((yy-0.72*MH)/(0.28*MH))**2))
       +5*np.exp(-(((xx-0.78*MW)/(0.20*MW))**2+((yy-0.28*MH)/(0.24*MH))**2))
       +2*np.sin(xx/MW*math.pi)*np.cos(yy/MH*math.pi))
    return xx,yy,z

def _add_operational_context(fig, positions):
    """Add terrain, range rings and command-post labels to a Plotly view."""
    tx,ty,tz=_terrain_surface()
    fig.add_trace(go.Surface(x=tx,y=ty,z=tz,colorscale=[[0,'#203b38'],[0.45,'#3b5f4f'],[0.75,'#8a825e'],[1,'#c6b87b']],
        opacity=0.78,showscale=False,name='地形',hoverinfo='skip'))
    for p in positions:
        theta=np.linspace(0,2*np.pi,100)
        rr=float(p.intercept_radius)
        fig.add_trace(go.Scatter3d(x=p.x+rr*np.cos(theta),y=p.y+rr*np.sin(theta),z=np.full_like(theta,3.0),
            mode='lines',line=dict(color='rgba(45,220,170,0.6)',width=3,dash='dash'),
            name=f'P{p.id} 防区',showlegend=False,hoverinfo='skip'))
        # Radar coverage fan (stylized, with explicit disclosure in UI).
        r2=rr*0.72; ang=np.linspace(0,2*np.pi,50)
        fig.add_trace(go.Scatter3d(x=np.r_[p.x,p.x+r2*np.cos(ang),p.x],y=np.r_[p.y,p.y+r2*np.sin(ang),p.y],
            z=np.r_[8*np.ones(1),8*np.ones_like(ang),8*np.ones(1)],mode='lines',
            line=dict(color='rgba(69,180,255,0.3)',width=2),showlegend=False,hoverinfo='skip'))

def build_figure(msls,poss,ivs,destroyed,trails=True):
    fig=go.Figure()
    _add_operational_context(fig, poss)
    for xx in np.arange(0,MW,MW//5):
        fig.add_trace(go.Scatter3d(x=[xx,xx],y=[0,MH],z=[0,0],mode='lines',
            line=dict(color='rgba(140,160,180,0.3)',width=0.5),showlegend=False,hoverinfo='skip'))
    for yy in np.arange(0,MH,MH//5):
        fig.add_trace(go.Scatter3d(x=[0,MW],y=[yy,yy],z=[0,0],mode='lines',
            line=dict(color='rgba(140,160,180,0.3)',width=0.5),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Mesh3d(x=[0,MW,MW,0],y=[0,0,MH,MH],z=[0,0,0,0],
        i=[0,0],j=[1,2],k=[2,3],color='#C8D6B8',opacity=0.3,name='地面',showlegend=True,hoverinfo='skip'))
    for p in poss:
        u=np.linspace(0,2*np.pi,14);v=np.linspace(0,np.pi,9)
        sx=p.x+p.intercept_radius*np.outer(np.cos(u),np.sin(v))
        sy=p.y+p.intercept_radius*np.outer(np.sin(u),np.sin(v))
        sz=p.z+p.intercept_radius*np.outer(np.ones_like(u),np.cos(v))
        fig.add_trace(go.Surface(x=sx,y=sy,z=sz,colorscale=[[0,C_SPHERE],[1,C_SPHERE]],
            opacity=0.08,showscale=False,showlegend=False,hoverinfo='skip'))
    if poss:
        hx,hy,hz,hi,hj,hk=_combined_house_msh(poss)
        fig.add_trace(go.Mesh3d(x=hx,y=hy,z=hz,i=hi,j=hj,k=hk,
            color=C_POSITION,opacity=0.9,flatshading=True,name='防御阵地',showlegend=True,hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[p.x for p in poss],y=[p.y for p in poss],z=[p.z+35 for p in poss],
            mode='text',text=[f'P{p.id}' for p in poss],textfont=dict(color=C_POSITION,size=14),
            showlegend=False,hoverinfo='skip'))
        lx,ly,lz,li,lj,lk=[],[],[],[],[],[]
        for p in poss:
            vx,vy,vz,vi,vj,vk=_launcher_mesh(p.x,p.y,p.z)
            off=len(lx); lx+=vx;ly+=vy;lz+=vz;li += [q+off for q in vi];lj += [q+off for q in vj];lk += [q+off for q in vk]
            dx,dy,dz=_radar_mesh(p.x,p.y,p.z)
            fig.add_trace(go.Scatter3d(x=dx,y=dy,z=dz,mode='lines',line=dict(color='#8ee8ff',width=4),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Mesh3d(x=lx,y=ly,z=lz,i=li,j=lj,k=lk,color='#4b6474',opacity=0.95,flatshading=True,name='机动发射单元',showlegend=True,hoverinfo='skip'))
    alive=[m for m in msls if m.alive]
    if alive:
        mx,my,mz,mi,mj,mk=_combined_msl_msh(alive)
        fig.add_trace(go.Mesh3d(x=mx,y=my,z=mz,i=mi,j=mj,k=mk,
            color=C_MISSILE,opacity=0.92,flatshading=True,name='来袭导弹',showlegend=True,hoverinfo='skip'))
    aiv=[i for i in ivs if i.alive]
    if aiv:
        ix,iy,iz=[],[],[]
        for iv in aiv:
            sp=math.sqrt(iv.vx**2+iv.vy**2+iv.vz**2)
            vx,vy,vz=(iv.vx/sp*100,iv.vy/sp*100,iv.vz/sp*100) if sp>1e-6 else (0,100,0)
            mx,my,mz,mi,mj,mk=_missile_mesh(iv.x,iv.y,iv.z,vx,vy,vz,body_len=20,radius=3)
            off=len(ix); ix+=mx;iy+=my;iz+=mz
            if aiv.index(iv)==0:
                fig.add_trace(go.Mesh3d(x=ix,y=iy,z=iz,i=mi,j=mj,k=mk,
                    color=C_INTERCEPTOR,opacity=0.88,flatshading=True,name='拦截弹',showlegend=True,hoverinfo='skip'))
    if destroyed:
        fig.add_trace(go.Scatter3d(x=[d[0] for d in destroyed],y=[d[1] for d in destroyed],
            z=[d[2] for d in destroyed],mode='markers',
            marker=dict(symbol='x',size=8,color=C_DESTROYED,line=dict(width=2)),name='已拦截',hoverinfo='skip'))
    if trails:
        # 轨迹实线加粗并按目标配色: 旧实现虚线宽1透明度0.5, 很难看清
        for i,m in enumerate(msls):
            t=m.trail
            if len(t)>=2:
                fig.add_trace(go.Scatter3d(x=[p[0] for p in t],y=[p[1] for p in t],z=[p[2] for p in t],
                    mode='lines',line=dict(color=MSL_COLORS[i%len(MSL_COLORS)],width=4),
                    showlegend=False,hoverinfo='skip',opacity=0.95))
    fig.update_layout(scene=dict(
        xaxis=dict(title='X',range=[0,MW],showgrid=True,gridcolor='rgba(180,190,210,0.3)'),
        yaxis=dict(title='Y',range=[0,MH],showgrid=True,gridcolor='rgba(180,190,210,0.3)'),
        zaxis=dict(title='Z',range=[0,MD],showgrid=True,gridcolor='rgba(180,190,210,0.3)'),
        aspectmode='data',camera=dict(eye=dict(x=0.5,y=0.5,z=0.45)),bgcolor='#E8EDF5'),
        width=750,height=580,margin=dict(l=0,r=0,t=15,b=0),paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0),
        hovermode='closest',uirevision='constant')
    return fig

def build_animated_figure(frames, positions_static, quality="流畅", speed=1.0):
    """轻量推演动画视图(仅展示层, 不改变仿真/分配/导引逻辑)。

    与旧实现的关键区别:
    - 展示帧降采样: 仿真仍按0.1s步长, 展示帧按画质控制在约200/400帧;
    - 动态对象用Scatter3d标记/轨迹, 不再逐帧重建Mesh3d; 静态地形/阵地/网格只构建一次;
    - 按对象ID累积历史轨迹并逐帧增量更新, 每个对象固定 轨迹/位置/标签 Trace;
    - go.Frame(traces=[...])显式指定帧数据对应的Trace索引, 帧间只更新x/y/z;
    - 时间轴控制点限制在约80个, 标签使用真实推演时间。
    返回 (fig, meta), meta为性能基准数据(原始帧数/展示帧数/构建耗时/序列化体积)。
    """
    global _ANIM_BUILD_COUNT
    _ANIM_BUILD_COUNT += 1  # 模块级后备计数(无Streamlit上下文时使用)
    # 正式计数放session_state: Streamlit每次rerun会重新执行模块级代码, 全局变量会被清零
    try:
        _cnt = int(st.session_state.get("anim_build_count", 0)) + 1
        st.session_state.anim_build_count = _cnt
    except Exception:
        _cnt = _ANIM_BUILD_COUNT
    t0=time.perf_counter()
    preset=QUALITY_PRESETS.get(quality,QUALITY_PRESETS["流畅"])
    n_orig=len(frames)
    # 展示帧降采样: 按总帧数动态计算采样间隔, 把展示帧控制在target_frames附近
    step=max(1,math.ceil(n_orig/preset["target_frames"]))
    dsp=frames[::step]
    if len(dsp)<2: dsp=list(frames[:2])
    elif dsp[-1] is not frames[-1]: dsp.append(frames[-1])  # 保证动画播到最终态势
    dsp_t=[min(i*step,n_orig-1)*DT for i in range(len(dsp))]

    msl_slots=[d["id"] for d in dsp[0]["msl"]]  # 导弹槽位顺序固定(快照顺序稳定)
    trail_buf=[[] for _ in msl_slots]  # 历史轨迹缓冲区(按展示帧累积)
    trail_len=preset["trail_len"]

    fig=go.Figure()
    # ---- 静态层: 只创建一次, 不进入任何动画帧 ----
    _add_operational_context(fig, positions_static)
    lx,ly,lz,li,lj,lk=[],[],[],[],[],[]
    for p in positions_static:
        vx,vy,vz,vi,vj,vk=_launcher_mesh(p.x,p.y,p.z)
        off=len(lx); lx+=vx;ly+=vy;lz+=vz;li += [q+off for q in vi];lj += [q+off for q in vj];lk += [q+off for q in vk]
        dx,dy,dz=_radar_mesh(p.x,p.y,p.z)
        fig.add_trace(go.Scatter3d(x=dx,y=dy,z=dz,mode='lines',line=dict(color='#8ee8ff',width=4),showlegend=False,hoverinfo='skip'))
    if lx:
        fig.add_trace(go.Mesh3d(x=lx,y=ly,z=lz,i=li,j=lj,k=lk,color='#4b6474',opacity=0.95,flatshading=True,name='机动发射单元',showlegend=True,hoverinfo='skip'))
    for xx in np.arange(0,MW,MW//5):
        fig.add_trace(go.Scatter3d(x=[xx,xx],y=[0,MH],z=[0,0],mode='lines',
            line=dict(color='rgba(140,160,180,0.3)',width=0.5),showlegend=False,hoverinfo='skip'))
    for yy in np.arange(0,MH,MH//5):
        fig.add_trace(go.Scatter3d(x=[0,MW],y=[yy,yy],z=[0,0],mode='lines',
            line=dict(color='rgba(140,160,180,0.3)',width=0.5),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Mesh3d(x=[0,MW,MW,0],y=[0,0,MH,MH],z=[0,0,0,0],
        i=[0,0],j=[1,2],k=[2,3],color='#C8D6B8',opacity=0.3,name='地面',showlegend=True,hoverinfo='skip'))
    for p in positions_static:
        u=np.linspace(0,2*np.pi,12);v=np.linspace(0,np.pi,8)
        sx=p.x+p.intercept_radius*np.outer(np.cos(u),np.sin(v))
        sy=p.y+p.intercept_radius*np.outer(np.sin(u),np.sin(v))
        sz=p.z+p.intercept_radius*np.outer(np.ones_like(u),np.cos(v))
        fig.add_trace(go.Surface(x=sx,y=sy,z=sz,colorscale=[[0,C_SPHERE],[1,C_SPHERE]],
            opacity=0.08,showscale=False,showlegend=False,hoverinfo='skip'))
    if positions_static:
        hx,hy,hz,hi,hj,hk=_combined_house_msh(positions_static)
        fig.add_trace(go.Mesh3d(x=hx,y=hy,z=hz,i=hi,j=hj,k=hk,
            color=C_POSITION,opacity=0.9,flatshading=True,name='防御阵地',showlegend=True,hoverinfo='skip'))

    # ---- 动态层: 每个对象固定Trace(轨迹/位置/标签), 帧间只更新x/y/z ----
    dyn_idx=[]  # 动态Trace索引, go.Frame(traces=...)按此顺序映射帧数据
    simple_topo=_simple_missile_verts(0,0,0,0,1,0)[3:]  # 简化模型拓扑只算一次
    for i,mid in enumerate(msl_slots):
        c=MSL_COLORS[i%len(MSL_COLORS)]
        fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode='lines',
            line=dict(color=c,width=5 if quality=="流畅" else 6),
            name=f'M{mid}轨迹',showlegend=True,hoverinfo='skip')); dyn_idx.append(len(fig.data)-1)
        fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='markers',
            marker=dict(symbol='diamond',size=11,color=c,line=dict(color='#FFFFFF',width=2)),
            name=f'M{mid}标记',showlegend=False,hoverinfo='skip')); dyn_idx.append(len(fig.data)-1)
        fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='text',
            text=[f'M{mid}'],textfont=dict(color='#334155',size=13),
            name=f'M{mid}标签',showlegend=False,hoverinfo='skip')); dyn_idx.append(len(fig.data)-1)
        if preset["model"]=="simple":
            _n6=[float('nan')]*6
            fig.add_trace(go.Mesh3d(x=_n6,y=_n6,z=_n6,i=simple_topo[0],j=simple_topo[1],k=simple_topo[2],
                color=c,opacity=0.92,flatshading=True,
                name=f'M{mid}模型',showlegend=False,hoverinfo='skip')); dyn_idx.append(len(fig.data)-1)
    fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='markers',
        marker=dict(symbol='circle',size=6,color=C_INTERCEPTOR,line=dict(width=1,color='#0B5C2E')),
        name='拦截弹',showlegend=True,hoverinfo='skip')); dyn_idx.append(len(fig.data)-1)
    fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='markers',
        marker=dict(symbol='x',size=8,color=C_DESTROYED,line=dict(width=2)),
        name='已拦截',showlegend=True,hoverinfo='skip')); dyn_idx.append(len(fig.data)-1)

    def _r(v): return float(round(v,1))  # 展示坐标保留0.1精度, 大幅压缩传输体积

    # ---- 动画帧: 只携带动态Trace的增量数据, 结构固定 ----
    plotly_frames=[]
    for fi,s in enumerate(dsp):
        if fi==0: continue
        m_by_id={q['id']:q for q in s['msl']}
        fe=[]
        for i,mid in enumerate(msl_slots):
            d=m_by_id.get(mid)
            if d is not None and d['alive']:
                px,py,pz=_r(d['x']),_r(d['y']),_r(d['z'])
                trail_buf[i].append((px,py,pz))
                if len(trail_buf[i])>trail_len: del trail_buf[i][:len(trail_buf[i])-trail_len]
            else:
                px=py=pz=None
            fe.append(go.Scatter3d(x=[p[0] for p in trail_buf[i]],
                y=[p[1] for p in trail_buf[i]],z=[p[2] for p in trail_buf[i]]))
            fe.append(go.Scatter3d(x=[px],y=[py],z=[pz]))
            fe.append(go.Scatter3d(x=[px],y=[py],z=[None if pz is None else pz+12],text=[f'M{mid}']))
            if preset["model"]=="simple":
                if d is not None and d['alive']:
                    mx,my,mz,_,_,_=_simple_missile_verts(d['x'],d['y'],d['z'],d['vx'],d['vy'],d['vz'])
                    mx,my,mz=[_r(q) for q in mx],[_r(q) for q in my],[_r(q) for q in mz]
                else:
                    mx=my=mz=[float('nan')]*6
                fe.append(go.Mesh3d(x=mx,y=my,z=mz))
        iv_alive=[d for d in s['iv'] if d['alive']]
        fe.append(go.Scatter3d(x=[_r(d['x']) for d in iv_alive],
            y=[_r(d['y']) for d in iv_alive],z=[_r(d['z']) for d in iv_alive]))
        fe.append(go.Scatter3d(x=[_r(d[0]) for d in s['des']],
            y=[_r(d[1]) for d in s['des']],z=[_r(d[2]) for d in s['des']]))
        plotly_frames.append(go.Frame(data=fe,traces=dyn_idx,name=str(fi)))

    # 时间轴控制点: 限制在约80个(旧实现每帧一个step, 上千帧时时间轴本身也很重)
    n_dsp=len(dsp)
    slider_idx=sorted(set(int(round(1+k*(n_dsp-2)/(MAX_SLIDER_STEPS-1))) for k in range(MAX_SLIDER_STEPS)))
    slider_idx=[j for j in slider_idx if 0<j<n_dsp]
    # 播放帧间隔: 1x约60ms, 随倍速缩放(30~200ms), 显示帧降采样后无需每帧40ms硬刷新
    frame_ms=max(MIN_FRAME_MS,min(MAX_FRAME_MS,round(preset["frame_ms"]/max(speed,1e-3))))
    fig.frames=plotly_frames
    fig.update_layout(scene=dict(
        xaxis=dict(title='X',range=[0,MW],showgrid=True,gridcolor='rgba(180,190,210,0.3)'),
        yaxis=dict(title='Y',range=[0,MH],showgrid=True,gridcolor='rgba(180,190,210,0.3)'),
        zaxis=dict(title='Z',range=[0,MD],showgrid=True,gridcolor='rgba(180,190,210,0.3)'),
        aspectmode='data',camera=dict(eye=dict(x=0.5,y=0.5,z=0.45)),bgcolor='#E8EDF5'),
        width=750,height=580,margin=dict(l=0,r=0,t=15,b=0),paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0),
        hovermode='closest',uirevision='constant',
        updatemenus=[dict(type='buttons',showactive=False,buttons=[
            dict(label='▶',method='animate',args=[None,dict(frame=dict(duration=frame_ms,redraw=True),fromcurrent=True,mode='immediate')]),
            dict(label='⏸',method='animate',args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')])],
            x=0.05,y=0)],
        sliders=[dict(active=0,steps=[dict(method='animate',
            args=[[str(j)],dict(mode='immediate',frame=dict(duration=0,redraw=True))],label=f'T+{dsp_t[j]:.1f}s')
            for j in slider_idx],currentvalue=dict(prefix='推演时间:'),len=0.9,x=0.1,y=0)])

    build_s=round(time.perf_counter()-t0,3)
    payload_mb=round(len(fig.to_json())/1e6,2)
    meta={"原始帧数":n_orig,"展示帧数":len(dsp),"采样间隔":step,"构建耗时s":build_s,
          "序列化体积MB":payload_mb,"轨迹点数":trail_len,"动态模型":preset["model"],
          "画质":quality,"播放间隔ms":frame_ms,"倍速":speed,"构建次数":_cnt}
    return fig,meta


def build_game_trajectory_demo(frame_count=180):
    """轻量级游戏动画视图；路径为预设曲线，不读取推演或导引数据。"""
    duration = 18.0
    times = np.linspace(0.0, duration, frame_count)
    colors = ("#22D3EE", "#FB923C", "#A78BFA")
    names = ("飞行体 A", "飞行体 B", "飞行体 C")

    def path(track_id):
        phase = track_id * 2 * np.pi / 3
        angle = 2 * np.pi * times / duration + phase
        x = MW / 2 + 145 * np.cos(angle)
        y = MH / 2 + 105 * np.sin(angle)
        z = 105 + 58 * np.sin(2 * angle + phase)
        return x, y, np.clip(z, 20, MD - 15)

    paths = [path(i) for i in range(3)]
    fig = go.Figure()

    # 完整路线只绘制一次，用于帮助观察整体空间形状。
    for i, (x, y, z) in enumerate(paths):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name=f"{names[i]}完整路径",
            line=dict(color=colors[i], width=2, dash="dot"), opacity=0.22,
            hoverinfo="skip", showlegend=False,
        ))

    # 固定三个历史轨迹 Trace；动画帧只更新坐标，避免反复创建网格模型。
    for i, (x, y, z) in enumerate(paths):
        fig.add_trace(go.Scatter3d(
            x=x[:1], y=y[:1], z=z[:1], mode="lines", name=names[i],
            line=dict(color=colors[i], width=7), hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=[p[0][0] for p in paths], y=[p[1][0] for p in paths],
        z=[p[2][0] for p in paths], mode="markers+text",
        marker=dict(size=8, color=colors, line=dict(color="#FFFFFF", width=2)),
        text=names, textposition="top center",
        textfont=dict(color="#E2E8F0", size=13), name="当前位置",
        hovertemplate="%{text}<br>X %{x:.0f}<br>Y %{y:.0f}<br>Z %{z:.0f}<extra></extra>",
    ))

    animation_frames = []
    for index in range(1, frame_count):
        trail_start = max(0, index - 90)
        trail_data = [
            go.Scatter3d(x=x[trail_start:index + 1], y=y[trail_start:index + 1],
                         z=z[trail_start:index + 1])
            for x, y, z in paths
        ]
        trail_data.append(go.Scatter3d(
            x=[p[0][index] for p in paths], y=[p[1][index] for p in paths],
            z=[p[2][index] for p in paths], text=names,
        ))
        animation_frames.append(go.Frame(
            name=str(index), data=trail_data, traces=[3, 4, 5, 6],
        ))
    fig.frames = animation_frames

    # 时间轴只设置少量跳转点，避免为每一帧创建复杂控件。
    slider_indices = sorted(set(range(1, frame_count, max(1, frame_count // 18))) | {frame_count - 1})
    slider_steps = [dict(
        method="animate", label=f"{times[i]:.0f}s",
        args=[[str(i)], dict(mode="immediate", transition=dict(duration=0),
                             frame=dict(duration=0, redraw=True))],
    ) for i in slider_indices]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, MW], gridcolor="rgba(148,163,184,.18)",
                       backgroundcolor="#0B1421"),
            yaxis=dict(title="Y", range=[0, MH], gridcolor="rgba(148,163,184,.18)",
                       backgroundcolor="#0B1421"),
            zaxis=dict(title="高度", range=[0, MD], gridcolor="rgba(148,163,184,.18)",
                       backgroundcolor="#0B1421"),
            aspectmode="data", bgcolor="#0B1421",
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.9)),
        ),
        height=610, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#0B1421", font=dict(color="#CBD5E1"),
        legend=dict(orientation="h", x=0, y=1.03), uirevision="game-demo-camera",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.02, y=0.02,
            bgcolor="rgba(15,23,42,.85)", bordercolor="#334155",
            buttons=[
                dict(label="▶ 播放", method="animate", args=[
                    None, dict(fromcurrent=True, mode="immediate",
                               transition=dict(duration=0),
                               frame=dict(duration=55, redraw=True))]),
                dict(label="⏸ 暂停", method="animate", args=[
                    [None], dict(mode="immediate", transition=dict(duration=0),
                                 frame=dict(duration=0, redraw=False))]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.18, y=0.02, len=0.78, pad=dict(t=25),
            currentvalue=dict(prefix="时间 "), steps=slider_steps,
        )],
    )
    return fig

# ============================================================
# LLM
# ============================================================
def call_llm(prompt, default, system_prompt=None, max_tokens=1200, temperature=0.4):
    try:
        from openai import OpenAI
        k=os.getenv("DEEPSEEK_API_KEY","")
        if not k: return default
        msgs=[]
        if system_prompt: msgs.append({"role":"system","content":system_prompt})
        msgs.append({"role":"user","content":prompt})
        c=OpenAI(api_key=k,base_url=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com/v1"),
                  timeout=30, max_retries=2)
        r=c.chat.completions.create(model=os.getenv("DEEPSEEK_MODEL","deepseek-chat"),
            messages=msgs,temperature=temperature,max_tokens=max_tokens)
        return r.choices[0].message.content.strip() or default
    except Exception as e:
        return f"{default}\n\n*(LLM不可用: {str(e)[:80]})*"

# ============================================================
# UI
# ============================================================
def init_session():
    defaults={"missiles":[],"positions":[],"interceptors":[],"assignments":[],
        "allocation":{},"manual_allocation":{},"sim_initialized":False,
        "sim_running":False,"sim_finished":False,"stats":SimulationStats(),
        "assessment_report":"","optimization_plan":"","report_source":"",
        "report_generated_at":"","report_signature":"","anim_frames":None,"sim_events":[],
        "mc_result":None,"comparison_result":None,"sweep_result":None,
        "algorithm":"greedy","llm_enabled":False,
        "afsim_scenario_path":"","afsim_output_dir":"","afsim_result":None,
        "afsim_payload":None,"afsim_compare":None,"afsim_message":"","validation_result":None,"assignment_errors":[],
        "destroyed_markers":[],"display_quality":"流畅",
        "anim_fig":None,"anim_fig_sig":None,"anim_fig_meta":None,"anim_frames_token":0,"anim_build_count":0}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def render_sidebar():
    st.sidebar.title("🛡️ 防空推演 v5.0")
    with st.sidebar.expander("⚙️ LLM配置",expanded=False):
        k=st.text_input("API Key",value=os.getenv("DEEPSEEK_API_KEY",""),type="password")
        if k: os.environ["DEEPSEEK_API_KEY"]=k
        st.text_input("Base URL",value=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com/v1"))
        st.text_input("Model",value=os.getenv("DEEPSEEK_MODEL","deepseek-chat"))
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 想定设置")
    scenario=st.sidebar.selectbox("场景",["normal","saturation","high_speed","multi_dir"],
        format_func=lambda x: {"normal":"普通攻击","saturation":"饱和攻击","high_speed":"高速突防","multi_dir":"多方向协同"}.get(x,x))
    n_msl=st.sidebar.slider("导弹数量",2,8,5,1)
    spd_min=st.sidebar.slider("最低速度",50,250,100,10)
    spd_max=st.sidebar.slider("最高速度",100,400,250,10)
    if spd_min>spd_max: spd_min=spd_max
    n_pos=st.sidebar.slider("阵地数量",2,4,3,1); ammo=st.sidebar.slider("拦截弹/阵地",2,8,4,1)
    radius=st.sidebar.slider("拦截半径",80,200,150,5); seed=st.sidebar.number_input("随机种子",0,99999,42,1)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 任务筹划")
    algo=st.sidebar.selectbox("分配算法",["greedy","random","nearest","optimized"])
    llm_on=st.sidebar.checkbox("LLM动态重分配",value=False)
    st.sidebar.subheader("⏱️ 倍速")
    speed=st.sidebar.select_slider("倍速",[0.25,0.5,1,2,5,10,20],value=1,format_func=lambda v:f"{v}x",label_visibility="collapsed")
    st.sidebar.subheader("🎨 显示画质")
    quality=st.sidebar.radio("画质模式",["流畅","清晰"],horizontal=True,
        key="display_quality",
        help="流畅: 约200展示帧 · 标记点 · 100点轨迹\n清晰: 约400展示帧 · 简化模型 · 200点轨迹")
    st.sidebar.markdown("---")
    c1,c2=st.sidebar.columns(2)
    with c1: btn_init=st.button("🎲 初始化",use_container_width=True)
    with c2: btn_reset=st.button("🔄 重置",use_container_width=True)
    btn_start=st.button("▶️ 推演" if not st.session_state.get("sim_running",False) else "⏸️ 暂停",use_container_width=True,type="primary")
    with st.sidebar.expander("🔬 效能评估",expanded=False):
        mc_n=st.number_input("蒙特卡洛次数",10,500,50,10)
        if st.button("🔬 蒙特卡洛",use_container_width=True):
            with st.spinner(f"{mc_n}次..."):
                strat=STRATEGIES[algo]
                r=mc_run(mc_n,strat,n_msl,(spd_min,spd_max),n_pos,ammo,radius,seed)
                st.session_state.mc_result=r
        if st.button("🧪 基础验证场景", use_container_width=True):
            with st.spinner("正在运行基础验证..."):
                st.session_state.validation_result = run_validation_suite()
            st.rerun()
        if st.button("⚔️ 策略对比",use_container_width=True):
            with st.spinner("对比中(20次/策略)..."):
                gr=[]; rr=[]
                for i in range(20):
                    ms,ps=gen_scenario(n_msl,(spd_min,spd_max),n_pos,ammo,seed=seed+i)
                    for p in ps: p.intercept_radius=radius
                    a1=STRATEGIES["greedy"].allocate(ms,ps); r1=run_full_simulation(ms,ps,a1)
                    a2=STRATEGIES["random"].allocate(ms,ps); r2=run_full_simulation(ms,ps,a2)
                    s1=r1.stats; s2=r2.stats
                    gr.append(s1.interception_rate); rr.append(s2.interception_rate)
                st.session_state.comparison_result=paired_ttest(gr,rr)
            st.rerun()
    return {"n_msl":n_msl,"spd":(spd_min,spd_max),"n_pos":n_pos,"ammo":ammo,
        "radius":radius,"seed":seed,"speed":speed,"algo":algo,"llm_on":llm_on,
        "btn_init":btn_init,"btn_start":btn_start,"btn_reset":btn_reset,"scenario":scenario,
        "quality":quality}

def render_right(stats,assessment,events):
    st.subheader("📊 效能评估")
    st.metric("拦截率 (MOE)",f"{stats.interception_rate*100:.1f}%")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("拦截",stats.intercepted); c2.metric("漏网",stats.escaped)
    c3.metric("命中率",f"{stats.hit_rate*100:.0f}%"); c4.metric("发射",stats.launched)
    st.progress((stats.intercepted+stats.escaped)/max(stats.total_missiles,1),
        f"进度: {stats.intercepted+stats.escaped}/{stats.total_missiles}")
    ok,msg=stats.validate()
    if not ok: st.warning(f"⚠️ 统计不一致: {msg}")
    if events:
        with st.expander("⏱️ 时间线",expanded=False):
            for t,et,d in events:
                ic={'launch':'🚀','hit':'💥','escape':'⚠️','miss':'❌','llm':'🤖','warn':'⚠️'}
                st.caption(f"T+{t:.1f}s {ic.get(et,'•')} {d}")
    if st.session_state.get("sim_initialized"):
        st.caption("战后总结与 AI 优化方案已移动到中央规划预览下方。")


def render_after_action_panel(stats, assessment, events):
    """Render the post-run analysis beneath the central planning preview."""
    opt_plan = st.session_state.get("optimization_plan", "")
    st.markdown("---")
    st.subheader("📝 战后复盘与 AI 优化")
    if not assessment and not opt_plan:
        st.caption("完成一次推演后，这里会显示战后总结、问题定位和可执行优化方案。")
        return
    if assessment:
        src=st.session_state.get("report_source","")
        ts=st.session_state.get("report_generated_at","")
        cap=f"来源: {src}" if src else ""
        if ts: cap+=f" | 生成: {ts}"
        with st.expander("📝 战后总结", expanded=True):
            if cap: st.caption(cap)
            st.markdown(assessment)
    if opt_plan:
        with st.expander("🤖 AI 优化方案", expanded=True):
            st.markdown(opt_plan)
    c1, c2 = st.columns(2)
    with c1:
        if assessment and st.button("🔄 重新生成报告", use_container_width=True, key="regenerate_report_bottom"):
            st.session_state.report_signature=""; st.rerun()
    with c2:
        if st.session_state.get("sim_initialized") and st.button("📥 导出报告", use_container_width=True, key="export_report_bottom"):
            _export(stats, events)

def render_afsim_panel(cfg):
    """Render the file-exchange bridge without changing the local simulator."""
    st.info("研究闭环：Python 方案配置 → 任务分配 → AFSIM 批处理 → Mystic AER 回放。网页三维图是筹划预览，AFSIM/Mystic 输出用于最终演示和结果核验。")
    st.markdown("---")
    st.subheader("🛰️ AFSIM 联动演示")
    st.caption("Python 负责任务筹划，Warlock 负责仿真，Mystic 负责态势显示。第一版采用场景文件和事件文件交换。")
    af_cfg = AFSIMConfig.from_env()
    with st.expander("AFSIM 路径与运行参数", expanded=False):
        warlock_path = st.text_input("Warlock 可执行文件", value=af_cfg.warlock_exe, key="afsim_warlock_path")
        batch_path = st.text_input("批处理引擎（mission.exe）", value=af_cfg.batch_exe, key="afsim_batch_path")
        mystic_path = st.text_input("Mystic 可执行文件", value=af_cfg.mystic_exe, key="afsim_mystic_path")
        timeout = st.number_input("Warlock 超时（秒）", min_value=1, max_value=3600, value=int(af_cfg.timeout_seconds), key="afsim_timeout")
        af_cfg.warlock_exe = warlock_path
        af_cfg.batch_exe = batch_path
        af_cfg.mystic_exe = mystic_path
        af_cfg.timeout_seconds = int(timeout)
    if not st.session_state.get("sim_initialized"):
        st.info("请先点击左侧“初始化”，再导出 AFSIM 场景。")
        return
    if st.session_state.get("assignment_errors"):
        st.warning("当前方案未完全通过约束校验：" + "；".join(st.session_state.assignment_errors))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        export_clicked = st.button("📤 导出场景", use_container_width=True, key="afsim_export")
    with c2:
        run_clicked = st.button("▶️ 运行 AFSIM", use_container_width=True, key="afsim_run")
    with c3:
        mystic_clicked = st.button("🗺️ 打开 Mystic", use_container_width=True, key="afsim_mystic")
    with c4:
        read_clicked = st.button("📥 读取结果", use_container_width=True, key="afsim_read")

    with c5:
        warlock_clicked = st.button("Open Warlock", use_container_width=True, key="afsim_warlock")

    source_missiles = st.session_state.get("initial_missiles") or st.session_state.get("missiles", [])
    source_positions = st.session_state.get("initial_positions") or st.session_state.get("positions", [])
    source_assignments = st.session_state.get("initial_assignments") or st.session_state.get("assignments", [])
    payload = build_exchange_payload(
        source_missiles, source_positions, source_assignments, scenario_name=cfg.get("scenario", "air_defense_demo"),
        duration=60.0, algorithm=cfg.get("algo", ""), seed=cfg.get("seed"),
    )
    project_root = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.join(project_root, af_cfg.workspace) if not os.path.isabs(af_cfg.workspace) else af_cfg.workspace
    workspace = os.path.join(workspace_root, payload.get("scenario_id", scenario_id_for_payload(payload)))
    scenario_path = st.session_state.get("afsim_scenario_path", "")
    if export_clicked or (run_clicked and not scenario_path):
        with st.spinner("正在生成 AFSIM 场景文件..."):
            scenario_path = str(generate_afsim_scenario(payload, workspace, af_cfg))
            st.session_state.afsim_scenario_path = scenario_path
            st.session_state.afsim_output_dir = os.path.join(workspace, "warlock_output")
            st.session_state.afsim_payload = payload
            st.session_state.afsim_message = f"场景已生成：{scenario_path}"
        st.success(st.session_state.afsim_message)
    if run_clicked:
        if not scenario_path:
            st.error("无法运行：请先导出场景。")
        else:
            with st.spinner("正在运行 AFSIM 批处理仿真..."):
                result = run_batch(scenario_path, af_cfg)
            st.session_state.afsim_result = result.as_dict()
            st.session_state.afsim_output_dir = result.output_dir
            st.session_state.afsim_message = result.error or "Warlock 运行完成。"
            if result.ok:
                st.success("AFSIM 批处理仿真完成，可打开 Mystic 查看 AER 回放。")
            else:
                st.error(st.session_state.afsim_message)
    if mystic_clicked:
        if not scenario_path:
            st.error("请先导出场景，再打开 Mystic。")
        else:
            mystic_input = os.path.join(st.session_state.get("afsim_output_dir", ""), "air_defense_demo.aer")
            if not os.path.exists(mystic_input):
                st.error("当前实验还没有生成 AER 回放文件，请先点击“运行 AFSIM”，等待批处理完成后再打开 Mystic。")
            else:
                ok, message = open_mystic(mystic_input, af_cfg)
                (st.success if ok else st.error)(message)
    if warlock_clicked:
        if not scenario_path:
            st.error("请先导出场景，再打开 Warlock。")
        else:
            ok, message = open_warlock(scenario_path, af_cfg)
            (st.success if ok else st.error)(message)
    if read_clicked:
        output_dir = st.session_state.get("afsim_output_dir") or os.path.join(workspace, "warlock_output")
        parsed = parse_afsim_output(output_dir, target_count=len(st.session_state.get("missiles", [])))
        old = st.session_state.get("afsim_result") or {}
        old.update({"events": parsed["events"], "stats": parsed["stats"], "output_dir": output_dir})
        old["stats"]["plan_execution"] = compare_plan_execution(
            st.session_state.get("afsim_payload") or payload, parsed["events"])
        st.session_state.afsim_result = old
        st.session_state.afsim_compare = compare_stats(st.session_state.get("stats", SimulationStats()), parsed["stats"])
        st.session_state.afsim_message = f"已扫描 {len(parsed['files_scanned'])} 个输出文件，解析 {len(parsed['events'])} 条事件。"
        st.info(st.session_state.afsim_message)

    result = st.session_state.get("afsim_result") or {}
    stats = result.get("stats") or {}
    if result:
        if result.get("error"):
            st.warning(result["error"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AFSIM 拦截率", f"{float(stats.get('interception_rate', 0))*100:.1f}%")
        m2.metric("发射", int(stats.get("launched", 0)))
        m3.metric("命中", int(stats.get("hit", 0)))
        m4.metric("逃逸", int(stats.get("escaped", 0)))
        plan_exec = stats.get("plan_execution")
        if plan_exec:
            st.caption(f"计划/实际发射匹配：{plan_exec.get('matched_count', 0)}/{plan_exec.get('planned_count', 0)}；"
                       f"实际发射 {plan_exec.get('actual_launch_count', 0)} 次")
            if plan_exec.get("unmatched_plans"):
                st.warning("存在未执行的计划任务，请检查目标是否被探测、是否进入交战包线或弹药/通道是否不足。")
        if result.get("log_path"):
            st.caption(f"日志：{result['log_path']}")
        events = result.get("events") or []
        if events:
            with st.expander("AFSIM 事件时间线", expanded=False):
                for item in events:
                    try: t, kind, detail = item
                    except (TypeError, ValueError): continue
                    st.caption(f"T+{float(t):.1f}s · {kind} · {detail}")
    comparison = st.session_state.get("afsim_compare")
    if comparison:
        st.markdown("#### Python 与 AFSIM 结果对比")
        rows = []
        for key, label in (("total_missiles", "目标数"), ("intercepted", "拦截"), ("escaped", "逃逸"), ("launched", "发射"), ("hit", "命中"), ("missed", "脱靶"), ("interception_rate", "拦截率")):
            item = comparison[key]
            rows.append({"指标": label, "Python": f"{item['local']*100:.1f}%" if key == "interception_rate" else item["local"], "AFSIM": f"{item['afsim']*100:.1f}%" if key == "interception_rate" else item["afsim"], "差值": f"{item['delta']*100:.1f}pp" if key == "interception_rate" else item["delta"]})
        st.dataframe(rows, use_container_width=True, hide_index=True)
    if st.session_state.get("afsim_scenario_path"):
        st.caption(f"最近场景：{st.session_state.afsim_scenario_path}")
        replay_path = os.path.join(st.session_state.get("afsim_output_dir", ""), "air_defense_demo.aer")
        if os.path.exists(replay_path):
            st.caption(f"Mystic 回放文件：{replay_path}")

def _export(stats,events):
    mc=st.session_state.get("mc_result"); cp=st.session_state.get("comparison_result")
    aar=st.session_state.get("assessment_report","")
    opt=st.session_state.get("optimization_plan","")
    seed=st.session_state.get("report_signature","")
    lines=["# 防空推演报告",f"时间: {datetime.now():%Y-%m-%d %H:%M} 种子标识: {seed}","",
        f"## 效能指标",f"拦截率: {stats.interception_rate*100:.1f}%",
        f"拦截:{stats.intercepted} 漏网:{stats.escaped} 命中率:{stats.hit_rate*100:.0f}%",
        f"发射:{stats.launched} 命中:{stats.hit} 未中:{stats.missed} 冗余:{stats.redundant}","",
        "## 时间线"]
    for t,et,d in events:
        lines.append(f"- T+{t:.1f}s {d}")
    if mc: lines+=["",f"## 蒙特卡洛(N={mc['n']})",f"拦截率:{mc['mean']*100:.1f}% ±{mc['ci95']*100:.1f}pp (95%CI)"]
    if cp:
        gm=cp.get('greedy_mean',0)*100; rm=cp.get('random_mean',0)*100
        lines+=["",f"## 策略对比",f"贪心:{gm:.1f}% vs 随机:{rm:.1f}%",
            f"d={cp['cohens_d']:.2f} p={cp['p_val']:.4f} {'✅显著' if cp['significant'] else '❌不显著'}"]
    if aar: lines+=["","## 战后总结","",aar]
    if opt: lines+=["","## AI优化方案","",opt]
    st.download_button("⬇ 下载报告",'\n'.join(lines),"report.md","text/markdown")
    # CSV导出
    if mc:
        from experiments.monte_carlo import export_csv
        csv_data=export_csv(mc['trials'],mc['strategy'])
        st.download_button("⬇ CSV",csv_data,f"mc_{mc['strategy']}.csv","text/csv")

def render_mc(result):
    st.markdown("---"); st.subheader("🔬 蒙特卡洛")
    c1,c2,c3=st.columns(3)
    c1.metric("拦截率",f"{result['mean']*100:.1f}%")
    c2.metric("95%CI",f"±{result['ci95']*100:.1f}pp")
    c3.metric("N",result['n'])
    iv=[r['interception_rate'] for r in result['trials']]
    fig=go.Figure(go.Histogram(x=iv,nbinsx=15,marker_color=C_INTERCEPTOR))
    fig.update_layout(title="拦截率分布",height=200,margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig,use_container_width=True)

def render_comparison(result):
    st.markdown("---"); st.subheader("⚔️ 策略对比")
    sign="✅ 显著(p<0.05)" if result['significant'] else "❌ 不显著"
    st.caption(f"d={result['cohens_d']:.2f} | t={result['t_stat']:.2f} | p={result['p_val']:.4f} | {sign}")

def _render_perf_panel(meta):
    """渲染性能基准面板(仅展示层诊断, 不影响推演)。"""
    if not meta: return
    with st.expander("📈 渲染性能基准", expanded=False):
        c1,c2,c3,c4=st.columns(4)
        c1.metric("原始帧数",meta["原始帧数"]); c2.metric("展示帧数",meta["展示帧数"])
        c3.metric("采样间隔",f"1/{meta['采样间隔']}"); c4.metric("轨迹点数",meta["轨迹点数"])
        c5,c6,c7,c8=st.columns(4)
        c5.metric("传输体积",f"{meta['序列化体积MB']}MB"); c6.metric("构建耗时",f"{meta['构建耗时s']}s")
        c7.metric("播放间隔",f"{meta['播放间隔ms']}ms"); c8.metric("动态模型",meta["动态模型"])
        st.caption(f"图构建次数: {meta.get('构建次数','?')}(推演帧/画质/倍速不变时不重建, 数字不变即缓存命中)。"
                   "浏览器帧率在动画播放时自动测量(下方探针, 挂在父文档 plotly 事件上); "
                   "也可用浏览器 DevTools→Performance 复核。")
        try:
            import streamlit.components.v1 as stc
            stc.html(FPS_PROBE_HTML, height=40)
        except Exception:
            pass

# ============================================================
# 主函数
# ============================================================
def main():
    st.set_page_config(page_title="防空推演 v5.0",page_icon="🛡️",layout="wide",initial_sidebar_state="expanded")
    st.title("🛡️ 多对多协同防空任务筹划与推演系统 v5.0")
    st.caption("模块化架构 · 真正多对多分配 · PN制导 · 统计检验 · LLM校验 · CSV导出")
    st.caption("Research loop: configure -> allocate -> Python baseline -> AFSIM batch -> Mystic AER replay -> evaluation")
    init_session(); cfg=render_sidebar()

    if cfg["btn_init"]:
        with st.spinner("生成想定..."):
            ms,ps=gen_scenario(cfg["n_msl"],cfg["spd"],cfg["n_pos"],cfg["ammo"],seed=cfg["seed"])
            for p in ps: p.intercept_radius=cfg["radius"]
            strat=STRATEGIES[cfg["algo"]]
            assignments=strat.allocate(ms,ps)
            _plan_ok, _plan_errors = validate_assignments(assignments, ms, ps)
            st.session_state.missiles=ms; st.session_state.positions=ps; st.session_state.interceptors=[]
            st.session_state.assignments=assignments; st.session_state.sim_initialized=True
            # Preserve immutable initial conditions for Python/AFSIM comparison
            # even after the local simulation mutates live entity state.
            st.session_state.initial_missiles = copy.deepcopy(ms)
            st.session_state.initial_positions = copy.deepcopy(ps)
            st.session_state.initial_assignments = copy.deepcopy(assignments)
            st.session_state.assignment_errors = _plan_errors
            st.session_state.sim_running=False; st.session_state.sim_finished=False
            st.session_state.stats=SimulationStats(total_missiles=len(ms))
            st.session_state.assessment_report=""; st.session_state.optimization_plan=""
            st.session_state.anim_frames=None; st.session_state.sim_events=[]
            st.session_state.destroyed_markers=[]
            st.session_state.anim_fig=None; st.session_state.anim_fig_sig=None; st.session_state.anim_fig_meta=None
            st.session_state.anim_frames_token=0
            st.session_state.mc_result=None
            # A new plan must never reuse an older AFSIM scenario or result.
            st.session_state.afsim_scenario_path=""; st.session_state.afsim_output_dir=""
            st.session_state.afsim_result=None; st.session_state.afsim_payload=None
            st.session_state.afsim_compare=None; st.session_state.afsim_message=""
        st.success(f"✅ {len(ms)}导弹 {len(ps)}阵地 {cfg['algo']}算法 {len(assignments)}分配 seed={cfg['seed']}")
        if _plan_errors:
            st.warning("任务分配存在约束问题：" + "；".join(_plan_errors))
        st.rerun()
    if cfg["btn_reset"]:
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    if cfg["btn_start"]:
        if not st.session_state.get("sim_initialized"): st.sidebar.warning("请先初始化")
        else:
            st.session_state.sim_running=not st.session_state.get("sim_running",False)
            if st.session_state.sim_running:
                st.session_state.sim_finished=False; st.session_state.assessment_report=""; st.session_state.anim_frames=None
                st.session_state.anim_fig=None; st.session_state.anim_fig_sig=None; st.session_state.anim_fig_meta=None
            st.rerun()

    af=st.session_state.get("anim_frames"); sa=st.session_state.get("sim_running",False)
    left,main,right=st.columns([0.15,0.55,0.30])
    with main:
        view_mode=st.radio(
            "三维视图",
            ("推演预览", "游戏轨迹演示"),
            horizontal=True,
            label_visibility="collapsed",
            key="main_3d_view_mode",
        )
        if view_mode == "游戏轨迹演示":
            st.caption("预设游戏动画，与任务分配、导引和推演结果隔离；拖动可旋转，滚轮可缩放。")
            fig=build_game_trajectory_demo()
            st.plotly_chart(fig,use_container_width=True)
        else:
            if st.session_state.get("afsim_result"):
                st.info("AFSIM result is ready. Use the AFSIM panel below to open Mystic for the real replay.")
            elif st.session_state.get("sim_initialized"):
                st.caption("The web 3D view is a planning preview; the AFSIM/Mystic replay is the authoritative demo view.")
            if af is not None and sa:
                quality=st.session_state.get("display_quality","流畅")
                speed=float(cfg.get("speed",1.0))
                # 缓存签名: 推演帧令牌/画质/倍速不变时不重建图表
                # (令牌在每次推演帧计算时自增, 比id(af)对session_state序列化更稳健)
                sig=(st.session_state.get("anim_frames_token",0),quality,speed)
                if st.session_state.get("anim_fig_sig")!=sig:
                    with st.spinner("🎬 构建动画视图(展示帧降采样+历史轨迹)..."):
                        fig,meta=build_animated_figure(af,st.session_state.get("positions",[]),
                            quality=quality,speed=speed)
                        st.session_state.anim_fig=fig; st.session_state.anim_fig_sig=sig
                        st.session_state.anim_fig_meta=meta
                st.plotly_chart(st.session_state.anim_fig,use_container_width=True)
                _render_perf_panel(st.session_state.get("anim_fig_meta"))
            else:
                fig=build_figure(st.session_state.get("missiles",[]),st.session_state.get("positions",[]),
                    st.session_state.get("interceptors",[]),st.session_state.get("destroyed_markers",[]))
                st.plotly_chart(fig,use_container_width=True)
        stt_main=st.session_state.get("stats",SimulationStats())
        render_after_action_panel(stt_main, st.session_state.get("assessment_report", ""),
                                  st.session_state.get("sim_events", []))
    with right:
        stt=st.session_state.get("stats",SimulationStats())
        render_right(stt,st.session_state.get("assessment_report",""),st.session_state.get("sim_events",[]))
        cp=st.session_state.get("comparison_result")
        if cp: render_comparison(cp)
        mc=st.session_state.get("mc_result")
        if mc: render_mc(mc)

    validation = st.session_state.get("validation_result")
    if validation:
        with st.expander("🧪 基础验证结果", expanded=True):
            rows = []
            for row in validation:
                rows.append({"场景": row["case"], "结果": "通过" if row["passed"] else "失败",
                             "指标": json.dumps(row.get("metrics", {}), ensure_ascii=False),
                             "错误": row.get("error", "")})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    render_afsim_panel(cfg)

    if sa and af is None:
        with st.spinner("🔄 预计算推演帧..."):
            result=run_full_simulation(
                st.session_state.missiles,st.session_state.positions,
                st.session_state.assignments)
            if not isinstance(result, SimulationResult):
                raise RuntimeError(
                    f"接口不一致: 需要SimulationResult, 实际{type(result).__name__}。"
                    "请完全停止Streamlit进程后重新运行: python -m streamlit run app.py")
            st.session_state.anim_frames=result.frames; st.session_state.stats=result.stats
            st.session_state.sim_events=result.events; st.session_state.interceptors=result.interceptors
            st.session_state.destroyed_markers=result.frames[-1].get('des',[])
            st.session_state.anim_fig=None; st.session_state.anim_fig_sig=None; st.session_state.anim_fig_meta=None
            st.session_state.anim_frames_token=int(st.session_state.get("anim_frames_token",0))+1
            st.session_state.sim_finished=True
        # 生成LLM战后总结 + AI优化方案
        import json as _json
        from core.analysis import (build_analysis_context, build_local_after_action,
                                    build_local_optimization)
        from core.models import SimulationConfig
        _cfg=SimulationConfig()
        ctx=build_analysis_context(
            st.session_state.stats, st.session_state.positions,
            st.session_state.assignments, st.session_state.interceptors,
            st.session_state.sim_events, _cfg,
            mc_result=st.session_state.get("mc_result"),
            comparison_result=st.session_state.get("comparison_result"),
            scenario_name=cfg.get("scenario","normal"),
            algorithm=cfg.get("algo","greedy"), seed=cfg.get("seed",42))
        ctx_json=_json.dumps(ctx, ensure_ascii=False, indent=1)

        # 战后总结
        aar_sys="你是一名防空任务筹划、作战仿真与效能评估专家。请严格依据仿真数据撰写战后分析,不得虚构。区分数据事实与推断。输出Markdown, 600-1000字。"
        aar_prompt=f"""请依据以下仿真数据撰写结构化战后总结, 按以下6部分输出Markdown:
### 1. 总体结论 (场景/算法/参数/拦截率/命中率/总体评价/核心问题)
### 2. 交战过程分析 (发射时间线/命中/未命中/冗余/取消)
### 3. 资源使用分析 (弹药消耗率/阵地负载/效率)
### 4. 任务分配与算法表现 (任务完成/取消/分配合理性)
### 5. 主要问题和风险 (2-4项, 每项附数据依据)
### 6. 结论可信度 (单次/蒙特卡洛/可推广性/后续实验建议)
仿真数据:\n{ctx_json}"""
        local_aar=build_local_after_action(ctx)
        st.session_state.assessment_report=call_llm(aar_prompt, local_aar,
            system_prompt=aar_sys, max_tokens=1400, temperature=0.4)

        # 优化方案
        opt_sys="你是一名防空任务规划、运筹优化和仿真实验设计专家。请依据真实仿真数据提出可实施、可验证的优化方案。禁止笼统建议。每条需含证据/操作/预期/代价/验证。"
        opt_prompt=f"""请依据以下仿真数据撰写优化方案, 按P0/P1/P2优先级输出Markdown (800-1200字):
### 优化目标 (当前最需改进的指标)
### P0 立即处理 (1-2项, 每项含: 证据/操作/参数修改位置/预期/验证实验/验收指标)
### P1 方案优化 (算法对比/负载均衡/动态接管/补射/约束改进)
### P2 实验验证 (4场景×100次MC/灵敏度/共同种子/效应量)
### 推荐实施顺序
仿真数据:\n{ctx_json}"""
        local_opt=build_local_optimization(ctx)
        st.session_state.optimization_plan=call_llm(opt_prompt, local_opt,
            system_prompt=opt_sys, max_tokens=1600, temperature=0.4)

        st.session_state.report_source="AI生成" if os.getenv("DEEPSEEK_API_KEY","") else "本地规则生成"
        st.session_state.report_generated_at=datetime.now().strftime("%H:%M:%S")
        st.session_state.report_signature=f"{cfg.get('seed',0)}_{cfg.get('algo','')}_{st.session_state.stats.intercepted}"
        st.rerun()
    if not st.session_state.get("sim_initialized"): st.info("👈 左侧配置 → 点🎲初始化")

if __name__=="__main__": main()
