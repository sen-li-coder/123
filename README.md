# 多对多协同防空任务筹划与推演系统 v5.0

模块化架构 · 真正多对多分配 · PN比例导引 · 统计检验 · LLM约束校验 · CSV导出

## 研究问题
> 在有限拦截资源条件下，多阵地如何协同分配拦截任务以最大化拦截成功率？

## 系统架构
```
├─ app.py                     # Streamlit入口
├─ core/                      # 仿真核心(零UI依赖)
│  ├─ constants.py, models.py, math_utils.py
│  ├─ simulation.py, guidance.py, threat.py
│  ├─ scenario.py, validation.py
├─ algorithms/                # 分配算法
│  ├─ base.py, random_strategy.py
│  ├─ nearest.py, greedy.py, optimized.py
├─ experiments/               # 实验与统计
│  ├─ monte_carlo.py, statistics.py
├─ services/                  # LLM服务
│  └─ llm_service.py
├─ tests/                     # 单元测试
└─ README.md, requirements.txt
```

## 安装与运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 多对多分配模型
- **random**: 随机分配
- **nearest**: 最近距离优先
- **greedy**: 威胁度贪心
- **optimized**: SciPy Hungarian算法最小化总代价

约束：弹药/射程/制导通道/发射冷却/在途拦截弹

## 威胁评估公式
```
threat = 0.2×speed + 0.2×distance + 0.3×TTI + 0.1×RCS + 0.2×terminal
```
可配置、归一化、可解释

## 统计方法
- 拦截率均值 ± 95%CI (ddof=1)
- 配对t检验 + Cohen's d
- 灵敏度分析(共同随机数)
- CSV数据导出

## LLM校验管道
LLM候选方案 → 确定性解析 → 约束检查 → 通过/拒绝 → 记录

## 实验复现
```bash
# 固定种子推演
seed=42 → 初始化 → 推演 → 记录拦截率
# 蒙特卡洛
🔬蒙特卡洛 → N=100 → 导出CSV
```

## 已知限制
- 仿真步长固定100ms
- 导弹匀速直线(末端蛇形机动)
- 拦截弹PN制导(最大加速度限制)
