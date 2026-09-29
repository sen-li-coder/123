# AFSIM 联动使用说明

## 程序分工

- Streamlit/Python：设置想定、运行任务分配算法、导出交换数据、读取结果和生成指标。
- `mission.exe`：以批处理方式运行 AFSIM 场景并生成 `.aer`、`.csv`、`.evt` 文件。
- `Warlock`：交互式 AFSIM 操作界面，适合观察和演示；它通常不会自动退出，因此不作为同步批处理引擎。
- `Mystic`：打开批处理生成的 `.aer` 文件，查看态势和交战动画。

## 操作步骤

1. 启动系统：

   ```powershell
   streamlit run app.py
   ```

2. 在左侧设置场景、目标数、防空阵地数、弹药量、分配算法和随机种子，点击“初始化”。
3. 在页面下方“AFSIM 联动演示”区域检查三个路径：
   - `D:\asfim\bin\warlock.exe`
   - `D:\asfim\bin\mission.exe`
   - `D:\asfim\bin\mystic.exe`
4. 点击“导出场景”。系统会在项目目录的 `_afsim_work` 下生成：
   - `scenario_exchange.json`：Python 与 AFSIM 的统一交换数据；
   - `air_defense_demo.txt`：本次生成的 AFSIM 输入文件。
5. 点击“运行 Warlock”。界面实际调用 `mission.exe` 批处理运行，成功后生成：
   - `warlock_output/air_defense_demo.aer`：Mystic 回放文件；
   - `warlock_output/air_defense_demo.csv`：可解析事件；
   - `warlock_output/air_defense_demo.evt`：文本事件日志；
   - `warlock_output/warlock.log`：运行日志。
6. 点击“打开 Mystic”。Mystic 会加载 `.aer` 文件，可在其中调整视角、时间轴并录制演示视频。
7. 点击“读取结果”，查看 AFSIM 的发射、命中、脱靶、逃逸和拦截率，并与 Python 本地仿真对比。

## 重要说明

- 第一版使用 AFSIM 安装目录中的 `launcher` 示例类型模板，只读取模板，不修改 `D:\asfim` 文件。
- Python 的 Assignment 会保存到 `scenario_exchange.json` 和场景注释中；AFSIM 第一版由模板内的 SAM 战术处理器完成实际探测与发射。
- AFSIM 和 Python 的命中结果可能不同，应以统一场景编号、随机种子和输出日志进行对比，不能直接把两者混为同一套物理模型。
- 不要把受限制的 AFSIM 安装文件、示例和输出文件上传到公共仓库或公开平台。

## 常见问题

### Warlock 按钮超时

这是因为 Warlock 是交互式 GUI。请使用“运行 Warlock”按钮调用批处理 `mission.exe`，然后使用 Mystic 查看 `.aer`；需要实时交互时再单独打开 Warlock。

### 输出为空

检查 `_afsim_work\warlock_output\warlock.log`，确认 AFSIM 路径、输入文件语法和输出目录权限。界面会保留返回码和日志路径。

### 拦截率为 0

先点击“读取结果”，再查看事件时间线。如果只有探测/任务事件而没有 `WEAPON_FIRED` 或 `WEAPON_HIT`，说明当前目标没有进入模板武器的有效交战包线，需要调整目标初始位置、速度或演示时长。
