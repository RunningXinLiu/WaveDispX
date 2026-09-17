# 2026-09-17 操作流程收尾记录

收尾范围：WaveDispX 本地桌面、网页和共用 Python/C++ 工作流的可操作版本。
当前主操作入口为 `WaveDispX_USER_MANUAL_CN.html`，原手册保留历史布局案例并链接到新版。

## 用户操作

- 自动建议 f/v/df/dv、主动记录时长或被动窗长/lag；建议值可见、可逐项覆盖。
- 区分重新计算当前模型建议、恢复模板默认、完整载入模板。
- 显式显示频率步长、主动真实记录时长、被动最大 lag，中文悬停帮助。
- line/grid/file 和逐道 DAS、模型/源谱/观测系统审核、阶段计算、结果保存。
- 新手册完整说明源位置、PSD、Poisson事件时间、接收器和炮点文件、输出和复现。

## 回归证据

`PYTHONPATH=python_src conda run -n seisloc pytest -q --tb=short`：
**251 passed in 42.90s**。原有已跟踪修改的 `git diff --check` 无错误；
纳入此前未跟踪研究文件后，暂存检查提示11处文件末尾空行（含上游NNWS源码/许可）。
这些为格式提示，保留上游文件原貌，未为消除提示改写第三方源码。

网页 computer use 已操作自动填参、手动覆盖、重置及 active/passive 完整运行。
主动 FFT 周期3.5 s、末样点3.495 s、显示df=0.05 Hz；被动lag=2.5 s、显示df=0.1 Hz，
实际HDF5轴与元数据核对一致。具体哈希见 `forward_workflow/EVIDENCE_REGISTER.md`。
桌面通过Qt测试和完整窗口截图；原生 computer use 未识别Conda Python进程，原生人工点击验收未完成。

修复被动保存变量引用、自动填入新频带后的时长重算、正演缓存依赖，以及桌面工具栏拥挤。

## 同步范围及边界

同步本工作树代码、相关研究模块、测试、示例、Notebook、文档和精选截图；
不把研究模块打包进来解释成效果通过。NNWS/SPZ/窗口筛选研究的结论以各自登记为准。
本地生成的 `output_python/`、编译缓存、临时模块和个人机器配置不纳入本次Git快照。

收尾不是跨模型精度或被动收敛认证。复杂模型、噪声时长、DAS NCF等价性、独立求解器对照
仍按 `forward_workflow/WORKFLOW_zh.md` 的具体物理输入和观测算子逐项验证。
