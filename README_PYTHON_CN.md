# WaveDispX Python/C++ 0.2

WaveDispX 是本目录原 MATLAB 主动源/被动源面波程序的高效、可复现版本。MATLAB 保留为
数值基准；日常算例由 Python 管理配置、噪声、互相关、频散、拾取、理论曲线与输出，耗时
最大的离散波数法（DWM）和加速径向变换由多线程 C++17 完成。

当前稳定范围：水平层状各向同性弹性/黏弹性模型、规则水平线阵、参数化二维 grid 或 CSV
输入的任意二维地表接收阵列、主动源五分量 Green 函数与 RR/ZR/RZ/ZZ 联合 Rayleigh、TT Love
频散，以及被动竖直快速路径或物理三分量随机力的真实混合记录。三分量被动流程输出按台站对方向
旋转的 RR/TT/ZZ NCF 与联合 Rayleigh 频散。二维横向非均匀体、地形和方位各向异性反演不在
0.2 的稳定范围内。

## 安装与验证

本项目按本机 `seisloc` Conda 环境验收：

```bash
conda run -n seisloc python -m pip install -e . --no-build-isolation
conda run -n seisloc wavedispx validate examples/active_two_layer.toml
conda run -n seisloc wavedispx validate examples/passive_two_layer.toml
conda run -n seisloc pytest -q
```

`pip install -e` 会编译 `cpp/dwm_core.cpp`。不要把依赖安装到 base 环境。

## 运行

日常使用推荐启动本地界面：

```bash
conda run -n seisloc wavedispx-ui
```

新增的原生 PySide6 桌面工作台不依赖浏览器，也不要求封装成 `.app`：

```bash
conda run -n seisloc wavedispx-desktop
# 或直接从源码入口启动
conda run -n seisloc python -m wavedispx.desktop
```

它提供集中可视参数树与完整 TOML 双向同步、模型层增删、line/grid/file 接收器、噪声源环/源线/CSV、
Shot/PSD CSV、六联输入审核图、后台分阶段运行、日志、结果图廊与输出目录浏览。桌面版操作见
[原生桌面工作台手册](docs/DESKTOP_UI_GUIDE_CN.md)。

界面支持主动/被动模板、完整 TOML、线阵、二维面阵、接收器/Shot/噪声源 CSV、PSD CSV、模型表格、
输入图与混叠检查、分阶段运行、一键运行和结果下载。详细操作见
[完整 HTML 界面与案例手册](docs/WaveDispX_UI_MANUAL_CN.html)包含主动/被动 × 线阵/面阵四套已运行案例、完整参数页和结果图；
[Markdown 快速索引](docs/UI_GUIDE_CN.md)适合在代码浏览器中查看。界面只组织输入和显示结果，所有按钮仍调用与命令行相同的
`WaveDispXWorkflow`，没有独立的简化正演。

Notebook 现在按任务拆开，避免在同一本里混合所有分支：

- [主动源快速工作流](notebooks/01_active_quickstart.ipynb)；
- [被动源真实有限记录工作流](notebooks/02_passive_quickstart.ipynb)；
- [方法、公式与适用边界](notebooks/03_methods_and_validation.ipynb)；
- [入口导航](notebooks/WaveDispX_workflow.ipynb)。

前两本把模型、line/grid/file receiver、Shot/噪声源/PSD 文件、源时间函数、频散、拾取和性能
参数全部集中在一个 `PARAMETERS` Cell，并自动显示中文参数路径、单位、约束和调整影响；也可令
`INPUT_TOML` 指向完整配置。最后一个 Cell 的 `RUN_ALL_NOW` 是一键全流程开关。required 失败会
逐项显示当前值、判据和怎么改。只改拾取或频散参数时保留安全的上游缓存；改模型、几何或
同名上传文件内容时，会按文件内容哈希正确作废受影响阶段。

[纯开算前审核 Notebook](notebooks/preflight_review.ipynb) 仍保留给只想快速检查输入、完全
不运行波场正演的场景。

本机可直接用已安装的 VS Code Jupyter 扩展打开：

```bash
code notebooks/WaveDispX_workflow.ipynb
```

Notebook 已固定使用 `Python (seisloc)` 内核，四本入口均经过自顶向下执行验证。要重新生成
或验收 Notebook：

```bash
conda run -n seisloc python scripts/build_notebooks.py
conda run -n seisloc python scripts/execute_notebooks.py
```

```bash
# 主动源：五分量炮集 + FJ/Hankel Rayleigh/Love 频散
conda run -n seisloc wavedispx run examples/active_two_layer.toml

# 被动源：384 个独立噪声源混合 + 1800 s 有限记录 NCF + phase-shift
conda run -n seisloc wavedispx run examples/passive_two_layer.toml

# 高阶模态基准：主动五分量 + 被动 192 个物理 3C 源/1800 s + 三方法对照
conda run -n seisloc wavedispx run examples/active_high_modes_3c.toml
conda run -n seisloc wavedispx run examples/passive_high_modes_3c.toml

# 只生成观测系统坐标，便于运行前检查
conda run -n seisloc wavedispx geometry examples/passive_two_layer.toml \
  --output output_python/geometry_check
```

不规则二维接收器示例为 `examples/active_irregular_quick.toml` 和
`examples/passive_irregular_quick.toml`，坐标均来自 `examples/receivers_irregular.csv`。
主动源可以直接对每个 `(x,y)` 接收点正演；被动源会先
对所有接收器对做互相关，再按台间距分箱。二维方位在当前径向频散图中被平均掉，因此它
适用于一维层状、近似各向同性照明；要分析方位各向异性，需增加方位分箱流程。
规则面阵列示例见 `examples/passive_grid_quick.toml`。

完整输入说明见 [输入参数手册](docs/INPUT_PARAMETERS_CN.md)，数值与性能结果见
[迁移验收报告](docs/MIGRATION_STATUS_CN.md)。把正演数据交给独立频散项目时，直接使用
[正演算法移交说明](docs/FORWARD_HANDOFF_CN.md) 中的 HDF5 路径、分量约定和最小移交包。

## 被动处理是否“真实”

是。每个源先生成有限时长随机频谱，乘离散波数法计算的 Green 函数传播到每个接收器，所有源在接收器
处相加成一份真实混合记录；随后互相关函数只接收 `time × receiver` 或
`time × receiver × (X,Y,Z)` 波形，不接受源坐标、源编号或理论曲线。程序没有逐源内部
互相关后再叠加的捷径。

`ambient_sources.component="three_component"` 时，每个源产生独立全局 `Fx/Fy/Fz` 随机力。
程序按每个源—接收器方位把它分解为径向/切向载荷，用完整 `RR/TT/ZR/RZ/ZZ` Green 张量传播，
再旋回接收器全局 X/Y/Z。任意二维阵列计算 NCF 时，又按每一个台站对方位把 X/Y 旋成 RR/TT；
因此这不是把三张单分量结果在频散图上人为相加。

为节省内存，20 s 独立窗生成后立即做接收器互相关并叠加，默认只保存前 60 s 混合波形。
这与把全部 1800 s 常驻内存后按相同窗处理数值等价，但内存更稳定。

## 每次输出

每个结果目录包含原始/解析配置、模型与源—检波器 CSV、HDF5 数值结果、自动拾取、理论
曲线、逐模态理论曲线能量支持指标、耗时/峰值内存、被动 60/600/最终时长收敛表，以及一张同时含
`Vp`、`Vs`、密度、观测系统和频散结果的 `summary.png`。多分量运行另存
`active_components.png` 或 `passive_components.png`；`processing.compare_methods=true` 时另存
phase-shift / Bessel(FJ) / Hankel 对照图。
Poisson-Ricker 被动源还会输出逐事件的 `noise_events.csv`；完整文件观测系统示例见
`examples/observation_system/`。
