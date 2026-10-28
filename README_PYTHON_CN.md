# WaveDispX Python/C++ 0.2

推荐交互入口现为 **[WaveDispX Studio](docs/studio/README_zh.md)**：普通Qt窗口、源台/模型/结果同屏、可选R/L与HVSR、CPU workers及自定义CSV。运行`sh scripts/launch_studio.sh`（已有seisloc环境），或在配置好后端的源码环境中运行`python -m wavedispx.desktop.studio`。新`experiment`产品与旧分阶段/CLI流程有明确边界，不支持的组合会报错；下文旧网页/Notebook仍保留，不代表已同步全部新产品。

当前操作入口：[新版完整操作手册（HTML）](docs/WaveDispX_USER_MANUAL_CN.html)。
覆盖自动填参/重置、手动 f/v/df/dv、主动记录时长、被动 lag、line/array/CSV、DAS、
完整窗口截图及本轮主动/被动实际结果。2026-09-17 全量回归：251 项通过；科学验证边界见证据登记。

WaveDispX 是本目录原 MATLAB 主动源/被动源面波程序的高效、可复现版本。MATLAB 保留为
数值基准；日常算例由 Python 管理配置、噪声、互相关、频散、拾取、理论曲线与输出，耗时
最大的离散波数法（DWM）和加速径向变换由多线程 C++17 完成。

当前稳定范围：水平层状各向同性弹性/黏弹性模型、规则水平线阵、参数化二维 grid 或 CSV
输入的任意二维地表接收阵列、主动源五分量 Green 函数与 RR/ZR/RZ/ZZ 联合 Rayleigh、TT Love
频散，以及被动竖直快速路径或物理三分量随机力的真实混合记录。三分量被动流程输出按台站对方向
旋转的 RR/TT/ZZ NCF 与联合 Rayleigh 频散。二维横向非均匀体、地形和方位各向异性反演不在
0.2 的稳定范围内。

受控 active/passive 多算法比较现覆盖 PS、FJ、FH、FK/Bartlett、τ–p、L2/稀疏
Radon、Capon/MVDR、MUSIC、SPAC 和 eSPAC，并用同一两层模型在 1–24 Hz 检查
inline、off-axis、passive SPZ、均匀方位和非 SPZ 场景。模型、观测系统、噪声源
分布及全部 5 × 11 频散能量结果可输出为一张总图；详见
[Active / passive 多方法频散能量基准](docs/DISPERSION_METHOD_BENCHMARK_CN.md)。

## NCF–EGF 筛选研究主线

噪声源方位分布、短时 CCF 选择/加权、NNWS/RMSR 与 NCF–EGF 等价性实验的唯一执行基线是
[双台 NCF 时间窗筛选与 EGF 等价性主推进计划](docs/NCF_EGF_EQUIVALENCE_PLAN_CN.md)。
FTAN 真值审计、Xie/Fan 作者尺度机制复现和 Gate 2 v5.0 单一 1-D 模型五种子盲测已经完成。
v5.0 的基线与方向污染均有效，但 RMSR、robust、temporal 单台对适配和原版 NNWS 没有任何一个
通过预注册的波形/相位/precursor/FTAN/匹配置乱联合判据，因此主线暂不进入真实资料 efficacy
验证。完整负结果与下一步物理约束路线见
[Gate 2 v5.0 稀疏事件盲测结果](docs/GATE2_SPARSE_EVENT_RESULTS_CN.md)。旧 V3
virtual-source subarray/FJ-PS 协议仍暂停；不得仅凭 SNR、自动拾取或更亮的频散图宣称恢复了
EGF。任何计划变更必须先同步更新主计划及其变更记录。

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

界面支持主动/被动模板、完整 TOML、线阵、二维面阵、统一 DAS 和逐道转弯/变标距 DAS 光缆 CSV、接收器/Shot/噪声源 CSV、PSD CSV、模型表格、
输入图与混叠检查、分阶段运行、一键运行和结果下载。详细操作见
[完整 HTML 界面与案例手册](docs/WaveDispX_UI_MANUAL_CN.html)包含主动/被动 × 线阵/面阵四套已运行案例、完整参数页和结果图；
[Markdown 快速索引](docs/UI_GUIDE_CN.md)适合在代码浏览器中查看。界面只组织输入和显示结果，所有按钮仍调用与命令行相同的
`WaveDispXWorkflow`，没有独立的简化正演。

Notebook 现在按任务拆开，避免在同一本里混合所有分支：

- [主动源快速工作流](notebooks/01_active_quickstart.ipynb)；
- [被动源真实有限记录工作流](notebooks/02_passive_quickstart.ipynb)；
- [方法、公式与适用边界](notebooks/03_methods_and_validation.ipynb)；
- [DAS 有限标距主动/被动工作流](notebooks/04_das_active_passive.ipynb)；
- [入口导航](notebooks/WaveDispX_workflow.ipynb)。

前两本把模型、line/grid/file receiver、Shot/噪声源/PSD 文件、源时间函数、频散、拾取和性能
参数全部集中在一个 `PARAMETERS` Cell，并自动显示中文参数路径、单位、约束和调整影响；也可令
`INPUT_TOML` 指向完整配置。最后一个 Cell 的 `RUN_ALL_NOW` 是一键全流程开关。required 失败会
逐项显示当前值、判据和怎么改。只改拾取或频散参数时保留安全的上游缓存；改模型、几何或
同名上传文件内容时，会按文件内容哈希正确作废受影响阶段。

正演核心是频率—波数域 DWM，不是逐步时间推进；但当前主动/被动生产频散图分别来自有限炮集和
有限窗混合噪声 NCF，所以时长仍是物理参数。主动 `record_duration_s=0` 时按理论群速度、最大偏移和
最低处理频率自动设计；被动 `window_s` 控制传播时延与真实频率分辨率，`duration_s` 控制随机统计
收敛。预检查会在用户裁剪窗之外计算源带内多模态理论曲线，并给出模型、源谱、孔径、道距及 DAS
标距共同约束的 f-v 建议范围；`zero_padding` 只加密显示，不能替代真实记录时长。

[纯开算前审核 Notebook](notebooks/preflight_review.ipynb) 仍保留给只想快速检查输入、完全
不运行波场正演的场景。

本机可直接用已安装的 VS Code Jupyter 扩展打开：

```bash
code notebooks/WaveDispX_workflow.ipynb
```

Notebook 已固定使用 `Python (seisloc)` 内核；DAS Notebook 会显式把本项目 `python_src` 放到
导入路径首位，避免本机另一份 editable WaveDispX 被误用。DAS 完整入口已自顶向下执行并保留输出。要重新生成
或验收 Notebook：

```bash
conda run -n seisloc python scripts/build_notebooks.py
conda run -n seisloc python scripts/execute_notebooks.py
```

```bash
# 主动源：五分量炮集 + FJ/Hankel Rayleigh/Love 频散
conda run -n seisloc wavedispx run examples/active_two_layer.toml

# DAS：主动二维面阵；逐道转弯/变标距光缆；被动逐道光缆
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/active_das_grid_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/active_das_channel_file_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/passive_das_channel_file_quick.toml

# 600 m 单台站对：主动 ZZ Green 函数的群/相速度 FTAN 真值验收
conda run -n seisloc python scripts/plot_active_pair_ftan_qc.py

# 把完整 QEDispInv 包复制到项目的 Git 忽略工具目录（首次使用）
conda run -n seisloc python scripts/install_qedispinv_local.py \
  /path/to/QEDispInv-main

# 低速夹层、多模态、mode-kissing、漏根与 Vs 灵敏度核对照
conda run -n seisloc python scripts/plot_qedisp_low_velocity_benchmark.py

# 被动源：384 个独立噪声源混合 + 1800 s 有限记录 NCF + phase-shift
conda run -n seisloc wavedispx run examples/passive_two_layer.toml

# 已知源 Oracle 筛选 quick：同一噪声实现并行累计 baseline、hard/soft 与公平随机对照
conda run -n seisloc wavedispx run examples/passive_source_selection_quick.toml

# 预注册时变各向异性源活动表：验证完整 EGF 的 isotropic_egf 目标
conda run -n seisloc wavedispx run \
  examples/passive_source_selection_anisotropic_quick.toml

# Fan 等（2026）逐台站对 NNWS 研究算例（默认流程仍关闭）
conda run -n seisloc wavedispx run examples/passive_nnws_pairwise_quick.toml

# 60 窗时变各向异性源 benchmark：稳定性消融 / 原作者精确损失
conda run -n seisloc wavedispx run examples/passive_nnws_anisotropic_benchmark.toml

# 600 m 单虚拟源—多接收点已知真值基准
conda run -n seisloc wavedispx run examples/passive_nnws_virtual_source_600m_benchmark.toml

# NNWS-v2：连续时间块 OOF、全道集共享权重、最低 Neff 约束
conda run -n seisloc wavedispx run \
  examples/passive_nnws_v2_virtual_source_600m_benchmark.toml

# 已知真源 pair×frequency SPZ 上限实验，并与 NNWS-v2 组合
conda run -n seisloc wavedispx run \
  examples/passive_nnws_v2_pair_spz_600m_benchmark.toml

# 216 源方位加密基准：数据驱动 tau-p 双向不平衡筛选
conda run -n seisloc python scripts/generate_directional_density_source_case.py
conda run -n seisloc wavedispx run \
  examples/passive_nnws_v2_directional_density_1000m.toml

# 五个固定参数的未见噪声种子 + 波形/道集/频散/多指标 QC
conda run -n seisloc python scripts/run_nnws_seed_benchmark.py \
  --config examples/passive_nnws_v2_directional_density_1000m.toml \
  --output-root output_python/directional_density_seed_benchmark \
  --seeds 314272:314277
conda run -n seisloc python scripts/plot_directional_density_qc.py \
  --run-directory output_python/passive_nnws_v2_directional_density_1000m \
  --output-directory \
  output_python/passive_nnws_v2_directional_density_1000m/directional_density_qc \
  --seed-root output_python/directional_density_seed_benchmark

# USArray TA.A12A--TA.A17A 非重叠时间段实际数据审计
conda run -n seisloc python scripts/run_nnws_usarray_pair_validation.py
conda run -n seisloc wavedispx run \
  examples/passive_nnws_anisotropic_upstream_benchmark.toml

# 高阶模态基准：主动五分量 + 被动 192 个物理 3C 源/1800 s + 三方法对照
conda run -n seisloc wavedispx run examples/active_high_modes_3c.toml
conda run -n seisloc wavedispx run examples/passive_high_modes_3c.toml

# 只生成观测系统坐标，便于运行前检查
conda run -n seisloc wavedispx geometry examples/passive_two_layer.toml \
  --output output_python/geometry_check
```

双台 FTAN 已独立于阵列 `f-v` 成像实现，支持群速度包络、时间域/频率域相速度、`T/8` 远场校正、
逐周期 SNR 与相速度周跳锚定。公式、主动源误差、MATLAB 示例对照及实际 NCF 使用边界见
[双台 FTAN 实现与验收](docs/PAIR_FTAN_VALIDATION_CN.md)。

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

启用 `[source_selection] enabled=true` 后，程序会从正演中已知的逐窗源系数计算线阵轴向
stationary-phase/endfire 功率、双向平衡、非轴向功率、方位熵和有效源数。它只在 NCF 进入
流式累计器时施加权重；互相关数值核仍只读取混合接收波形。一次噪声实现并行产生：

- `baseline_linear`：全部窗单位权重；
- `oracle_hard_mask` 与同保留窗数的 `random_budget_control`；
- `oracle_soft_weight` 与权重置换后严格同 `N_eff` 的 `random_neff_control`。

该方法读取正演真源，因此是方法上限而不是可直接用于实测资料的选择器；当前范围明确标记为
`array_axis_approximation`，只允许线阵。二维台阵需先实现 pair×frequency×azimuth NCF 才能
作台站对特异的方向筛选。

Fan 等（2026）的 NNWS v1.0 核心源码与可选 PyTorch 适配器已经以默认关闭的研究模式接入。
启用 `[nnws] enabled=true` 后，竖直分量流程会缓存逐窗、逐台站对的正/负/对称 NCF，在每个
台站对上独立训练权重，再按距离分箱进入同一套频散与主动参考 EGF 评价。程序先依据到时窗与
远端噪声窗判定几何合格台站对；近距不合格对不算训练失败。NNWS、单位权重、随机置换同
`N_eff` 及 Oracle/随机源筛选均可在完全相同的台站对支持上比较。失败台站对不会回退到线性栈；
逐窗×台站对权重、资格原因、`N_eff`、训练指标和失败原因保存在
`results.h5/nnws_pairwise` 与 `nnws_pair_audit.csv`。上游精确损失和显式数值稳定性消融由
`nnws.std_mode` 区分，不能混报。
另有明确命名的 `nnws_hybrid_full_aperture` 与
`nnws_hybrid_random_neff_control`：只对几何合格且训练成功的远距对使用 NNWS/置换权重，对
NNWS 无法定义 pre-signal 窗的近距对保留单位线性权重，从而恢复完整频散孔径。它们是独立研究
策略，不会替换严格共同支持结果，也不能隐藏训练失败或覆盖率不足。
来源、配置、论文数值复现差异和完整结果见
[NNWS 上游接入说明](docs/NNWS_UPSTREAM_INTEGRATION_CN.md)。作者四套常速度合成数据的
linear/RMSR/NNWS/selective、retained/rejected/effective sources、多初始化和参考速度敏感性见
[NNWS Gate 1 复现审计](docs/NNWS_GATE1_REPRODUCTION_CN.md)。

独立实验模块 `nnws_v2` 不修改作者 v1 代码。它用连续时间块交叉拟合产生逐窗 OOF
权重，把同一条时间权重应用于完整虚拟源道集，并显式约束最低 `N_eff`。输出
`nnws_v2_shared_oof` 与精确保留权重集合的 `nnws_v2_random_neff_control`，同时保存折号、
代理 CCF 与每折损失。当前 5 个 600 m 已知真值实现显示波形指标多数改善，但频散歧义和
拾取并不稳定；USArray 时间外门槛仍失败。完整结果见
[NNWS 可靠性审计](docs/NNWS_RELIABILITY_AUDIT_CN.md)。

`pair_spz` 是另一个默认关闭、范围更窄的上限实验。它对每个接收器对和频率计算
stationary-phase 核，用正演中已知的逐窗真源功率优化一条全道集共享权重；因此不能直接用于
实测数据。单独输出 `oracle_pair_spz` 及同 `N_eff` 置乱对照；与 `nnws_v2` 同时启用时，
用等权 log-space 几何平均组合两条权重并再次执行最低 `N_eff` 约束，输出
`nnws_v2_spz_constrained` 及置乱 SPZ 对照。5 个噪声实现中，组合方法的频散歧义比例和拾取
MAE 均为 5/5 优于线性及置乱对照，但完整波形指标存在代价、理论 EGF 频散自身也未通过
干净脊线门槛，所以当前完整 EGF 门和频散专项门均失败；不能称为已恢复真实 EGF。

进一步的 1000 m 干净理论基准把共享权重、台站对数据代理、台站对逐频真源 SPZ 与
“直接用理论 EGF 拟合权重”的监督上限放在同一套波形和频散指标中。现有可部署代理与真源
SPZ 都没有改善完整频散主脊；只有明确带真值泄漏的监督上限在 5 个独立噪声种子上同时改善
全部 7 项指标。监督权重在不同噪声种子间的细粒度相关仅约 0.13，表明上限包含明显的
逐实现过拟合。新增的不读真值连续块交叉拟合 tau--p/SPZ 代理虽改善波形和前到时能量，仍使
频散主脊误差变大。当前只证明台站对特异、正负支分离的结构具有表达能力，尚未得到可用于实际
数据的可靠特征。结果、限制与数值表见
[NNWS 可靠性审计](docs/NNWS_RELIABILITY_AUDIT_CN.md)。

```bash
# 源分布/活动、窗口权重、三支 NCF–EGF 波形、道集、频散与严格有效性闸门
python scripts/plot_nnws_qc.py \
  --run-directory output_python/passive_nnws_anisotropic_benchmark

# v2 源场、共享权重、波形、道集、频散和 5 个噪声实现配对 QC
conda run -n seisloc python scripts/plot_nnws_v2_qc.py \
  --seed-summary-csv \
  output_python/nnws_v2_virtual_source_600m_seed_benchmark/seed_metrics.csv

# pair-SPZ 真源分布、权重、叠加波形、道集、频散、5 种子配对与双重验收门
conda run -n seisloc python scripts/plot_pair_spz_qc.py \
  --run-directory output_python/passive_nnws_v2_pair_spz_600m_benchmark \
  --seed-summary-csv \
  output_python/nnws_v2_pair_spz_600m_seed_benchmark/seed_metrics.csv

# 1000 m 架构审计：源活动、逐距波形、NCF 道集、频散图和 5 种子上限对照
conda run -n seisloc python scripts/plot_pairwise_egf_qc.py \
  --run-directory output_python/passive_nnws_v2_pair_spz_1000m_validation \
  --output-directory \
  output_python/passive_nnws_v2_pair_spz_1000m_validation/pairwise_qc \
  --seed-root \
  output_python/nnws_v2_pair_spz_1000m_pair_oracle_development

# 先用无噪声主动 EGF 检查观测孔径能否分辨目标频散
python scripts/scan_egf_dispersion_aperture.py
```

`source_selection.objective` 必须按科学终点预先声明：`stationary_phase` 偏好正反端射/SPZ
照明，适合定向传播与相位/频散实验；`isotropic_egf` 偏好高方位熵并保留双向平衡，用于完整
Green 恢复。二者不能混用。`ambient_sources.activity_file` 可用长表
`window_index,source_index,amplitude_multiplier` 预先定义时变各向异性源场；未列出的源为 0。

## 每次输出

每个结果目录包含原始/解析配置、模型与源—检波器 CSV、HDF5 数值结果、自动拾取、理论
曲线、逐模态理论曲线能量支持指标、耗时/峰值内存、被动 60/600/最终时长收敛表，以及一张同时含
`Vp`、`Vs`、密度、观测系统和频散结果的 `summary.png`。多分量运行另存
`active_components.png` 或 `passive_components.png`；`processing.compare_methods=true` 时另存
phase-shift / Bessel(FJ) / Hankel 对照图。
Poisson-Ricker 被动源还会输出逐事件的 `noise_events.csv`；完整文件观测系统示例见
`examples/observation_system/`。

Oracle 运行还会输出 `source_selection_windows.csv`，逐窗保留起止时间、每频带评分、mask 与
权重；`results.h5` 保存主动 interreceiver EGF、NCF 正/负/对称三支、每策略
`sum_weight/sum_weight_squared/N_eff/sufficient`、逐策略频散图及 EGF 指标。NCF 与主动参考
比较前使用干涉理论的 `−∂/∂t` 校准，不允许最佳时移。当前绝对振幅比例仍未标定，特别是
whitening/one-bit 会破坏物理振幅，因此 HDF5 会明确记录
`absolute_amplitude_available=false`。

NNWS 运行另存 `nnws_pair_audit.csv`。HDF5 中的 `nnws_pairwise/weights` 轴为
`[window_index, receiver_pair]`，与同组 `pair_i/pair_j`、窗口起止时间、信号窗、逐对
SNR/对称性/最终损失及失败原因一一对应；`ncf/strategies/nnws_pairwise` 是通过覆盖率门槛后
才允许评价的距离分箱结果。`qc/nnws_effectiveness_gate.json` 不允许仅凭 SNR、能量变亮或单一
EGF 指标宣称有效；当前 200 m benchmark 的共同支持和全孔径混合结果均未通过整体闸门。
