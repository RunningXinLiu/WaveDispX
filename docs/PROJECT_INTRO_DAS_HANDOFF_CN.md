# WaveDispX 项目介绍与 DAS 模拟接手说明

更新：2026-09-17。面向准备使用本项目、继续开发噪声源筛选算法，或接手 DAS 模拟的其他 chat / 开发者。

本文件依据当前工作目录的源码与研究记录编写。有限标距地表 DAS v1 已实现并实跑；井中/弯曲光纤、三分量随机源 DAS、DAS FJ/Hankel 核和 DAS–DAS 主动 EGF 参考仍未实现或未验证。不要把 v1 的 smoke/快速案例描述成所有 DAS 场景已验证。

## 1. 项目是什么

项目目录名为 `PassiveFW_versionstable`，Python 软件包名为 `wavedispx`，当前声明版本为 `0.2.0`。它由原 MATLAB 主动/被动面波程序发展而来，使用 Python 组织实验，使用 C++17 / pybind11 的离散波数法（DWM）计算层状介质 Green 函数。

它可以从速度模型、震源和观测几何生成主动源炮集或被动噪声混合记录，进一步计算噪声互相关函数（NCF）、频散能量、拾取结果和理论频散参考。当前的研究目的之一是：控制噪声源空间分布与时间活动，检验时间窗选择或加权能否改善 NCF 与经验 Green 函数（EGF）的关系。

当前物理范围是**一维水平分层、各向同性介质中的地表波场**，可设弹性或 Q 衰减。接收点支持水平线阵、二维规则网格及 CSV 指定的二维坐标；二维接收几何不意味着支持二维横向变化介质。DAS v1 支持统一标距/方位，也支持 CSV 逐道指定水平端点、方位和标距，从而表达离散的转弯或变标距地表光缆；接口仍没有接收深度、地形、弯曲段连续积分或光学解调模型，不能直接当作井中 DAS 或三维非均匀介质模拟器。

## 2. 接手前先确认运行的是哪一份代码

本次检查的工作目录：

```text
/Volumes/lx_exFAT/PassiveFW_versionstable
```

当前 HEAD 为 `a5b9c8d`，但目录内有大量已修改和未跟踪的代码、配置、研究文档，因此这个提交号不能唯一描述当前软件。移交时必须包含当前工作文件，不能只让另一个 chat checkout 该提交。保留原有改动，不要 reset/clean。

**2026-09-11 现场发现：**直接在 `seisloc` 中 `import wavedispx` 得到的是：

```text
/Volumes/lx_exFAT/WaveDispX-NCF-EGF-Lab/python_src/wavedispx/__init__.py
```

它是另一份目录。建议先用下面的方式限定源码路径，每次更换项目后核对 `__file__`：

```bash
cd /Volumes/lx_exFAT/PassiveFW_versionstable
env PYTHONPATH="$PWD/python_src" conda run -n seisloc python -c \
  'import wavedispx; from wavedispx import core; print(wavedispx.__file__); core._require_core()'
```

`PYTHONPATH` 只限定本次命令，不改变其他项目的安装关系。若 C++ 扩展缺失，需要在确认依赖后于目标环境编译本项目；常用安装命令为 `conda run -n seisloc python -m pip install -e . --no-build-isolation`，但 editable 安装会改变该环境默认指向的项目。本次没有重装。Python 默认使用 `seisloc`，不在 base 中安装依赖，新增依赖先与用户确认。

依赖以 `pyproject.toml` 为准：Python ≥3.11，核心为 NumPy、SciPy、h5py、disba；构建使用 setuptools、wheel、pybind11。Streamlit、PySide6、PyTorch、Notebook 工具和 stockwell 属于可选功能依赖。QEDispInv 是另行接入的本地理论求解器，不是 DWM 波场正演后端。

DWM 核在复频率—水平波数域求解，但当前交付的主动频散是有限炮集 FFT 成像，被动频散是有限窗
混合记录内部互相关后的 NCF 成像；它们不是省略时窗的“直接频率域能量”。主动保存时长、被动单窗
长度、被动总记录长度及 DWM synthetic period 已分别记录。详细定义和 HDF5 属性见
`docs/FORWARD_HANDOFF_CN.md` 的 1.1–1.2 节。

## 3. 代码入口与数据流

下表路径均相对于项目根目录，便于整体复制后继续使用。

| 位置 / 符号 | 用途与接手重点 |
|---|---|
| `python_src/wavedispx/config.py`：`RunConfig`、`load_config` | TOML 配置、类型与校验；含 `[das]` 标距、测量量和方位 |
| `python_src/wavedispx/das.py` | 端点几何、有限标距轴向应变/应变率、sinc 响应与 FFT 约定 |
| `geometry.py`：`receiver_positions`、`build_geometry` | 地表接收器和噪声源坐标；CSV 目前读取 `x_m,y_m` |
| `core.py`、`cpp/dwm_core.cpp` | Python 包装与 C++ DWM 数值核 |
| `forward.py`：`compute_full_fk_green`、`synthesize_full_receivers` | 五分量 f-k Green 与逆 Hankel 接收位移合成 |
| `active.py`：`simulate_active_shot` | 主动源 Green、源子波卷积和炮集 |
| `passive.py`：`build_passive_transfer` | 逐源—接收器传递响应、径向插值与误差记录 |
| `noise.py`：`synthesize_mixed_noise_window_3c_band` | 独立全局随机力经局部 Green 基传播、旋回全局 XYZ 并混合 |
| `passive.py`：`accumulate_passive_correlation` | 分窗合成、预处理、互相关、策略权重和累计；研究分支较多 |
| `processing/ncf.py`、`stacking.py` | 互谱/NCF、正负支、距离分箱与叠加 |
| `source_activity.py`、`source_selection.py`、`spz_selection.py` | 时变源活动与已知源方向/SPZ 筛选 |
| `nnws.py`、`nnws_v2.py`、`python_src/nnws/` | NNWS 接入与研究变体，不等于验证有效的生产筛选器 |
| `beamforming_selection.py`、`pair_tau_p_selection.py` | 阵列方位与 tau-p 选择相关组件 |
| `egf_reference.py`、`quality.py` | 主动台间 Green 参考、NCF 校准和质量评价 |
| `processing/dispersion.py`、`array_dispersion.py`、`ftan.py` | 阵列频散及双台群/相速度 FTAN |
| `processing/theory.py`、`qedisp.py` | disba / QEDisp 理论曲线，用于独立比较 |
| `artifacts.py` | HDF5、CSV、配置和运行元数据写出 |
| `workflow.py`：`WaveDispXWorkflow` | 分阶段接口与缓存，界面/Notebook 接入重点 |
| `cli.py`、`runner.py` | `validate`、`geometry`、`run` 命令与命令行运行组织 |
| `ui/`、`desktop/`、`app_service.py` | Streamlit、PySide6 与应用服务 |
| `examples/`、`python_tests/`、`scripts/` | 配置样例、测试、独立实验与 QC 脚本 |

`WaveDispXWorkflow` 提供 `from_toml → review_inputs → compute_forward → simulate_observations → compute_dispersion → save_results`，也可用 `run_all`。CLI 经 `runner.py` 组织运行，两者复用底层模拟/分析函数，但不能假定两条路径完全共用所有前置检查或缓存行为。

主动链路：层模型与源检距 → 五分量 Green → 主动子波卷积 → 炮集 → 频散/理论比较。

被动链路：层模型与源检几何 → 传递响应 → 每窗随机源实现 → **所有源先叠加成接收器混合记录** → 预处理和接收器互相关 → 时间窗加权/叠加 → NCF、EGF 和频散比较。源真值可以用于标记为 oracle 的诊断，但不能伪装成仅从观测数据得到的方法输入。

## 4. 最小使用方式

以下命令均在项目根目录执行，并显式指定本目录源码：

```bash
# 配置读取与校验，不运行波场正演
env PYTHONPATH="$PWD/python_src" conda run -n seisloc python -m wavedispx validate examples/active_two_layer.toml
env PYTHONPATH="$PWD/python_src" conda run -n seisloc python -m wavedispx validate examples/passive_irregular_quick.toml

# 最小被动二维几何案例：24 源、21 接收器、60 s / 10 s 独立窗
env PYTHONPATH="$PWD/python_src" conda run -n seisloc python -m wavedispx run examples/passive_irregular_quick.toml

# 原生桌面工作台
env PYTHONPATH="$PWD/python_src" conda run -n seisloc python -m wavedispx.desktop

# 后续改动时按范围运行测试；这里给出核心起点
env PYTHONPATH="$PWD/python_src" conda run -n seisloc python -m pytest -q \
  python_tests/test_cpp_core.py python_tests/test_active.py \
  python_tests/test_noise.py python_tests/test_processing.py
```

执行 `run` 前复制示例配置，并为 `[run].output_directory` 指定新的实验目录；输出文件使用固定名称，重复运行同一路径可能覆盖已有结果。路径通常相对于 TOML 所在目录解析。配置校验通过只证明输入可读取且通过该级校验，不是正演精度或 EGF 恢复通过。

其他起点：`examples/active_high_modes_3c.toml` / `passive_high_modes_3c.toml` 用于五分量/3C；`passive_source_selection_quick.toml` 用于已知源筛选；`handoff_active_qedisp_quick.toml` 和 `handoff_passive_qedisp_quick.toml` 依赖本地 QEDisp。标准配置的主要成像选项是 `phase_shift/bessel/hankel`；更多算法在专门的 benchmark 模块/脚本中，不能把 benchmark 方法名直接当作通用 TOML 选项。

## 5. 当前数组与输出约定

| 数据 | 轴与物理含义 |
|---|---|
| `ActiveShotGather.components` / `results.h5/shot/components` | `time × receiver × 5`，末轴 `RR,TT,ZR,RZ,ZZ` |
| 五分量命名 | 第一字母为接收位移，第二字母为源力方向；例如 RZ 为竖直力产生的径向位移 |
| `PassiveTransfer.spectrum` | 竖直模式 `frequency × receiver × source`；3C 模式再加五分量轴 |
| 3C 混合波形 / `preview/mixed_record` | `time × receiver × 3`，全局 `X,Y,Z`；竖直模式为二维 |
| `ncf/components/{zz,rr,tt}/gather` | `lag × distance_bin`，配套 `ncf/lag_time_s`、`distance_m`、`pair_count`；可用组取决于模式 |
| `dispersion/rayleigh_combined/energy` | `velocity × frequency`，配套同组坐标；其他能量图读取各自分组 |
| `theory/rayleigh_mps`、`love_mps` | `frequency × mode`，单位 m/s；缺根 `NaN` 不能当作零速度 |

主动五分量是随源—接收器方位定义的局部柱坐标 Green 基，并非同一个实体震源的五条独立传感器观测。构造固定全局方向的源时，必须先按源力和方位组合、旋转，再执行传感器响应。

波形时间单位 s，坐标 m，速度 m/s，密度 kg/m³。位移绝对幅值还依赖源归一化和数值约定，不能在未校准时宣称仪器级绝对幅值准确。标准频散图逐频归一化，不适合评价绝对振幅恢复；理论相速度标签也不等于 EGF 波形真值。

常见交付文件有 `input.toml`、`resolved_config.json`、`model.csv`、几何 CSV、`results.h5`、`manifest.json` 与结果图。预览波形只保存部分记录；不能把 `preview` 当作完整长时间连续数据。逐窗、逐台对和权重数据取决于开启的研究分支，接手时检查实际 HDF5 内容。按距离平均的 NCF 已丢失部分台对/方位信息，也不要重复乘入 `pair_count`。

## 6. 噪声源筛选研究已经知道什么

研究状态以 `docs/NCF_EGF_EQUIVALENCE_PLAN_CN.md` 及结果文档为依据；以下是已有记录摘要，本次未重跑验证：

- 当前双台研究曾在特定 600 m、单一层模型下验证 4–12 Hz 公共频带；这个范围不能自动继承到其他模型或 DAS。
- `GATE2_SPARSE_EVENT_RESULTS_CN.md`：v5.0 五个冻结种子的稀疏事件盲测完成，但没有可部署方法通过全部联合判据。
- `GATE2_SOURCE_AWARE_RESULTS_CN.md`：v5.1 辅助阵列的方位估计通过开发门，整窗角向均匀化未通过 EGF/FTAN 联合门。
- `GATE2_BRANCH_SPZ_RESULTS_CN.md`：v5.2 真值知情的分支 SPZ 筛选改善部分总体指标，但两支/相位/FTAN 联合门仍未通过，记录中的 beam 和后续盲测未运行。

因此，现有代码是实验基础，不能描述成“已经解决源分布引起的 EGF 偏差”。SNR 增加、对称性变好、频散图变亮都不足以单独证明恢复 EGF。oracle 读取真源或 active 参考，属于机制诊断/可达性分析；实际方法必须与它明确区分。

## 7. DAS v1：已加入的测量算子与剩余边界

已实现首版范围为**一维层状介质 + 地表直光纤 + 理想耦合 + 有限标距平均轴向应变/应变率**。它复用现有位移正演，并在主动/被动链路中于非线性预处理和互相关之前作用测量算子。井中、弯曲/起伏光纤、海底压力耦合和真实仪器传递函数分别是进一步扩展。

令直光纤单位切向为 `t`，通道中心为 `x`，标距为 `L`，位移为 `u`。理想有限标距应变为：

```text
eps_bar(x, time) = t · [u(x + L*t/2, time) - u(x - L*t/2, time)] / L
strain_rate      = d eps_bar / d time
```

这里应变单位为 1（m/m），应变率为 s⁻¹；若要输出 nε 或 nε/s，显式记录 10⁹ 的转换。通道间距与标距是不同参数，不能用相邻道距默认为标距。有限标距平均与方向响应的物理依据可参阅 [Näsholm 等，2022](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023587) 和 [Quantifying the Surface Strain Field…，2022](https://doi.org/10.3390/s22124589)。此处采用理想矩形标距模型，不含具体解调仪的脉冲/光学响应。

在上述假设下，我们建议直接在每个通道标距两端正演位移，使用同一源实现，并形成轴向差分。水平光纤需要水平位移；现有仅保留 ZZ 的被动快速路径不够。可复用完整 Green 基及全局坐标混合逻辑，但要特别注意：当前 `three_component` 会生成独立 Fx/Fy/Fz 源，若研究要求保持原先竖直力噪声场，应保留相同 Fz 事件，只使用 RZ 等响应得到水平运动，不能因需要水平接收位移而悄悄改变源机制。

标距算子是线性的，可作用于物理位移混合记录或已正确旋转的逐源响应；两种实现应以同一随机实现比较。它必须位于 one-bit、白化等非线性预处理之前。近距离端点差分可能放大插值误差，因此现有径向插值的位移误差门不能直接视为应变精度门。

弯曲光纤不能普遍套用固定切向的端点公式。需按弧长对局部 `tᵀ ε t` 积分，或采用经验证的分段模型；当前地表 XY 接口也不能通过新增一列深度就变成井中正演。

### 7.1 DAS 的 NCF 参考也必须改变

下面是从线性测量关系得到的设计约束。若通道观测为 `d_i = D_i u`，相关为双线性运算，则：

```text
C_d(i,j) = D_i C_u(i,j) D_j^T
```

因此 DAS–DAS 相关包含两个观测端的方向与标距作用，不能直接拿 ZZ 位移 Green 或只在接收端差分的点力→DAS 响应作同一真值。要先确定目标是点力→DAS 主动记录，还是与 DAS–DAS 干涉关系一致的广义响应，再构造相应参考。应变率还引入时间导数；频域互谱包含相应频率因子，需显式核对 FFT、互相关正负号和单位。

现有 `egf_reference.py` 记录了 NCF 的 `-d/dt` 校准约定，并明确没有建立绝对幅值常数。DAS 不应不经推导就照搬。噪声源方向性、光纤接收方向性和标距滤波也会同时作用，必须分别控制，避免把传感器响应的变化解释成筛选算法成功。

### 7.2 当前实现位置与接口

当前已实现的主路径如下；表中未列的研究分支不应自动视为兼容：

| 修改点 | 首版内容 |
|---|---|
| `config.py`、`das.py` | `[das]`、统一/逐道端点几何、有限标距算子、应变率和单位/时间约定 |
| `active.py` | 竖直点力的 RZ 端点水平位移→全局 XY→DAS；保留节点五分量对照 |
| `passive.py` | 竖直噪声点力的端点完整 Green→DAS 传递函数→随机混合记录→DAS–DAS NCF |
| `artifacts.py` | `/das` 几何/波形/元数据、`ncf/components/das`、`picked_das.csv` |
| `workflow.py`、`preflight.py` | DAS 缓存键、端点源距、第一标距零点、几何图与结果图 |
| `desktop/`、`ui/`、Notebook | 主动/被动线阵与面阵、逐道光缆模板、中文 hover tip、集中参数和一键运行 |
| `python_tests/test_das.py` | 刚体、均匀应变、单频应变率、第一零点和配置边界测试 |

当前输出契约为主动 `das/time_s`、`das/data`，被动预览 `das/time_s` 加 `das/strain` 或 `das/strain_rate`（`time × channel`），以及 `channel_id`、`channel_center_xy_m`、两端点、`tangent_xy`、`gauge_length_by_channel_m`、`fiber_azimuth_by_channel_deg`、算子版本、时间约定和单位属性。被动 DAS NCF 位于 `ncf/components/das/gather`。当前没有 z 坐标。

### 7.3 首版验收顺序

1. 解析算子：刚体平移应变为零，均匀轴向伸长得到规定应变；平面波验证方向、标距和应变率响应。
2. 数值算子：主动场端点直接正演与插值结果对照，检查空间/时间采样收敛，不能仅检查位移收敛。
3. 被动链路：相同噪声源实现，比较“先混合后测量”和“先测量后混合”，预处理在 DAS 算子之后。
4. DAS 相关基线：先验证理想源场与匹配参考，再引入方向偏置，最后测试筛选；保留位移对照和公平随机对照。
5. 性能与导出：测量道数/窗数增长的内存成本；输出含通道映射、标距、种子、配置与失败信息的小型可重现实例。

DAS 道数可能远大于现有算例；全道对相关为 O(N²)，不要一开始就保存全部“窗 × 台对 × lag”数组。先用小道数或明确的目标通道对完成物理验证。

## 8. 给下一位 chat 的接手提示词

可直接复制以下内容：

> 请接手 `/Volumes/lx_exFAT/PassiveFW_versionstable` 的 WaveDispX 项目。先读 `docs/PROJECT_INTRO_DAS_HANDOFF_CN.md`、`docs/forward_workflow/EVIDENCE_REGISTER.md` 和 `notebooks/04_das_active_passive.ipynb`。项目为 Python/C++ DWM 层状介质主动/被动模拟，已实现地表水平有限标距 DAS（应变/应变率、竖直力、phase-shift），支持规则 line/grid 的平行标距以及逐通道 CSV 端点/标距/方位的折线光缆。先核对 `wavedispx.__file__`，因为 `seisloc` 默认 editable 安装指向旁边项目；保留当前未提交文件。下一步若研究 DAS NCF/EGF，优先构造同物理量的 DAS–DAS 主动参考并验证双端测量算子，不要拿节点 ZZ 参考替代。三分量随机源、FJ/Hankel DAS 核、连续弯曲/井中光纤及 NNWS/SPZ+DAS 仍未验证。已有 NNWS/SPZ 负结果也不能当作已验证有效的方法。

## 9. 阅读索引与本次验证范围

- `README_PYTHON_CN.md`：软件能力、案例与运行入口。
- `docs/INPUT_PARAMETERS_CN.md`：参数、单位和约束。
- `docs/FORWARD_HANDOFF_CN.md`：已有正演/频散接口与历史验收；本文件补充当前源码/环境与 DAS 方向。
- `docs/DESKTOP_UI_GUIDE_CN.md`：桌面工作台。
- `docs/NCF_EGF_EQUIVALENCE_PLAN_CN.md`：当前研究主线、判据与历史阶段状态。
- `docs/GATE2_SPARSE_EVENT_RESULTS_CN.md`、`GATE2_SOURCE_AWARE_RESULTS_CN.md`、`GATE2_BRANCH_SPZ_RESULTS_CN.md`：负结果和机制边界。

2026-09-17 本次 DAS 实现已实际运行主动线阵、被动线阵、主动/被动规则面阵、主动逐道折线光缆和被动逐道折线光缆快速案例，检查 HDF5 `/das`、标准化 `das_channels.csv` 与 DAS–DAS NCF，并自顶向下执行、转为 HTML、在真实浏览器检查 `04_das_active_passive.ipynb`。完整回归为 242 passed；逐项散列和结论边界见 `docs/forward_workflow/EVIDENCE_REGISTER.md`。这些是有限标距算子解析证据及端到端 smoke 证据，不是长时被动收敛、DAS–DAS EGF 等价、连续光纤力学或全模型精度证书。此前 CodeGraph 统计不代表本次代码快照。
