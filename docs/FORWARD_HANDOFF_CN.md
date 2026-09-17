# WaveDispX 波场正演与频散接口移交说明

本文面向只需要接收正演数据、独立开发频散能量谱算法的项目。生产入口、Notebook 和界面均调用
同一套 Python/C++ 代码；下面列出的 HDF5 数组可直接作为算法输入。

## 0. 当前结论与实跑状态（2026-08-22）

这个项目具备从模型到“波形/NCF → 频散能量图 → 自动拾取 → 理论频散曲线”的主动源和背景噪声
完整流程。项目已有健康的 CodeGraph 索引；本次收尾时包含 184 个文件、2,171 个符号节点和 4,778
条关系边，状态为 `Index is up to date`。

需要严格区分三个计算层：

| 层 | 当前实现 | 是否 QEDisp | 是否 MATLAB |
|---|---|---:|---:|
| 波场正演 | Python 调度 C++17/pybind11 离散波数法 DWM | 否 | 否 |
| 频散能量成像 | Python/C++ 的 phase-shift、Bessel/FJ 或 Hankel | 否 | 否 |
| 理论相速度真值 | `processing.theory_solver = "qedisp"` 时调用本地 QEDispInv `forward` | 是 | 否 |

因此，“QEDisp backend”只描述理论频散曲线求解器，不是波场正演后端，也不是能量图成像后端。
`theory_solver` 没写时仍默认 `disba`；要用 QEDisp 真值做自动拾取评测，必须显式写
`theory_solver = "qedisp"`，并核对 `results.h5/theory` 组属性 `solver` 为
`QEDispInv/quadratic-extrema`。QEDisp 不可用时程序会报错，不会静默回退到 disba。

本次在 `seisloc` Conda 环境实际完成：

| 用途 | 配置 | 结果 | 说明 |
|---|---|---|---|
| QEDisp 接口单测 | `python_tests/test_qedisp.py` | `4 passed` | 含 QEDisp/disba 基阶对照、核函数、原生群速度与漏根测试 |
| active smoke | `examples/handoff_active_qedisp_quick.toml` | 成功 | 21 接收点；只检查完整链路，不用于准确率结论 |
| passive smoke | `examples/handoff_passive_qedisp_quick.toml` | 成功 | 24 源、21 接收点、6 个独立窗；只检查完整链路 |
| active 多模态评测 | `examples/handoff_active_multimode_qedisp.toml` | 成功 | 165 道、Rayleigh/Love 各 4 模态；推荐给自动拾取项目 |

推荐评测算例的实跑结果在 `output_python/handoff_active_multimode_qedisp/`：

- 炮集 `shot/components` 为 `(2210, 165, 5)`；
- Rayleigh 联合能量为 `(261, 819)`，轴顺序为 `velocity × frequency`，全部有限且坐标严格递增；
- QEDisp Rayleigh 真值为 `(819, 4)`，4 个模态分别有 `819/752/667/429` 个有效频点；
- QEDisp Love 真值为 `(819, 4)`，分别有 `819/669/465/199` 个有效频点；
- 联合 Rayleigh 与 Love 的 `clean_ridge_pass` 均为 `true`；项目自带基阶拾取的 MAE 分别为
  `3.036 m/s` 和 `2.145 m/s`。这些内置拾取只应作为下游算法的 sanity-check 基线。

## 1. 正演范围与核心

- 介质：一维水平分层、各向同性、弹性或常 Q 衰减；每层输入厚度、Vp、Vs、密度、Qp、Qs；
- 核心：C++17 使用离散波数法（DWM），在频率—水平波数域组装动力刚度并求解 Green 响应；Python 做参数、几何、
  径向逆 Hankel、噪声合成、互相关、输出与验收；
- 空间 Green 基：`RR, TT, ZR, RZ, ZZ`。命名为“接收分量—源力方向”，例如 `ZR` 是径向
  水平力引起的竖直接收位移；
- 理论频散由配置选择 `QEDispInv` 或 `disba` 独立求解，只用于标签、叠加和验收，不进入波场、
  NCF 合成或能量成像。标准 `run` 保存的是相速度；QEDisp 的原生群速度接口另有单测，但标准
  HDF5 交接不把群速度当作自动拾取标签。

### 1.1 频率域核心、有限记录产品与三个不同的“时长”

WaveDispX 不是有限差分式逐时间步推进。C++ DWM 在离散的复频率—水平波数网格上求
$G(f,k)$，再由 Python 做径向逆 Hankel。当前生产能量图仍是**有限记录波形产品**：

- 主动源：$G(f,k)$ → 时域 Green → 子波卷积 → 截取炮集 → 对炮集重新 FFT →
  phase-shift、Bessel/FJ 或 Hankel 成像；
- 被动源：$G(f,k)$ → 时域 Green → 转到每个噪声窗的传递谱 → 随机源有限窗合成 →
  接收端混合 → 接收器互相关 → NCF 成像。

因此，不能因为核心是频率域就把当前产品当作“与时长无关的直接频谱图”。需要分别记录：

| 量 | 控制什么 | 不能与什么混淆 |
|---|---|---|
| 主动 `record_duration_s` | 慢群速度到时是否完整；炮集真实 $\Delta f=1/(N\Delta t)$ | `zero_padding` 只加密显示 |
| 被动 `noise.window_s` | 单窗 Green/NCF 最大时延与 $\Delta f=1/T_w$ | 不是统计收敛时长 |
| 被动 `noise.maximum_lag_s` | 实际保存、展示并成像的 NCF 正时延范围 | 不是事后绘图裁剪 |
| 被动 `noise.duration_s` | 独立窗数与随机互相关收敛 | 不直接改善单窗频率分辨率 |
| `processing.frequency_step_hz` | 频散图显示/扫描频率网格 | 不是物理 $1/T$ 分辨率 |
| DWM synthetic period | DWM 频点步长及复频移位 $\gamma=2\pi/T_{DWM}$ | 不是保存波形的 FFT 周期 |

主动 `record_duration_s=0`（或省略）时，程序用模型多模态群速度、最大炮检距、源持续时间和
最低处理频率自动设计。被动预检查用相同群速度逻辑检查 Green 响应时长和 `window_s`。HDF5 根属性
和 `manifest.json` 分别保存 observation FFT period、DWM synthetic period、两种 $\Delta f$ 与
复频移位，避免下游把它们混为一谈。

当前没有把“未经有限波形窗的实频率直接有限阵列响应”冒充生产结果。若今后增加，应单独命名为
`coherent_frequency_array`，显式记录实频率/复频率、Laplace 权重和阵列算子，并与现有
`windowed_waveform_image`、`finite_record_ncf_waveform_image` 分开验收。

### 1.2 频散图 f-v 范围的模型感知预检查

预检查现在先在源有效带内计算多模态理论相速度/群速度，再同时施加源功率（相对峰值不低于
-20 dB）、至少约两个波长孔径、空间 Nyquist，以及 DAS 第一标距零点约束，给出建议频带。
速度窗由保留频带中所有有限理论模态的最小/最大相速度再留上下边界，而不是只围住基阶曲线。
建议值只产生明确警告，不会静默改写用户输入；若研究目标是有意裁剪的子频带，应在结果中声明。

## 2. 主动源数据

配置示例：`examples/active_high_modes_3c.toml`。

流程：

1. `compute_full_fk_green` 求五分量 f-k Green；
2. `synthesize_full_receivers` 逆 Hankel 到每个源检距；
3. 与配置的主动源时间函数线性卷积（Ricker；理想单位脉冲校准时保持 Green 响应）；
4. 输出 `time × receiver × 5` 炮集。

生产 HDF5 根属性 `dispersion_product="windowed_waveform_image"`。`record_frequency_step_hz`
对应保存炮集的真实 FFT 周期；`dispersion_display_frequency_step_hz` 对应实际输出图的频率网格；`dwm_frequency_step_hz` 和
`dwm_complex_frequency_shift_radps` 对应内部 DWM 网格，两者通常不同。

`results.h5/shot/components` 的末轴顺序固定为：

| 索引 | 名称 | 含义 |
|---:|---|---|
| 0 | RR | 径向水平力 → 径向位移 |
| 1 | TT | 切向水平力 → 切向位移 |
| 2 | ZR | 径向水平力 → 竖直位移 |
| 3 | RZ | 竖直力 → 径向位移 |
| 4 | ZZ | 竖直力 → 竖直位移 |

同时读取 `shot/time_s`、`shot/distance_m`、`shot/receiver_x_m`、`shot/receiver_y_m`。这些五分量
是局部柱坐标 Green 基；线阵位于源同一侧时 R/T 就是全局 X/Y。任意二维阵列若要模拟一个固定
方位的实体水平震源，应按源—接收器方位自行把该 Green 基旋转到全局坐标。

## 3. 被动真实有限记录

配置示例：`examples/passive_high_modes_3c.toml`。程序不输出或使用“逐源理想 NCF”。

每个独立窗内，源谱为 PSD 振幅乘独立复高斯过程，或由 Poisson 随机事件叠加。3C 源对每个位置
生成独立全局 `Fx,Fy,Fz`。设源到接收器方位为 $\theta$：

\[
F_r=F_x\cos\theta+F_y\sin\theta,\quad
F_t=-F_x\sin\theta+F_y\cos\theta,
\]
\[
u_r=G_{rr}F_r+G_{rz}F_z,\quad u_t=G_{tt}F_t,\quad
u_z=G_{zr}F_r+G_{zz}F_z.
\]

局部响应旋回全局 X/Y/Z 后，所有源先在接收器相加。互相关器只接收混合数组
`time × receiver × 3`，按窗做 detrend、taper、可选 whitening/one-bit，然后形成接收器对互谱。
线阵以 X/Y/Z 形成 RR/TT/ZZ；任意二维阵列针对每一台站对重新旋转水平分量，再按距离箱叠加。

主要 HDF5 路径：

- `preview/mixed_record`：前 60 s 混合记录；竖直模式为二维，3C 为三维且末轴 X/Y/Z；
- `ncf/lag_time_s`, `ncf/distance_m`, `ncf/pair_count`；
- `ncf/components/zz/gather`, `rr/gather`, `tt/gather`：`lag × distance`；
- `transfer/frequency_hz`；仅当 `store_source_responses=true` 时保存
  `transfer/full_green_spectrum`，末轴顺序与上表一致；
- `convergence.csv`：60/600/最终时长相对 L2 与拾取差异。

生产 HDF5 根属性 `dispersion_product="finite_record_ncf_waveform_image"`。
`ncf_frequency_step_hz=1/noise.window_s`；总记录时长和独立窗数另存，不能用总记录的倒数作为
单窗 NCF 能量图的频率分辨率。`ncf_maximum_lag_s` 保存实际 lag 终点，
`dispersion_display_frequency_step_hz` 保存成像输出网格；两者都不得替代真实窗长元数据。

## 4. 频散能量与理论标签的 HDF5 契约

### 4.1 DAS 有限标距接口（2026-09-17）

下游若需要 DAS 波形/NCF，应使用 `examples/active_das_line_quick.toml`、
`passive_das_line_quick.toml` 或 `passive_das_grid_quick.toml`，不要从节点 ZZ/RZ 数组自行猜测标距。
WaveDispX v1 在每个道中心的 `±Lg/2` 端点计算完整 Green 位移，投影到光纤切向后做端点差除以
`Lg`；`strain_rate` 再按 NumPy rFFT 约定乘 `+i2πf`。被动源是先形成源→DAS 道传递函数，随后
生成有限混合记录并在 DAS 道之间内部互相关。

`das.geometry_mode="channel_file"` 时，不再使用全局 `Lg`/方位：接收器 CSV 每行给出两个端点，或给出该道的标距与方位。每道单独投影并除以自己的标距；规则面阵和转弯/变标距光缆因此共用同一测量接口。

主动 HDF5：

- `das/time_s`, `das/data`：`time × channel`；
- `das/channel_center_xy_m`, `endpoint_minus_xy_m`, `endpoint_plus_xy_m`, `tangent_xy`；
- `das/channel_id`、`gauge_length_by_channel_m`、`fiber_azimuth_by_channel_deg`；
- `/das` 属性：`measurement`、`units`、`gauge_length_m`、`fiber_azimuth_deg`、
  `geometry_mode`、`operator_version`、`time_convention`；逐道非统一时两个旧标量属性为 NaN，以逐道数据集为准；
- `dispersion/components/das/*` 和 `picked_das.csv`。

被动 HDF5：

- `das/time_s` 与 `das/strain` 或 `das/strain_rate` 保存前 60 s 预览；
- `ncf/components/das/gather` 是 `lag × distance` 的 DAS–DAS NCF；
- 主频散仍在 `dispersion/rayleigh/*`，同时在 `dispersion/components/das/*`；
- 若保存逐源响应，`transfer/das_spectrum` 为 `frequency × channel × source`；
- 当前没有把节点主动 Green 冒充 DAS–DAS EGF，manifest 明确记录
  `DAS-DAS reference not yet implemented`。
- 输出目录的 `das_channels.csv` 固化实际端点、逐道标距和方位，可直接作为 `channel_file` 下一次输入。

当前验证范围仅含竖直点力和 phase-shift。FJ/Bessel/Hankel 的 DAS 核、三分量随机源、弯曲/井中
光纤及 source-selection/NNWS/SPZ 组合均未宣称验证通过。下游频散项目应保留该 provenance。

主动算法直接读取 `shot/components[:,:,k]` 和 `shot/distance_m`；被动算法优先读取相应
`ncf/components/*/gather` 与 `ncf/distance_m`。不要把 `pair_count` 再乘入已经按距离平均的 NCF；
若算法希望按原始台站对加权，应改为读取/扩展未分箱互谱接口。

如果下游项目直接消费本项目已经形成的频散能量图，主动 Rayleigh 首选：

- `dispersion/rayleigh_combined/energy`：`[n_velocity, n_frequency]`；
- 同组 `frequency_hz`：`[n_frequency]`，严格递增；
- 同组 `velocity_mps`：`[n_velocity]`，严格递增；
- `theory/frequency_hz`：与能量频率网格一致；
- `theory/rayleigh_mps`：`[n_frequency, n_mode]`，列号 `0,1,...` 是基阶、一阶高模等；
- `theory` 组属性 `solver`：真值求解器 provenance。

Love 使用 `dispersion/love/*` 和 `theory/love_mps`。单分量能量在
`dispersion/components/{rr,tt,zr,rz,zz}`；联合 Rayleigh 是 RR/ZR/RZ/ZZ 各自成像、逐频归一后
的 RMS 组合。能量每个非零频率列的最大值为 1，HDF5 不保存 `raw_amplitude`，所以适合拾取与图像
算法比较，不适合反推跨频绝对振幅。

QEDisp 某一模态在某一频点没有找到根时保存 `NaN`。这是标签掩码，不是零速度；不能填 0，也不能
跨越缺失区把曲线强行插值连接。HDF5 数组的第 0 行对应最低速度；若下游图像库以左上角为原点，
必须在接口中显式决定是否用 `energy[::-1, :]`，不能在训练和评测两端各自猜测方向。

本项目内置三种参考成像：phase-shift、Bessel/FJ、Hankel。设置
`processing.compare_methods=true` 会把同一输入、同一频率/速度网格的三张能量写入
`dispersion/method_comparison/{phase_shift,bessel,hankel}`。可用这些结果检查新项目的坐标方向、
FFT 符号、速度轴和频率轴，但不要求复现某一种归一化颜色。

`picked_rayleigh.csv`、`picked_rayleigh_combined.csv` 和 `picked_love.csv` 是 WaveDispX 自带拾取器的
预测结果，不是真值。准确率标签只能取 `theory/*_mps`；`manifest.json/quality` 可作为已知基线。

## 5. 可复现运行、NPZ 导出与建议移交包

```bash
# 推荐的多模态 QEDisp 评测集
conda run -n seisloc wavedispx validate \
  examples/handoff_active_multimode_qedisp.toml
conda run -n seisloc wavedispx run \
  examples/handoff_active_multimode_qedisp.toml

# active/passive 最小链路 smoke test
conda run -n seisloc wavedispx run examples/handoff_active_qedisp_quick.toml
conda run -n seisloc wavedispx run examples/handoff_passive_qedisp_quick.toml
```

为避免下游误解 HDF5 轴顺序，可直接生成自描述 NPZ：

```bash
conda run -n seisloc python scripts/export_picker_benchmark.py \
  output_python/handoff_active_multimode_qedisp/results.h5 \
  output_python/handoff_active_multimode_qedisp/picker_benchmark_rayleigh.npz

conda run -n seisloc python scripts/export_picker_benchmark.py \
  output_python/handoff_active_multimode_qedisp/results.h5 \
  output_python/handoff_active_multimode_qedisp/picker_benchmark_love.npz \
  --wave-type love
```

NPZ 固定包含：

- `energy_vf`：`velocity × frequency` 的 `float32` 能量；
- `frequency_hz`、`velocity_mps`：原生坐标；
- `theory_phase_velocity_mps_fm`：`frequency × mode` 的 QEDisp 相速度；
- `theory_valid_mask_fm`：排除 QEDisp 漏根的布尔掩码；
- `metadata_json`：原 HDF5 路径、轴顺序、归一化、正演/理论求解器及版本。

当前目录已经生成可直接复制的 `picker_benchmark_rayleigh.npz` 和
`picker_benchmark_love.npz`。最小可审计移交包为
`input.toml + model.csv + receivers.csv + sources.csv + results.h5 + manifest.json + picker_benchmark_*.npz`。
`manifest.json` 记录运行时间、内存、插值误差、随机窗数、质量与逐模态理论曲线能量支持指标。

## 6. 自动拾取准确率的建议口径

单模态算法首先只评估 QEDisp 第 0 列。若预测已经对齐 NPZ 的原生频率网格，最小评分可写为：

```python
import numpy as np

data = np.load("picker_benchmark_rayleigh.npz")
truth = data["theory_phase_velocity_mps_fm"][:, 0]
prediction = ...  # shape == truth.shape，单位 m/s
valid = data["theory_valid_mask_fm"][:, 0] & np.isfinite(prediction)
error = prediction[valid] - truth[valid]
metrics = {
    "coverage": valid.mean(),
    "mae_mps": np.mean(np.abs(error)),
    "rmse_mps": np.sqrt(np.mean(error**2)),
    "p95_absolute_error_mps": np.quantile(np.abs(error), 0.95),
    "within_10_mps": np.mean(np.abs(error) <= 10.0),
}
```

如果预测频率与原生网格不同，只允许在预测曲线实际覆盖的连续频段内插值到真值频率，并同时报告
覆盖率。多模态算法不能在每个频点单独选择“最近的真值模态”，因为那会奖励跨模跳跃；应先按整条
轨迹的重叠频段 MAE 加覆盖惩罚做全局一一匹配，再逐模态报告 MAE/RMSE/覆盖率。训练集与测试集还
必须按速度模型分组切分，不能只换噪声种子，否则会泄漏完全相同的理论曲线。

推荐分两级验收：

1. 用 `handoff_active_multimode_qedisp` 检查读取、方向、单位、掩码、基阶和高模输出是否正确；
2. 再批量改变层厚/Vs/Vp/密度、孔径、道距、子波和噪声，实现跨模型盲测。本次生成的一个模型足以
   验证代码正确性，不足以证明深度学习模型的泛化能力。

passive quick 只有 6 个独立窗，`clean_ridge_pass=false`，只作为 NCF 接口 smoke test；不要用它
给拾取网络下准确率结论。要评估背景噪声鲁棒性，应复制长时 `passive_high_modes_3c.toml`，显式
加入 `theory_solver = "qedisp"`，用独立模型和多个噪声种子生成测试集，并先通过 `clean_ridge_pass`。

## 7. 明确没有使用的 tricks

- 没有沿理论频散曲线人工填能量；
- 没有逐噪声源先互相关再叠加；
- 没有把源编号或源位置传入 NCF/频散函数；
- 径向 Green 插值会抽取真实距离做相对 L2 检查；
- 流式处理只是不把 1800 s 全记录常驻内存，与相同独立窗的离线叠加等价；
- 联合 Rayleigh 是 RR/ZR/RZ/ZZ（主动）或 RR/ZZ（被动）单独成像后做列归一 RMS；原分量图始终
  单独保留。它适合检查分量互补性，不能替代各分量的物理解释。

## 8. 已知限制

- 一维层状模型不含横向非均匀、地形、各向异性或非线性；
- 被动等强点力源环并不严格等于闭合边界上的牵引加权 Green 表示，因此源数/时长收敛后，模态
  相对振幅仍可能不同于单炮 Green；
- line/grid/file 的被动频散目前按台站距离径向平均，不保留方位信息；
- 高频高阶同时受模型截止频率、分量耦合、道距混叠、孔径和噪声场照明控制。增加源数只能降低
  随机照明误差，不能制造速度模型不支持或分量不耦合的高阶。
