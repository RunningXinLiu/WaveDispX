# WaveDispX 输入参数手册

输入可在主动/被动 quickstart Notebook 顶部唯一的 `PARAMETERS` 参数总控单元中定义、在 `wavedispx-ui` 本地界面中编辑，
也可使用 TOML 文件。三种方式都进入同一个类型校验器和同一套计算函数。批处理时推荐从
`examples/active_two_layer.toml` 或 `examples/passive_two_layer.toml` 复制修改，并在运行前执行：

```bash
conda run -n seisloc python -m wavedispx validate your_case.toml
```

校验器会在正演前检查单位、半空间、Nyquist 频率、频带和几何一致性，并为每个 required
失败显示参数路径、当前值、判据和修改方法。Quickstart 还会自动渲染完整中文参数字典。

交互操作推荐运行 `conda run -n seisloc wavedispx-ui`；原理和逐 Cell 调试使用
`notebooks/01_active_quickstart.ipynb`、`02_passive_quickstart.ipynb` 和
`03_methods_and_validation.ipynb`。所有入口都先绘制 `Vp/Vs`、密度、源—检波器几何、主动
子波或被动 PSD/随机源样例及理论频散；required 检查失败时禁止正演。

## 1. `[run]`：运行与性能

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `name` | 算例名称 | 简短英文名 |
| `mode` | `active` 或 `passive` | 必填 |
| `output_directory` | 输出目录，相对路径按配置文件所在目录解释 | `../output_python/...` |
| `workers` | CPU 线程数；0 表示自动 | 0 |
| `frequency_block_size` | 离散波数法核心每个频率块大小 | 8–32 |
| `source_block_size` | 每个噪声源块大小 | 8–32 |
| `radial_interpolation_step_m` | 层状介质径向 Green 插值步长 | 0.5 m |
| `store_source_responses` | 是否保存每个源的完整响应 | 默认 `false` |

`store_source_responses=false` 是真实混合噪声的推荐设置。它只流式累加接收器记录或互谱，
避免旧 MATLAB 版本数 GB 的 `TT.r/TT.t/TT.z` 中间数组。只有诊断单个噪声源时才开启。

层状模型的 Green 函数只依赖源检距离。程序先在规则径向网格计算，再插值到所有真实距离。
校验器要求步长不超过最短 S 波波长的 5%；默认 0.5 m 在当前 18.8 Hz 模型中满足要求。

## 2. `[[model.layers]]`：层状模型

每层重复一个 `[[model.layers]]`：

| 参数 | 单位 | 含义 |
|---|---:|---|
| `thickness_m` | m | 层厚；最后一层必须为 `inf` |
| `vp_mps` | m/s | P 波速度 |
| `vs_mps` | m/s | S 波速度 |
| `density_kgm3` | kg/m³ | 密度 |
| `qp` | 无量纲 | P 波品质因子；弹性介质用 `inf` |
| `qs` | 无量纲 | S 波品质因子；弹性介质用 `inf` |

必须满足 `vp_mps > vs_mps > 0`。程序内部使用
`Damp_p=1/(2Qp)`、`Damp_s=1/(2Qs)`；`Q=inf` 对应零衰减。

## 3. `[geometry.receivers]`：接收器观测系统

支持规则水平直线阵列、直接参数化的二维规则面阵列，以及 CSV 输入的任意二维地表阵列。

### 规则线阵 `layout="line"`

| 参数 | 单位 | 含义 |
|---|---:|---|
| `layout` | — | `line` |
| `start_x_m` | m | 第一道 x 坐标 |
| `stop_x_m` | m | 最后一道 x 坐标 |
| `spacing_m` | m | 道间距 |
| `y_m` | m | 阵列 y 坐标 |

接收器数量为 `(stop-start)/spacing+1`。跨度必须能被道间距整除。被动互相关会使用空间
FFT 直接对相同台间距叠加，这是速度最快且与逐台站对算法数值一致的路径。

### 二维规则面阵列 `layout="grid"`

```toml
[geometry.receivers]
layout = "grid"
start_x_m = -25.0
stop_x_m = 25.0
spacing_x_m = 5.0
start_y_m = -25.0
stop_y_m = 25.0
spacing_y_m = 5.0
distance_bin_m = 5.0  # 被动必填，主动可省略
```

接收器数量为 `Nx × Ny`，上例是 11×11=121 个接收器。x、y 跨度都必须是对应间距的
整数倍。Notebook 用 `RECEIVER_LAYOUT='line'/'grid'/'file'` 在规则线阵、二维 grid 和文件阵列
之间切换，并在输入审核阶段显示当前选定 geometry。可运行示例见 `examples/passive_grid_quick.toml`。

### 任意二维阵列 `layout="file"`

```toml
[geometry.receivers]
layout = "file"
coordinates_file = "receivers_irregular.csv"
distance_bin_m = 10.0  # 被动必填，主动可省略
```

CSV 必须有表头 `x_m,y_m`，每行一个接收器；接收器数由文件行数自动推断。完整示例见
`examples/active_irregular_quick.toml`、`examples/passive_irregular_quick.toml` 和
`examples/receivers_irregular.csv`。同时包含接收器、炮点和噪声源文件的完整观测系统见
`examples/observation_system/`。

- 主动源：WaveDispX 按炮点到每个 `(x,y)` 的真实距离合成波形，频散计算按源检距排序；若对称
  阵列出现相同源检距，先对同半径波形平均再成像；
- 被动 grid/file：先对所有 `N(N-1)/2` 个接收器对做互相关，再把相近台间距舍入到
  `distance_bin_m` 的整数倍并叠加；
- 当前频散图是径向平均，台站对方位会被丢弃，适合一维层状模型和近似各向同性噪声场；
  方位各向异性或定向噪声研究不能直接用这个径向平均结果，需要方位分箱；
- `distance_bin_m` 通常取典型最近邻距离。过小会使每个距离箱台站对过少、能量不稳定；
  过大会平滑空间相位并降低分辨率。

被动面波的空间混叠上限近似为：

\[
f_{alias}\approx \frac{c_{min}}{2\Delta x}.
\]

因此 5 m 道间距适合约 20 Hz 以下；如需 40–50 Hz，应采用约 1–2 m 道间距。不规则阵列
没有唯一的 `Δx`：主动源预检使用唯一径向源距的中位间隔，被动源预检使用
`distance_bin_m`，与频散阶段实际看到的径向距离采样一致。

## 3.1 `[das]`：有限标距 DAS（可选）

`[geometry.receivers]` 仍定义**通道中心**；DAS 的标距和方向单独定义，不能把道间距当成标距：

```toml
[das]
enabled = true
geometry_mode = "uniform"       # uniform 或 channel_file
gauge_length_m = 8.0
measurement = "strain_rate"  # strain 或 strain_rate
fiber_azimuth_deg = 0.0       # 相对全局 +x 轴逆时针
```

| 参数 | 单位 | 含义与约束 |
|---|---:|---|
| `enabled` | — | 开启有限标距 DAS；关闭时保持原节点接收器行为 |
| `geometry_mode` | — | `uniform` 使用全局统一标距/方位；`channel_file` 从接收器 CSV 逐道读取实际端点或逐道标距/方位 |
| `gauge_length_m` | m | `uniform` 模式标距 `Lg>0`；每道端点为中心 `±Lg/2`，允许相邻标距重叠；逐道模式由 CSV 覆盖 |
| `measurement` | — | `strain` 输出轴向应变；`strain_rate` 输出应变率 s⁻¹ |
| `fiber_azimuth_deg` | deg | `uniform` 模式光纤切向相对 +x 逆时针；规则面阵表示所有中心处放置同方位平行标距；逐道模式由 CSV 覆盖 |

`channel_file` 要求 `[geometry.receivers].layout="file"`，并复用同一个 `coordinates_file`。
文件行序就是通道序；`channel_id` 可选但若提供必须唯一。支持两种列组合：

```csv
channel_id,x_m,y_m,endpoint_minus_x_m,endpoint_minus_y_m,endpoint_plus_x_m,endpoint_plus_y_m
D001,0,0,-4,0,4,0
D002,10,2,6,2,14,2
```

或者：

```csv
channel_id,x_m,y_m,gauge_length_m,fiber_azimuth_deg
D001,0,0,8,0
D002,10,2,6,45
```

端点格式中，`x_m,y_m` 必须是两个端点的中点。逐道模式允许弯曲、折返和变标距；它表示一组离散有限标距测量段，并不模拟光纤连续介质、弯曲段积分或解调仪光学传递函数。结果会输出规范化的 `das_channels.csv`，可直接作为下一次输入。

计算不是对节点频散图乘经验滤波器。主动源在每个端点计算水平位移；被动源先构造每个源到两个端点的完整 Green 响应，再做轴向投影与端点差，然后才生成随机混合记录及 DAS–DAS 互相关：

\[
\epsilon_{tt}(t)=\frac{[\mathbf u(\mathbf x_+,t)-\mathbf u(\mathbf x_-,t)]\cdot\hat{\mathbf t}}{L_g}.
\]

NumPy rFFT 正变换采用 `exp(-iωt)`，所以应变率显式乘 `+iω`。有限标距响应为
`sinc(k_t Lg/2)`，第一零点约为 `f=c/Lg`；预检会提示目标频带是否穿过该物理灵敏度缺口。

DAS v1 当前有意限制为：竖直主动点力或竖直环境噪声点力、`phase_shift` 成像、
`compare_methods=false`，并暂不与 source-selection/NNWS/pair-SPZ 研究分支组合。FJ/Bessel 和 Hankel
的 DAS 径向核尚未验证，配置器会明确阻止而不是静默换算法。可运行模板为：

- `examples/active_das_line_quick.toml`；
- `examples/active_das_grid_quick.toml`；
- `examples/active_das_channel_file_quick.toml` 与 `examples/das_channels_bent_example.csv`；
- `examples/passive_das_line_quick.toml`；
- `examples/passive_das_grid_quick.toml`；
- `examples/passive_das_channel_file_quick.toml`；
- `notebooks/04_das_active_passive.ipynb`（一个 Cell 集中切换 active/passive、line/grid 或完整 TOML）。

## 4. `[active_source]`：主动源

仅 `run.mode="active"` 使用：

| 参数 | 单位 | 含义 |
|---|---:|---|
| `x_m`, `y_m` | m | 炮点坐标 |
| `component` | — | `vertical`、`horizontal` 或 `both` |
| `wavelet` | — | `ricker` 或用于理想宽频校准的 `unit_impulse` |
| `dominant_frequency_hz` | Hz | Ricker 主频；`unit_impulse` 时为正演时窗参考频率 |
| `maximum_frequency_hz` | Hz | 波场正演最高计算频率 |
| `dt_s` | s | 采样间隔 |
| `record_duration_s` | s | 去除 DWM 延迟后保存并用于频散成像的炮集时长；`0` 或省略表示按模型群速度、最大炮检距和 `processing.frequency_min_hz` 自动设计 |

必须满足 `maximum_frequency_hz < 1/(2*dt_s)`。

`record_duration_s` 决定炮集 FFT 的真实频率分辨率 $\Delta f=1/(N\Delta t)$。
`processing.zero_padding` 只加密频率轴显示，不会补回被截断的慢群速度到时，也不会提高真实分辨率。

界面还支持上传 `shot_id,x_m,y_m` Shot CSV。文件可包含多炮，但当前正演核心为“一炮一个
算例”：界面先选择一行，所选 x/y 再写入 `[active_source]`。示例见
`examples/observation_system/shots.csv` 和 `active_selected_shot.toml`。

## 5. `[ambient_sources]`：环境噪声源空间分布

仅 `run.mode="passive"` 使用。

### 环形源 `layout="annulus"`

| 参数 | 含义 |
|---|---|
| `count` | 同时参与混合的源数；推荐 192–384 |
| `seed` | 几何随机种子 |
| `inner_radius_m`, `outer_radius_m` | 环带内外半径 |
| `angular_sampling` | `random`、`stratified` 或 `uniform` |
| `component` | `vertical`：Fz→Z 快速兼容路径；`three_component`（也接受 `3c`）：物理 Fx/Fy/Fz 三分量路径 |

半径按面积均匀抽样，不是简单的半径均匀。正式模拟推荐 `stratified`，可减少方位空缺。

`three_component` 不是三份竖直记录的复制。设源到接收器方位为 $\theta$，全局随机力先变为

\[
F_r=F_x\cos\theta+F_y\sin\theta,\qquad
F_t=-F_x\sin\theta+F_y\cos\theta.
\]

局部响应使用五个 离散波数法 Green 分量：

\[
u_r=G_{rr}F_r+G_{rz}F_z,\quad
u_t=G_{tt}F_t,\quad
u_z=G_{zr}F_r+G_{zz}F_z,
\]

随后旋回全局 X/Y/Z。线阵直接以 X/Y/Z 形成 RR/TT/ZZ；二维或不规则阵列会针对每一台站对
重新按台站间方位旋转水平记录，然后再按 `distance_bin_m` 叠加。

### 道路或直线源 `layout="road"/"line"`

使用 `x_min_m`、`x_max_m`、`y_m` 定义一条源线。

### 文件源 `layout="file"`

使用 `coordinates_file="sources.csv"`。CSV 至少有两列 `x_m,y_m`，行数必须等于 `count`；
允许增加 `source_id` 等说明列。文件模式直接使用给定坐标，`ambient_sources.seed` 不改变位置。

### 逐窗源活动 `activity_file`

`activity_file="source_activity.csv"` 使用长表
`window_index,source_index,amplitude_multiplier` 固定每个独立窗的源活动。只要提供该文件，
**未列出的窗--源组合就是 0（不活动）**；只有完全不提供 `activity_file` 时，所有源的乘子才
默认为 1。每个窗必须至少有一个正乘子，索引从 0 开始且不允许重复。该表用于预先声明时变
各向异性照明和筛选基准，不允许根据最终 NCF 或频散图事后修改。

## 6. `[noise]`：噪声 PSD 与时间分布

| 参数 | 单位 | 含义 |
|---|---:|---|
| `temporal_model` | — | `stationary_gaussian` 或 `poisson_ricker` |
| `seed` | — | 噪声时间序列随机种子，与几何种子分开 |
| `duration_s` | s | 总记录长度；推荐 600–1800 s |
| `dt_s` | s | 采样间隔 |
| `window_s` | s | 独立处理窗；当前推荐 20 s |
| `maximum_lag_s` | s | NCF 保存并送入频散计算的最大正时延；`0` 或省略时按模型最慢群速度和阵列孔径自动设计，显式值必须小于 `window_s` |
| `overlap` | — | 当前必须为 0；各合成窗独立并由流式互相关器叠加 |
| `psd_shape` | — | `ricker`、`flat_band` 或 `file` |
| `dominant_frequency_hz` | Hz | PSD 主频 |
| `minimum_frequency_hz` | Hz | PSD 下限 |
| `maximum_frequency_hz` | Hz | PSD 上限 |
| `source_amplitude` | 相对量 | 每个源的振幅尺度 |
| `independent_sources` | — | 不同源是否独立；真实 NCF 基准用 `true` |
| `event_rate_hz` | 1/s | `poisson_ricker` 每个源的平均事件率 |

`stationary_gaussian` 在每个独立时间窗和每个源上生成独立复高斯频谱，再乘 PSD 包络，
经过 离散波数法 Green 函数传播后才在接收器处混合。处理阶段看不到源标签。

这里两个时长的物理含义不同：`window_s` 必须容纳最慢目标群速度对应的最大源—接收器传播时间
和最大台间 NCF 时延，并决定单窗频率分辨率 $\Delta f=1/window_s$；`duration_s` 决定独立窗数
及随机互相关的统计收敛。把 `duration_s` 从 600 s 增到 1800 s 会降低随机起伏，但不会把一个
20 s 窗的真实 $\Delta f$ 从 0.05 Hz 变成 1/1800 Hz。

`maximum_lag_s` 控制实际保存/展示的 NCF lag 轴，并非显示端的事后裁图。减小它可节省缓存，
但若短于最慢目标群速度的远台到时，会直接截断面波；增大它可同时检查早到体波、主面波和尾波，
但必须保持 `< window_s`。

`poisson_ricker` 用于模拟间歇交通或机械事件。对每个源、每个独立窗，事件数满足
`N~Poisson(event_rate_hz×window_s)`，窗内时刻均匀分布在 `[0,window_s)`；事件极性随机为
±1，绝对权重在 0.5–1.0 均匀抽样。第 `w` 个窗的绝对时刻为
`w×window_s+time_in_window_s`，整套事件由 `noise.seed` 固定。

Poisson 运行会保存 `noise_events.csv` 以及 HDF5 的 `noise_events/` 组，记录窗号、源号、源坐标、
窗内/绝对时刻和权重。`stationary_gaussian` 是连续随机过程，没有离散激发时刻，因此不生成
事件目录。

当 `psd_shape="file"` 时，还需指定 `psd_file`。文件应包含 `frequency_hz,power` 两列。

## 7. `[processing]`：互相关与频散

| 参数 | 含义 |
|---|---|
| `correlation_normalization` | `spectral_whitening`、`one_bit` 或 `none` |
| `dispersion_method` | `phase_shift`、`bessel` 或 `hankel` |
| `theory_solver` | `disba`（默认）或 `qedisp`；后者需本地 QEDispInv `forward`，适合低速夹层和 mode-kissing 审计 |
| `compare_methods` | `true` 时在相同网格同时输出三种方法对照；默认 `false`，计算时间约增加 2–3 倍 |
| `frequency_min_hz`, `frequency_max_hz` | 处理频带 |
| `velocity_min_mps`, `velocity_max_mps`, `velocity_step_mps` | 扫描相速度 |
| `frequency_step_hz` | 频散图输出频率网格 `df`；`0` 或省略表示由真实记录长度与 `zero_padding` 自动给出。显式值仅改变显示/扫描网格，不提高 `1/T` 物理分辨率 |
| `distance_taper_fraction` | 距离域 Tukey taper，0–1；推荐 0.5 |
| `zero_padding` | 时间 FFT 补零倍数；主动推荐 5，被动推荐 8 |
| `radial_kernel_lookup_step` | Bessel/Hankel 核参数查表步长，推荐 `0.002`；设 `0` 使用逐点精确特殊函数。仅影响 `bessel`/`hankel` |
| `picker_smoothness` | 拾取的相邻频率速度平滑约束 |
| `picker_maximum_jump_mps` | 相邻频点允许的最大速度跳变 |
| `picker_minimum_energy` | 输出拾取的最小归一化能量 |
| `rayleigh_modes`, `love_modes` | 理论曲线模态数 |

预检查不会只在用户填写的 f-v 小窗内画理论线。它会先在源有效频带上计算多模态相/群速度，并结合
阵列孔径、有效径向采样、空间混叠和 DAS 标距给出 `recommended_frequency_*` 与
`recommended_velocity_*`。用户范围过窄会得到逐参数警告，但程序不会静默替用户改值。速度窗建议
在所有保留理论分支外再留边界，以便看到主瓣、旁瓣和背景，而不是只得到一条被裁切的亮线。

当前推荐：

- 主动按源检距排列：`hankel`，必要时用 `phase_shift` 交叉验证；
- 被动距离域 NCF：`phase_shift`；
- 真实混合噪声：`spectral_whitening`；
- 被动正式记录：600 s 可用，1800 s 用于稳定展示。

高阶模态不能仅靠增加噪声源“生成”。它同时受层厚/速度对比、模态截止频率、源与接收分量耦合、
阵列孔径/道距及记录收敛影响。`examples/active_high_modes_3c.toml` 和
`examples/passive_high_modes_3c.toml` 使用经过扫描的三层模型；输出会分别保存 ZZ、RR 和联合
Rayleigh 图以及三方法对照。Bessel/FJ 对二维径向平均相关场更匹配，但在线阵有限孔径上通常比
phase-shift 产生更多确定性旁瓣，因此不能把每条亮纹都解释为高阶模态。

## 8. `[nnws]`：逐台站对神经网络加权叠加（研究选项）

默认不启用。只支持 `mode="passive"` 且 `ambient_sources.component="vertical"`：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `enabled` | 启用逐窗、逐台站对 NCF 缓存与独立 NNWS 训练 | `false` |
| `epochs`, `learning_rate` | 每对训练轮数和 Adam 学习率 | `200`, `0.001` |
| `snr_weight`, `symmetry_weight`, `std_weight` | 三个损失项系数 | `0.8`, `0.2`, `0.01` |
| `std_mode` | `upstream_sample` 精确上游损失，或 `population_eps`/`disabled` 消融 | `upstream_sample` |
| `seed`, `random_control_seed`, `device` | 网络初始化种子、随机同 `N_eff` 对照种子与计算设备 | `0`, `271828`, `auto` |
| `minimum_effective_windows` | 每个台站对最低权重有效窗数 | `2.0` |
| `minimum_pair_coverage` | 允许结果评价的最低成功台站对比例 | `1.0` |
| `maximum_cache_mb` | float32 三分支逐对缓存硬上限 | `512.0` |
| `pair_scope` | `all` 全台站对，或 `reference_receiver` 单虚拟源道集 | `all` |
| `reference_receiver_index` | 单虚拟源道集的接收点索引（从 0 开始） | `0` |

信号窗由每对距离和 `[processing]` 的速度上下限自动计算；最大周期取 `1/frequency_min_hz`。
启用后最大时延至少为 `aperture/velocity_min + 4/frequency_min`，以保存上游 reported-SNR 的
远端尾窗。几何上无法同时容纳 pre-signal、signal 和完整 remote-tail 的近距/超远距台站对会
标为 `eligible=false`，不进入覆盖率分母，也不算训练失败。若独立 `noise.window_s` 不够长或
缓存估计超限，程序在训练前中止。

输出 `nnws_pair_audit.csv` 和 `results.h5/nnws_pairwise`。后者保存完整
`weights[window,pair]`、台站对身份、窗口起止时间、`N_eff`、SNR、对称性、最终损失与失败原因。
失败对不回退到 linear。程序同时生成 `nnws_common_linear`、
`nnws_random_neff_control`，并在启用 Oracle 时生成 `nnws_common_oracle_*` 与
`nnws_common_random_*`；这些策略使用完全相同的成功台站对距离轴。`nnws_pairwise` 未通过
覆盖率门槛时不做频散或 EGF 解释。示例见
`examples/passive_nnws_pairwise_quick.toml`，科学边界见
`docs/NNWS_UPSTREAM_INTEGRATION_CN.md`。

另存两个显式全孔径研究策略：`nnws_hybrid_full_aperture` 在成功对使用 NNWS 权重、在几何不合格
近距对使用单位权重；`nnws_hybrid_random_neff_control` 对成功对置换同一权重分布并保留相同
近距单位权重。它们用于避免只剩远距道导致频散孔径坍缩，但不替代严格共同支持比较，且仍受
`minimum_pair_coverage` 闸门约束。先运行 `scripts/scan_egf_dispersion_aperture.py`，确认理论 EGF
在当前孔径下具有足够频散分辨率，再解释选择/加权后的频散图。

## 9. `[nnws_v2]`：连续时间块交叉拟合共享权重（实验选项）

`nnws_v2` 与作者 NNWS v1 完全分离，不修改 `python_src/nnws/` 的上游实现。它把
每个保存权重都改为连续时间块 out-of-fold 预测，并把同一条时间权重应用到整个
虚拟源道集；训练代理由多个可用距离的正、负支按预测走时对齐后平均得到。

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `enabled` | 启用 NNWS-v2 | `false` |
| `folds` | 连续时间验证块数量 | `5` |
| `epochs`, `learning_rate` | 每折训练轮数和 Adam 学习率 | `200`, `0.001` |
| `arrival_weight` | 主到时窗外 RMS 损失权重 | `0.55` |
| `precursor_weight` | 预测最早到时前能量损失权重 | `0.20` |
| `symmetry_weight` | 主到时窗内弱正负支对称约束 | `0.05` |
| `effective_count_weight` | 低于目标 `N_eff` 时的惩罚权重 | `0.20` |
| `temporal_smoothness_weight` | 相邻训练窗权重平滑项 | `0.02` |
| `minimum_weight` | 每窗非零权重下限 | `0.05` |
| `minimum_effective_fraction` | 最低 `N_eff / N_window` | `0.50` |
| `seed`, `random_control_seed`, `device` | 初始化、置乱控制与设备 | `0`, `271828`, `auto` |
| `maximum_cache_mb` | 逐对 NCF 缓存硬上限 | `512.0` |
| `pair_scope`, `reference_receiver_index` | 台站对范围和参考接收点 | `reference_receiver`, `0` |

输出策略为 `nnws_v2_shared_oof` 与 `nnws_v2_random_neff_control`。后者只置乱
权重—时间窗对应，精确保留权重集合与 `N_eff`。`results.h5/nnws_v2` 保存逐窗
权重、折号、代理矩阵、代理栈、可用台站对掩膜和每折损失；
`nnws_v2_window_audit.csv` 保存时间范围与两种权重。示例见
`examples/passive_nnws_v2_virtual_source_600m_benchmark.toml`。该方法仍是实验消融，
其训练目标改善不能替代理论/主动 EGF、非重叠时间块及频散分辨率检查。

## 10. `[pair_spz]`：台站对—频率 stationary-phase 上限实验

该模块只允许被动竖直分量，并会读取正演的真实源坐标、逐窗随机源系数与活动表。它不是
实测资料可用的源定位器，而是回答“如果源方位和逐窗功率完全已知，物理 SPZ 约束最多能改善
多少”的 Oracle 上限实验。每个源对每个台站对计算路径差相位误差；误差越小越接近该台站对的
stationary-phase 区。优化在整个时间窗集合上同时压低非 SPZ 功率、平衡正反两支，并限制
`N_eff`，因此允许两个各自单边照明的时间窗在总体上互补。

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `enabled` | 启用真源 pair-SPZ 上限实验 | `false` |
| `phase_tolerance_rad` | SPZ 高斯核的相位误差尺度；越小选择越窄 | `π/2` |
| `off_spz_weight` | 非 SPZ 功率占比损失权重 | `0.55` |
| `bilateral_balance_weight` | 整体正反支功率不平衡损失权重 | `0.30` |
| `effective_count_weight` | 低于目标 `N_eff` 的惩罚权重 | `0.15` |
| `minimum_weight` | 每窗非零权重下限 | `0.05` |
| `minimum_effective_fraction` | 最低 `N_eff / N_window` | `0.50` |
| `maximum_iterations` | L-BFGS-B 最大迭代数 | `500` |
| `random_seed` | 同 `N_eff` 置乱控制种子 | `161803` |
| `pair_scope` | `all` 或单虚拟源 `reference_receiver` | `reference_receiver` |
| `reference_receiver_index` | 单虚拟源参考接收点索引 | `0` |

频带沿用 `[source_selection].frequency_bands_hz`。输出 `oracle_pair_spz` 与
`oracle_pair_spz_random_neff_control`；`results.h5/pair_spz` 和
`pair_spz_window_audit.csv` 保存逐窗起止时间、正/负 SPZ 功率、非 SPZ 功率、各频带 SPZ
占比、平衡度、权重和优化结果。与 `[nnws_v2]` 同时启用时，两者必须使用相同
`pair_scope/reference_receiver_index`；程序对数据权重与物理权重取等权 log-space 几何平均，
再强制共同的最低 `N_eff`，额外输出 `nnws_v2_spz_constrained` 和
`nnws_v2_spz_random_neff_control`。

示例 `examples/passive_nnws_v2_pair_spz_600m_benchmark.toml` 中的 `π` 相位尺度与较强双支
平衡是先按源场 SPZ 占比、双支平衡和 `N_eff` 选择的物理消融，不是按某一个频散结果反向
调参。现有 5 种子检验只支持“目标频散拾取更稳”的专项结论，不支持完整 EGF 更真实。

1000 m 架构审计还输出台站对×正负支以及台站对×正负支×频带的真源 SPZ 诊断，并用
`supervised_pair_egf_oracle_upper_bound` 检验这种权重结构是否理论可达。监督 Oracle 直接使用
理论 EGF，属于带目标泄漏的合成上限，绝不能用于实测资料或与可部署方法混报。对应数据自洽
代理 `pair_crossfit_coherence` 虽然只从训练折 NCF 构造，但当前会强化各向异性源偏差，未通过
完整 EGF/频散门。`pair_crossfit_tau_p_spz_frequency` 进一步按训练折慢度、台站对距离、正负支
和频带加权，仍只作为失败消融输出：它改善波形但恶化频散主脊。两者都不是当前推荐策略，详见
`docs/NNWS_RELIABILITY_AUDIT_CN.md`。

## 11. 可复现性与输出

每次运行会保存：

- 原始 TOML 配置副本；
- 解析后的完整参数 JSON；
- `model.csv`、`receivers.csv`、`sources.csv`；
- 正演/NCF/频散结果 HDF5；
- Poisson 模型的 `noise_events.csv` 离散事件目录；
- 理论与拾取 CSV；
- 含模型、密度和源—接收器几何的汇总图；
- 运行时间、线程数、代码版本和随机种子。

随机几何和噪声分别使用明确种子。改变线程数不应改变统计结果，只允许浮点归约造成的微小差异。
