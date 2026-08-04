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

## 4. `[active_source]`：主动源

仅 `run.mode="active"` 使用：

| 参数 | 单位 | 含义 |
|---|---:|---|
| `x_m`, `y_m` | m | 炮点坐标 |
| `component` | — | `vertical`、`horizontal` 或 `both` |
| `wavelet` | — | 当前为 `ricker` |
| `dominant_frequency_hz` | Hz | Ricker 主频 |
| `maximum_frequency_hz` | Hz | 波场正演最高计算频率 |
| `dt_s` | s | 采样间隔 |

必须满足 `maximum_frequency_hz < 1/(2*dt_s)`。

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

## 6. `[noise]`：噪声 PSD 与时间分布

| 参数 | 单位 | 含义 |
|---|---:|---|
| `temporal_model` | — | `stationary_gaussian` 或 `poisson_ricker` |
| `seed` | — | 噪声时间序列随机种子，与几何种子分开 |
| `duration_s` | s | 总记录长度；推荐 600–1800 s |
| `dt_s` | s | 采样间隔 |
| `window_s` | s | 独立处理窗；当前推荐 20 s |
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
| `compare_methods` | `true` 时在相同网格同时输出三种方法对照；默认 `false`，计算时间约增加 2–3 倍 |
| `frequency_min_hz`, `frequency_max_hz` | 处理频带 |
| `velocity_min_mps`, `velocity_max_mps`, `velocity_step_mps` | 扫描相速度 |
| `distance_taper_fraction` | 距离域 Tukey taper，0–1；推荐 0.5 |
| `zero_padding` | 时间 FFT 补零倍数；主动推荐 5，被动推荐 8 |
| `radial_kernel_lookup_step` | Bessel/Hankel 核参数查表步长，推荐 `0.002`；设 `0` 使用逐点精确特殊函数。仅影响 `bessel`/`hankel` |
| `picker_smoothness` | 拾取的相邻频率速度平滑约束 |
| `picker_maximum_jump_mps` | 相邻频点允许的最大速度跳变 |
| `picker_minimum_energy` | 输出拾取的最小归一化能量 |
| `rayleigh_modes`, `love_modes` | 理论曲线模态数 |

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

## 8. 可复现性与输出

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
