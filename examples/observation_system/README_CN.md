# 观测系统文件输入示例

非均匀噪声源、可编辑线阵/不规则面阵CSV、逐窗振幅权重和CPU workers说明，见相邻[自定义输入包](../custom_geometry/README_CN.md)。

这个目录给出一套可直接查看、上传和运行的观测系统：

- `receivers.csv`：21 个接收器，`receiver_id` 用于人工识别，计算读取 `x_m,y_m`。
- `shots.csv`：3 个主动炮点。界面选择其中一炮；当前正演核心是“一炮一个算例”。
- `noise_sources.csv`：48 个被动噪声源位置，计算直接读取这些坐标，不再随机生成位置。
- `noise_psd.csv`：可直接由界面“PSD CSV”按钮载入的 `frequency_hz,power` 外部噪声谱。
- `active_selected_shot.toml`：读取接收器文件并运行 `S001=(0,0) m`。
- `passive_poisson.toml`：读取接收器和噪声源文件，使用间歇 Poisson-Ricker 激发。

## 文件格式

接收器和噪声源文件的必需表头都是：

```csv
x_m,y_m
```

可以增加 `receiver_id`、`source_id` 等说明列，计算器会读取 `x_m,y_m`。炮点表格式为：

```csv
shot_id,x_m,y_m
S001,0.0,0.0
```

界面允许一次载入多行炮点并选择一行。所选坐标会写入 `[active_source] x_m,y_m`，所以输出 TOML 不依赖原始 Shot CSV 也能复现该炮。

PSD 文件格式为：

```csv
frequency_hz,power
3.0,0.08
7.0,1.00
11.0,0.0
```

频率必须严格递增，功率必须非负；选中后界面会把 `noise.psd_shape` 改为 `file` 并写入绝对路径。

## 被动噪声位置和时间

`layout="file"` 时，噪声位置就是 `noise_sources.csv` 的 48 组坐标；`ambient_sources.seed` 不再改变位置。若改用 `annulus` 或 `line`，空间种子才用于生成位置。

`temporal_model="stationary_gaussian"` 表示每个独立窗内连续存在的带限平稳随机过程，因此没有一张离散“激发时刻表”。

`temporal_model="poisson_ricker"` 才有离散事件。对每个源、每个独立窗：

1. 事件数 `N ~ Poisson(event_rate_hz × window_s)`；
2. 每个事件的窗内时刻 `tau ~ Uniform(0, window_s)`；
3. 极性随机为 ±1，绝对权重在 0.5–1.0 均匀抽样；
4. 第 `w` 个窗的绝对时刻是 `w × window_s + tau`；
5. `noise.seed` 固定整套事件数、时刻、极性和权重。

各源先经 离散波数法 Green 传递函数传播并在接收器端相加，形成没有 source 维度的混合有限记录；随后只对接收器记录分窗互相关/互谱叠加。程序不会先把每个源的 Green 函数互相关后再冒充真实噪声记录。

Poisson 算例运行后会输出 `noise_events.csv`，列为 `window_index,source_index,source_x_m,source_y_m,time_in_window_s,event_time_s,weight`；同一内容也写入 `results.h5/noise_events/`。索引从 0 开始。平稳高斯模型没有离散激发事件，因此不生成这个文件。

## 运行

```bash
conda run -n seisloc wavedispx validate examples/observation_system/active_selected_shot.toml
conda run -n seisloc wavedispx validate examples/observation_system/passive_poisson.toml
conda run -n seisloc wavedispx run examples/observation_system/active_selected_shot.toml
conda run -n seisloc wavedispx run examples/observation_system/passive_poisson.toml
```

`passive_poisson.toml` 的48源、120 s用于演示文件与时间模型，不代表高收敛生产参数。增加源数和独立窗数后仍需固定其他条件检查收敛；192–384源、600–1800 s也不是通用保证。新H/V功率极化目前只支持独立平稳高斯源，不能直接与此Poisson例组合。Studio切换源布局/导入位置会保留已设极化和activity_file，必须核对活动表source_index对应的新坐标行顺序。
