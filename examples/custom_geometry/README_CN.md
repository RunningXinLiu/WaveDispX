# 自定义台站、噪声源与计算资源

这是可复制修改的小型输入包，使用现有WaveDispX格式，不需要编程。所有例子是1D水平层状模型、地表节点三分量、真实有限记录噪声相关；**非均匀源不等于介质各向异性**。短记录和短孔径只用于熟悉输入，不保证频散收敛。HVSR默认不计算。

## 在界面中使用

1. 右上「示例与说明」载入下列例子，或「打开自定义输入示例文件夹」复制整个目录到自己的工作目录。
2. 用文本编辑器或表格软件修改CSV（保存为UTF-8逗号分隔，不保留Excel索引列）。在TOML中修改模型、路径和源数；路径相对于TOML所在目录，不是启动命令所在目录。
3. 「导入CSV」可分别换接收器、噪声源、PSD及逐窗活动表。活动表还有明确的移除入口，不必手改TOML。
4. 「1 检查与预览」确认真实Geometry、源数量、模型和频带，再「2 运行计算」。切换源位置后必须核对活动表对应的行顺序。载入整个模板会替换全套输入；单独导入源坐标则保留极化/方向等已设参数。
5. 「⑤ 计算 / 理论」集中设置workers和分块大小。完整输入可另存TOML；大型试验输出请使用服务器工作目录。

## 三个可运行例子

| TOML | 台站文件 | 噪声源文件 | 设计含义 |
|---|---|---|---|
| passive_east_line.toml | receivers_line.csv：41站，x=−100…100m、5m间距 | sources_east.csv：12源都在东侧 | 单侧照明，NCF正负支可能不同；不能用长记录补上缺失源方位 |
| passive_clusters_array.toml | receivers_irregular_array.csv：25站不规则面阵 | sources_two_clusters.csv：东北8源、西南4源 | 源空间密度不均；台站面阵依旧使用1D地下模型 |
| passive_switching_array.toml | 同上 | 同上，另读source_activity_switching.csv | 同位置、同极化，只改变前后时间段的源活动强度 |

三例都是60s、6个10s独立窗、dt=.01s，源水平PSD份额.6、独立H/V，无优选水平力方向；PSD峰7Hz、3–10Hz。位置方位与力方向不是同一概念。物理参数及随机种子写在各自TOML中，不会由文件名隐式决定。使用Studio运行新energy_theory目标；不要把它交给不支持新产品的旧CLI然后期待相同处理。

## 坐标文件：单位、编号和顺序

接收器、噪声源至少需要两列：

```csv
x_m,y_m
300,100
320,150
```

| 字段 | 含义 | 约束 |
|---|---|---|
| x_m / y_m | 同一局部笛卡尔坐标系的位置，单位m；+X和+Y正交 | 有限数字；不是经纬度，不能直接填GPS度数 |
| receiver_id / source_id | 本输入包用于人阅读的可选标签 | 不用于计算索引或文件关联；可删除，不能代替坐标 |

台站坐标不能重复，源不要与接收器重合；坐标定义在地表，当前CSV没有可用的深度、逐源力方向、单源PSD等列。不要增加这些列后以为求解器会读取。源CSV有效数据行数必须等于ambient_sources.count；界面导入会更新count，直接改TOML则要自己维护。文件型源不受空间seed改变，但noise.seed仍控制随机时间过程。

```toml
[geometry.receivers]
layout = "file"
coordinates_file = "receivers_irregular_array.csv"
distance_bin_m = 5.0

[ambient_sources]
layout = "file"
count = 12
coordinates_file = "sources_two_clusters.csv"
component = "three_component"
polarization = "independent_hv"
horizontal_power_fraction = 0.6
horizontal_azimuth_deg = 0.0
```

distance_bin_m是被动面阵/文件阵列按台站对距离聚合的箱宽，不是台距，也不改善空间采样。空间孔径、典型台距、源照明和频率范围仍需检查。上面只是字段片段，运行时使用本目录完整TOML。

## 活动表：不均匀强度及随时间变化

表头严格使用：

```csv
window_index,source_index,amplitude_multiplier
0,0,1
0,1,0.2
```

- window_index从0开始：第w窗对应[w×window_s, (w+1)×window_s)。它不是某个离散事件的激发时刻。
- source_index从0开始：对应源坐标CSV的第index+1个数据行，不包括表头。不按source_id字符串匹配；重排源CSV时须同时更新活动表。
- amplitude_multiplier是非负**振幅倍数**。0.2意味着该源功率乘0.04；不是功率20%。希望功率为25%时填0.5。
- 未列出的窗/源组合为0；每窗至少一个正权重，不能有重复组合，索引不能超出源数和窗数。不使用活动表时所有源每窗倍数均为1。
- 本例前0–2窗：东北8源倍数1，西南4源倍数.2；后3–5窗反过来。它同时改变照明和总源功率，不是恒定总功率实验；如需恒定总功率，请事先按每窗Σa²归一化并保存明确权重。
- 改duration_s/window_s后必须重新定义活动表。不能直接延长记录并假定新窗口会自动沿用旧表。

本目录不包含逐源方向/事件目录导入；directed_hv只支持所有源共同的水平力方向。stationary_gaussian是连续随机源，不产生离散激发时刻清单。Poisson时刻模型见相邻observation_system目录，但新加权H/V极化暂不与Poisson组合。

## CPU与分块选项

| 参数 | 输入 | 实际作用 |
|---|---|---|
| run.workers | 非负整数；0=自动，1=单工作线程 | 控制C++正演及支持并行的径向变换，不是进程数/并行炮数；不限制所有NumPy、BLAS或QED线程 |
| run.frequency_block_size | 正整数 | 频率分批处理，影响内存和调度；不是频率df |
| run.source_block_size | 正整数 | 混合噪声时每批处理多少源；不是源总数 |

模板保留workers=4、两种分块=8；可先试workers=1/2/4。0可能使用更多硬件线程但不保证更快；小任务线程开销可能反而占主导。分块调小可能降低临时内存，但不保证总内存成比例下降。这些是运行选项，不改变模型、源数量、记录时长或f/v网格；本界面不是GPU或远程服务器提交器。

## 常见问题

- 提示找不到文件：先检查相对于TOML目录的路径，复制时保持整个文件夹结构。
- 源数不匹配：count必须等于源CSV行数；接收器数量由文件读取。
- 活动索引越界：核对源数据行数与duration_s/window_s；编号从0开始，不是标签S001。
- R/L只有一种有能量：检查源极化和方向；别通过选择波型期待程序自动改变源。
- 旁瓣/NCF不对称：这些示例有意给出非均匀照明，不以均匀源结果作为必然目标，也不代表程序自动剔除了体波或做了纯模态分离。

更多炮点、PSD和Poisson格式见[观测系统输入说明](../observation_system/README_CN.md)。

这三个例子的完整窗口、模型/Geometry、双边NCF和R/L能量见[运行截图与验证记录](../../docs/studio/custom_inputs_20261028/README_zh.md)。图中宽峰和波动是保留的实际短记录结果，不作为收敛展示。
