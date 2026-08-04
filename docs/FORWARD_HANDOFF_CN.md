# WaveDispX 波场正演与频散接口移交说明

本文面向只需要接收正演数据、独立开发频散能量谱算法的项目。生产入口、Notebook 和界面均调用
同一套 Python/C++ 代码；下面列出的 HDF5 数组可直接作为算法输入。

## 1. 正演范围与核心

- 介质：一维水平分层、各向同性、弹性或常 Q 衰减；每层输入厚度、Vp、Vs、密度、Qp、Qs；
- 核心：C++17 使用离散波数法（DWM），在频率—水平波数域组装动力刚度并求解 Green 响应；Python 做参数、几何、
  径向逆 Hankel、噪声合成、互相关、输出与验收；
- 空间 Green 基：`RR, TT, ZR, RZ, ZZ`。命名为“接收分量—源力方向”，例如 `ZR` 是径向
  水平力引起的竖直接收位移；
- 理论频散由 `disba` 独立求解，只用于叠加和验收，不进入波场或 NCF 合成。

## 2. 主动源数据

配置示例：`examples/active_high_modes_3c.toml`。

流程：

1. `compute_full_fk_green` 求五分量 f-k Green；
2. `synthesize_full_receivers` 逆 Hankel 到每个源检距；
3. 与 Ricker 子波线性卷积；
4. 输出 `time × receiver × 5` 炮集。

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

## 4. 频散项目的最小接入

主动算法直接读取 `shot/components[:,:,k]` 和 `shot/distance_m`；被动算法优先读取相应
`ncf/components/*/gather` 与 `ncf/distance_m`。不要把 `pair_count` 再乘入已经按距离平均的 NCF；
若算法希望按原始台站对加权，应改为读取/扩展未分箱互谱接口。

本项目内置三种参考成像：phase-shift、Bessel/FJ、Hankel。设置
`processing.compare_methods=true` 会把同一输入、同一频率/速度网格的三张能量写入
`dispersion/method_comparison/{phase_shift,bessel,hankel}`。可用这些结果检查新项目的坐标方向、
FFT 符号、速度轴和频率轴，但不要求复现某一种归一化颜色。

## 5. 可复现运行与建议移交包

```bash
conda run -n seisloc wavedispx validate examples/passive_high_modes_3c.toml
conda run -n seisloc wavedispx run examples/passive_high_modes_3c.toml
```

最小移交包：`input.toml + model.csv + receivers.csv + sources.csv + results.h5 + manifest.json`。
`manifest.json` 记录运行时间、内存、插值误差、随机窗数、质量与逐模态理论曲线能量支持指标。

## 6. 明确没有使用的 tricks

- 没有沿理论频散曲线人工填能量；
- 没有逐噪声源先互相关再叠加；
- 没有把源编号或源位置传入 NCF/频散函数；
- 径向 Green 插值会抽取真实距离做相对 L2 检查；
- 流式处理只是不把 1800 s 全记录常驻内存，与相同独立窗的离线叠加等价；
- 联合 Rayleigh 是 RR/ZR/RZ/ZZ（主动）或 RR/ZZ（被动）单独成像后做列归一 RMS；原分量图始终
  单独保留。它适合检查分量互补性，不能替代各分量的物理解释。

## 7. 已知限制

- 一维层状模型不含横向非均匀、地形、各向异性或非线性；
- 被动等强点力源环并不严格等于闭合边界上的牵引加权 Green 表示，因此源数/时长收敛后，模态
  相对振幅仍可能不同于单炮 Green；
- line/grid/file 的被动频散目前按台站距离径向平均，不保留方位信息；
- 高频高阶同时受模型截止频率、分量耦合、道距混叠、孔径和噪声场照明控制。增加源数只能降低
  随机照明误差，不能制造速度模型不支持或分量不耦合的高阶。
