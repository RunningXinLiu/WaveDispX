# MATLAB PS/FH/pFH/FJ 核函数组合基准

本基准补充主方法对比，专门回答：在相同地下模型、观测系统、波形和扫描网格下，
复数叠加量采用 `abs`、`real`、`imag`、`pi/4` 相位旋转、`sqrt(r)` 或 `r*dr`
径向权重时，频散能量会怎样变化。

所有 `way=1–16` 均按 MATLAB `PhaseShift2_modes3` 的公式映射。其输入频谱先逐道
相位归一化为 `F/abs(F)`。PS 核为 `exp(i*k*r)`；FH 核为一阶 Hankel 函数
`H0^(1)(k*r)`；pFH 只保留 Hankel 相位 `H0/abs(H0)`。FH/pFH 使用默认
`r*dr` 径向积分权重。

| MATLAB way | Python 方法 | 核与输出 |
|---:|---|---|
| 1 | `ps_abs` | `abs(sum(exp(i*k*r)*F))` |
| 2 | `ps_real` | `abs(real(E_PS))` |
| 3 | `ps_real_sqrt_r` | `abs(real(sum(sqrt(r)*exp(i*k*r)*F)))` |
| 4 | `fh_real_r` | `abs(real(E_FH))`，带 `r*dr` |
| 5 | `pfh_real_r` | `abs(real(E_pFH))`，带 `r*dr` |
| 6 | `fh_abs_r` | `abs(E_FH)`，带 `r*dr` |
| 7 | `ps_real_pi4` | `abs(real(E_PS*exp(-i*pi/4)))` |
| 8 | `pfh_abs_r` | `abs(E_pFH)`，带 `r*dr` |
| 9 | `ps_real_pi4_sqrt_r` | `sqrt(r)` PS 后乘 `exp(-i*pi/4)` |
| 10 | `ps_imag` | `abs(imag(E_PS))` |
| 11 | `fh_imag_r` | `abs(imag(E_FH))`，带 `r*dr` |
| 12 | `pfh_imag_r` | `abs(imag(E_pFH))`，带 `r*dr` |
| 13 | `ps_real_raw` | 保存有符号 `real(E_PS)`；能量显示取绝对值 |
| 14 | `fh_real_raw_r` | 保存有符号 `real(E_FH)`；能量显示取绝对值 |
| 15 | `pfh_real_raw_r` | 保存有符号 `real(E_pFH)`；能量显示取绝对值 |
| 16 | `fh_real_pi4_r` | `abs(real(E_FH*exp(+i*pi/4)))`，带 `r*dr` |

`way=13–15` 是诊断输出，不是新的非负能量定义。逐频取绝对值并归一化后，它们必然
分别与 `way=2、4、5` 完全相同；其价值在于 HDF5 的 `raw_amplitude` 中保留正负号。

另外实现 MATLAB `PhaseShift2_modes_FJ` 的六种组合：

| Python 方法 | 径向变换 |
|---|---|
| `fj_real_r` / `fj_imag_r` / `fj_abs_r` | `J0(k*r)` × `Re(F)` / `Im(F)` / `abs(F)`，带 `r*dr` |
| `fy_real_r` / `fy_imag_r` / `fy_abs_r` | `Y0(k*r)` × `Re(F)` / `Im(F)` / `abs(F)`，带 `r*dr` |

这里将 `Y0` 明确标为 FY，避免把实值 Neumann 核与复值 Hankel `H0^(1)=J0+iY0`
都简称为 FH。FJ/FY 六项保留原 MATLAB 文件的未白化频谱分量，因此它们与
`way=1–16` 的差异同时包含“核函数”和“频谱预处理”两部分；不能把全部差异只归因于核。

## 测试场景与公平性

22 个变体全部使用主基准的六个场景：active inline、active off-axis、passive SPZ、
uniform passive、30°/210° oblique non-SPZ 和 90° broadside non-SPZ。处理范围统一为
1–24 Hz、180–420 m/s，核心评价频带为 3–10.5 Hz。

active off-axis 行仍严格不作震源位置校正：波形按真实离轴距离合成，但所有核只接收
投影 `x`。因此该行用于比较不同核面对同一个未校正曲面波前的表现，不给 PS/FH
提供真实震源距。

## 完整分辨率结果摘要

以下误差均为 3–10.5 Hz 核心频带 ridge MAE；完整 1–24 Hz 指标见 `metrics.csv`。

- active inline 原始炮集中，保留复数叠加模长最稳：`ps_abs` 为 0.5 m/s，
  `pfh_abs_r` 为 0.6 m/s；只取 real、imag 或固定 `pi/4` 投影后，PS 系列误差增至
  12.1–13.8 m/s。这说明主动源子波的整体频域相位会直接控制 real/imag 投影，
  而 `abs(E)` 对公共相位旋转不敏感。
- passive SPZ 的近似对称 NCF 中，`ps_real_sqrt_r`、`fh_real_pi4_r`、`ps_real` 和
  `pfh_abs_r` 最好，分别约为 1.4、1.4、1.5 和 1.5 m/s。这里 FH 的 `+pi/4`
  补偿与二维 Hankel 渐近相位相符，但同一固定旋转不能直接移植到所有输入类型。
- uniform passive 中，`ps_real_pi4`、`ps_real_pi4_sqrt_r`、`pfh_real_r` 和
  `fh_real_r` 的核心误差约为 3.4–3.9 m/s；`ps_abs` 为 6.4 m/s。宽频误差明显更大，
  主要集中在低频孔径不足和高频空间混叠区，因此不能仅按核心频带排名宣布一个
  “全频最优核”。
- active off-axis 的全部变体仍有至少约 61.5 m/s 核心误差。改变 real/imag、
  `sqrt(r)`、`pi/4` 或 Hankel/Bessel 核都不能替代震源几何校正。
- oblique/broadside non-SPZ 中不存在能稳定恢复真值的核。斜入射结果主要跟随
  投影表观速度；宽边照明则退化为近零时延主导。这说明核函数不能补回缺失的驻相照明。
- `J0/Y0 × abs(F)` 在 active inline、passive SPZ 和 uniform passive 中的核心误差
  约为 55–80 m/s，明显差于保留有符号 real/imag 分量的 FJ/FY。先取 `abs(F)` 会
  丢失径向振荡的符号/相位信息，因此不能把它当作更稳健的“正能量”输入。
- `way=13–15` 的归一化能量与 `way=2、4、5` 数值完全一致；差别只存在于保存的
  有符号 `raw_amplitude`，这也验证了 MATLAB 注释中的诊断模式定义。

## 运行与输出

完整计算：

```bash
conda run -n seisloc python scripts/run_dispersion_kernel_benchmark.py \
  --output output_python/dispersion_kernel_benchmark
```

输出包括：

- `metrics.csv`：六个场景 × 22 个变体的核心/宽频 MAE、真值支持率和运行时间；
- `benchmark.h5`：每个变体的归一化能量与未归一化 `raw_amplitude`；
- `manifest.json`：MATLAB way 映射、输入契约、模型和场景定义；
- `matlab_kernel_variant_atlas.png/.pdf/.svg/.tiff`：全部 6 × 22 结果的单图总览。
