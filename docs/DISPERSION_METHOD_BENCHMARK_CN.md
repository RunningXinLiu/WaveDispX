# Active / passive 多方法频散能量基准

本基准用于比较下列 11 类方法：

- `phase_shift`（PS）
- `frequency_bessel`（FJ / frequency-Bessel）
- `frequency_hankel`（FH / frequency-Hankel）
- `fk`
- `tau_p`
- `linear_radon`（阻尼最小二乘 L2 Radon）
- `high_resolution_radon`（稀疏高分辨率线性 Radon）
- `capon`（MVDR）
- `music`
- `spac`
- `espac`（也接受别名 `esac`）

MATLAB `PhaseShift2_modes3` 的 16 个 PS/FH/pFH `way` 以及 J0/Y0 ×
real/imag/abs 的 FJ 组合另见
[MATLAB 核函数组合基准](DISPERSION_KERNEL_BENCHMARK_CN.md)，避免把 22 个核消融列
挤进本页的 11 类主方法图。

## 为什么不是一个函数接收所有数据

这些方法估计的对象相同，但原始观测量不同：

| 方法 | 输入契约 | 本基准中的坐标 |
|---|---|---|
| PS / FJ / FH | 时间 × 径向距离道集 | 通常为震源—检波器真实距离或虚拟源—台站对距离；active off-axis 对比行例外地只输入投影 `x` |
| FK / τ–p / 两类 Radon / Capon / MUSIC | 时间 × 线性阵列坐标道集 | 阵列投影坐标 |
| SPAC / eSPAC | 台站对归一化互谱 | 二维台站对距离及互谱相干系数 |

因此，SPAC/eSPAC 不会被强行用于 active shot gather。SPAC 的理论关系是
`rho(f,r) = J0(2*pi*f*r/c)`；eSPAC 对任意形状阵列的台站对直接作最小二乘拟合，
SPAC 则先作径向环带平均。其输入和假设与 active MASW 道集不同。

标准 τ–p 采用时域 slant stack 后沿 `tau` 轴做 FFT。`linear_radon` 在每个频率
求解阻尼最小二乘逆问题；高分辨率 Radon 则求解带复数软阈值的稀疏 L1 反演
（FISTA）。Capon/MVDR 与 MUSIC 使用局部频率平滑空间协方差和对角加载。它们与
FK/Bartlett 的常规波束形成、两类 Radon 均为不同的数学估计器，不是重复命名。

### active off-axis：所有方法都不作震源位置校正

off-axis 点源位于 `(0,120) m`，合成波形仍严格使用真实传播距离，故第 `x` 个检波点
满足 `t(x)=t0+sqrt(x^2+120^2)/c`。但这一行刻意不向任何算法提供震源位置：
PS/FJ/FH、FK、τ–p、两类 Radon、Capon 和 MUSIC全部只接收投影阵列坐标 `x`。
各扫描核因此都在用 `t(x)=t0+x/c_app` 一类的线性坐标假设解释曲面波前，预期均与
模型真值失配。该行是“同一道集、同一坐标、无校正”的直接算法对比，不是聚焦或
震源几何校正测试；失配来自有限点源波前曲率，而不是主动源的公共初相位。

### 当前 SPAC/eSPAC 的计算链

SPAC/eSPAC 不先生成时域 NCF 道集。代码对每个时间窗、每个台站做 FFT，累加台站对
互谱 `Sij` 和自谱 `Sii/Sjj`，计算复相干系数
`Cij=Sij/sqrt(Sii*Sjj)`；拟合量是 `real(Cij)`，置信权重为 `|Cij|^2`。SPAC 将
同距离台站对按环带加权平均后拟合 `J0(2*pi*f*r/c)`；eSPAC 不作环带平均，直接使用
全部台站对距离。它们与 NCF 由 Wiener–Khinchin 关系相连，但实现上跳过了 IFFT、
正负时延支路和虚拟源道集成像。

这不意味着 eSPAC 一定更快。本基准中只计成像步骤时，均匀 passive 的 PS 约
0.012 s、SPAC 约 0.23 s、eSPAC 约 2.95 s；eSPAC 较慢是因为它在每个频率和候选
速度上保留全部 300 个台站对，SPAC 环带平均后只保留距离 bins。上述时间均不包含
各方法共享的被动记录合成以及 NCF/相干系数预计算。

## 地下模型、观测系统与场景矩阵

频散真值不再使用人为指定的直线，而是由下列两层弹性模型通过
`disba/dunkin` 正演得到 Rayleigh 基阶相速度。合成和成像范围现为 **1–24 Hz**，
速度扫描为 180–420 m/s；稳定核心评价频带为 3–10.5 Hz，另行报告整个 1–24 Hz
的宽频误差。

passive 阵列孔径为 240 m、道距为 10 m。按本模型频变相速度计算，低于
**2.36 Hz** 时孔径不足 1.5 个波长；高于 **11.35 Hz** 时满足
`2*dx*f/c >= 1`，进入空间混叠区。大图在每个频散面板中以白色半透明区域和
竖点线标出这两个物理风险区，因此扩频不是把有效频带“硬说成更宽”，而是让低频
孔径不足、高频混叠和各算法的旁瓣/失焦直接可见。

| 层 | 厚度 | Vp (m/s) | Vs (m/s) | 密度 (kg/m³) |
|---|---:|---:|---:|---:|
| 低速表层 | 10 m | 380 | 220 | 2000 |
| 半空间 | ∞ | 760 | 440 | 2000 |

active 观测系统为 25 个检波点，`x=20–260 m`、间距 10 m；inline 源位于
`(0,0) m`，off-axis 源位于 `(0,120) m`。passive 观测系统为 25 个检波点，
`x=0–240 m`、间距 10 m，以首个检波点作为虚拟源。完整基准每个 passive case
使用 64 个远场噪声方向和 32 个独立时间窗。HDF5 中的 passive `source_x_m/y_m`
是位于半径 360 m 显示圆上的方位点，用于明确保存和绘制入射方向，不表示有限距震源。

| case | 几何 / 照明 | 目的 |
|---|---|---|
| `active_inline` | active 源位于测线延长线 | 所有道集型方法的正向恢复对照 |
| `active_off_spz` | active 源离测线 120 m；所有方法只输入投影 `x` | 不作震源位置校正的同输入直接对比 |
| `passive_spz` | 双侧端射源，方位在两端 ±8° | 台站对驻相区内的被动 NCF 恢复 |
| `passive_oblique_off_spz` | 双侧斜入射源，30°/210° 附近 ±8° | 非 SPZ、但非垂直阵列的中间表观慢度对照 |
| `passive_off_spz` | 双侧宽边源，方位在 ±90° 附近 ±8° | 驻相区外的预期失效对照 |
| `passive_diffuse` | 0–360° 均匀方位 | SPAC/eSPAC 所需的各向均匀对照 |

`passive_spz`、斜入射 non-SPZ 与宽边 `passive_off_spz` 明确改变源方位，而非
简单换一组随机噪声。
被动记录先经过项目现有的 pair-resolved NCF 流程，再进入各道集型方法；
SPAC/eSPAC 则从同一被动记录独立估计台站对归一化互谱。

## 当前完整基准结果

下表给出“核心频带 MAE / 1–24 Hz 宽频 MAE”（m/s）；`—` 表示方法在该输入上
不适用。驻相区外、投影坐标 stress case，以及超过空间 Nyquist 后的高频误差是
预期物理结果，不算代码失败。

| method | active inline | active off-axis | passive SPZ | passive uniform | oblique non-SPZ | broadside non-SPZ |
|---|---:|---:|---:|---:|---:|---:|
| PS | 0.5/0.8 | 81.7/57.7 | 1.7/2.3 | 6.4/11.4 | 50.0/31.8 | 81.6/60.0 |
| FJ | 5.0/24.7 | 81.2/67.5 | 10.0/26.3 | 14.0/54.6 | 49.8/51.6 | 80.3/65.2 |
| FH | 4.8/4.8 | 81.5/57.8 | 7.6/6.1 | 10.5/35.3 | 54.4/34.2 | 93.6/64.7 |
| FK / Bartlett | 0.5/0.8 | 84.1/59.2 | 1.6/2.2 | 6.1/10.0 | 49.3/31.3 | 82.4/51.8 |
| τ–p | 0.7/0.8 | 84.0/59.3 | 4.3/3.6 | 15.4/16.2 | 51.1/31.8 | 84.7/80.1 |
| LS Radon | 4.3/5.4 | 109.4/162.5 | 8.1/7.1 | 61.7/62.8 | 57.9/35.7 | 111.7/66.0 |
| HR Radon (L1) | 5.4/5.4 | 109.4/162.5 | 1.6/4.7 | 30.8/53.4 | 52.4/33.6 | 110.9/65.9 |
| Capon / MVDR | 0.5/0.8 | 78.5/55.3 | 3.1/3.4 | 10.2/11.9 | 48.1/30.2 | 80.1/54.8 |
| MUSIC | 0.5/0.8 | 85.8/60.5 | 1.5/2.3 | 4.9/11.4 | 49.3/30.1 | 88.1/55.8 |
| SPAC | — | — | 18.6/26.2 | 2.8/3.5 | 19.1/24.1 | 109.2/92.7 |
| eSPAC | — | — | 18.6/26.2 | 2.8/3.6 | 19.1/24.1 | 109.2/92.7 |

主要结论：

1. active inline 和 passive SPZ 中，道集型方法的真值脊清楚；PS、FK、HR-Radon、
   Capon 和 MUSIC 在 SPZ 核心频带的 MAE 为 1.5–3.1 m/s；
2. active off-axis 中，所有方法都只接收投影 `x`，没有任何方法获得真实径向距离；
   因而这一行直接显示各算法面对未校正曲面波前时的失配，不再给 PS/FJ/FH 特殊优势；
3. passive off-SPZ 中所有方法都应失败，算法不会凭空补回缺失的驻相照明；
   30°/210° 斜入射 case 显示约 `c/cos(30°)` 的投影表观速度偏移，误差约
   48–58 m/s；但 PS/FK/τ–p/Capon/MUSIC 相对该表观速度曲线的 MAE 仅为
   3.2–7.6 m/s。90° 宽边 case 则进一步退化为近零时延主导；
4. SPAC/eSPAC 在 uniform case 最好（约 2.8 m/s），但在端射 SPZ case 不是
   公平的主方法，因为端射场不满足各向均匀 SPAC 假设。
5. uniform passive 并不意味着所有估计器都等价：PS/FK/MUSIC 与 SPAC/eSPAC
   恢复良好；LS/稀疏 Radon 对这种双向、有限记录 NCF 更敏感。大图保留这些差异，
   不用统一归一化掩盖失败模式。
6. 每个频率列都独立归一化到 1，因此图像“很亮、很窄”不等于原始相干系数具有
   同样高的绝对信噪比；SPAC 在本合成均匀场中表现突出，是因为源方位、远场平面波、
   独立时间窗和模型恰好满足其 `J0` 假设。

完整配置还执行显式 passive 质量门槛。SPZ 中的 9 个道集型方法，以及均匀方位中
与其物理输入匹配的 PS/FJ/FH/SPAC/eSPAC，合计 14 个必需组合均通过。默认门槛为
ridge MAE ≤10 m/s、12 m/s 容差内支持率 ≥0.85、真值中位能量 ≥0.80；FJ/FH
和 LS-Radon 按有限孔径旁瓣使用放宽门槛。均匀 FJ 的核心 MAE 为 14.0 m/s，但
真值中位能量为 0.94，因此采用 MAE ≤15 m/s、支持率 ≥0.55 的显式旁瓣门槛。
`--quick` 仅用于 smoke test，不声明达到这一统计收敛门槛。

## 全方法单图

主交付物 `broadband_all_methods_atlas.*` 已把所有信息放在一张图中：顶部依次为
地下模型、active 观测系统、SPZ 噪声源、均匀方位噪声源、30°/210° 斜入射
non-SPZ 和 90° 宽边 non-SPZ；下部为 6 个场景 × 11 种方法的频散能量矩阵。
第二行 active off-axis 的所有道集型方法均只输入投影 `x`，不作震源位置校正。
所有面板统一采用 1–24 Hz、180–420 m/s、0–1 色标，青色虚线为模型真值，绿色
点划线是斜入射行的 `c/cos(30°)` 表观速度，右上角数字为“核心/宽频 MAE”。

另外保留原来的 passive 专项图，便于放大查看：

- `passive_wavefield_methods.*`：3 × 6 网格。列依次为 PS、FJ、FH、FK、τ–p、
  HR-Radon；行依次为 SPZ、均匀方位和非 SPZ。前三列接收径向距离道集，后三列
  接收线性阵列坐标道集，图中明确分组标注。
- `passive_spac_methods.*`：3 × 2 网格。列为 SPAC、eSPAC；三行照明条件与上图
  相同。二者输入为台站对归一化互谱，不与时间道集混用。

图上的主要模式为：SPZ 行的六种道集方法均沿真值形成能量脊；均匀方位行中
PS/FH/FK/τ–p 仍能恢复，FJ 旁瓣较强，HR-Radon 对稀疏噪声较敏感；SPAC/eSPAC
在均匀场形成最集中的真值脊。非 SPZ 行则整体失焦，是照明缺失的物理对照。

## 运行

完整基准：

```bash
conda run -n seisloc python scripts/run_dispersion_method_benchmark.py \
  --output output_python/dispersion_method_benchmark
```

快速 smoke test：

```bash
conda run -n seisloc python scripts/run_dispersion_method_benchmark.py \
  --quick --output output_python/dispersion_method_benchmark_quick
```

可通过 `--cases` 和 `--methods` 只运行部分组合。结果包括：

- `metrics.csv`：逐 case / method 的核心频带与 1–24 Hz 宽频 ridge MAE、偏差、
  P90 误差、真值支持率、真值能量和质量门槛结果；
- `benchmark.h5`：分层模型、源—检波点坐标、输入道集、台站对互谱、原始与
  归一化频散能量；
- `manifest.json`：模型、随机种子、输入契约、场景定义和质量门槛总结果；
- `summary.png/.svg/.pdf/.tiff`：模型、active/passive 观测系统、噪声源分布和
  SPZ/均匀/非 SPZ 的主结果图；
- `passive_wavefield_methods.png/.svg/.pdf/.tiff`：六种道集型方法在三种 passive
  照明下的频散能量；
- `passive_spac_methods.png/.svg/.pdf/.tiff`：SPAC/eSPAC 在三种 passive 照明下
  的频散能量；
- `method_matrix.png/.pdf`：全部 case × method 的频散能量矩阵。
- `broadband_all_methods_atlas.png/.svg/.pdf/.tiff`：地下模型、观测系统、四类
  passive 源分布及全部 6 × 11 频散能量结果的单图总览。

## 方法依据与边界

- SPAC 的随机场基础：K. Aki (1957), DOI
  [10.15083/0000033938](https://doi.org/10.15083/0000033938)。
- 任意阵列 eSPAC/ESAC：Ohori, Nobata & Wakamatsu (2002), DOI
  [10.1785/0119980109](https://doi.org/10.1785/0119980109)。
- τ–p / p–omega 波场变换：McMechan & Yedlin (1981), DOI
  [10.1190/1.1441225](https://doi.org/10.1190/1.1441225)。
- 高分辨率线性 Radon：Luo et al. (2008), DOI
  [10.1007/s00024-008-0338-4](https://doi.org/10.1007/s00024-008-0338-4)。

这一层首先是带精确频散真值的算法 gate。它使用上述分层模型的 Rayleigh 基阶相速度
合成受控色散波场，以隔离照明、坐标契约和算法误差；被动相关部分复用项目现有 NCF
实现。它仍不替代完整的分层介质 DWM/f-k 波场正演，因为当前受控案例不包含多模态、
几何扩散、散射与衰减。
下一层可直接把相同方法适配器接到 `simulate_active_shot`、`simulate_passive` 或已有
`results.h5`，用于真实分层模型、多模态、衰减和有限记录长度的系统测试。
