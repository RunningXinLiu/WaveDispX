# 双台 FTAN 实现与 Gate 0 验收

> **状态：Gate 0 已通过，但公共频带限定为 4–12 Hz（2026-08-15）**
>
> 2–12 Hz 仍作为完整候选频带展示；其中 2–4 Hz 的主动源相速度有效，但 600 m 双台群速度
> 包络不满足分辨率门槛，因此不得进入正式 NCF 筛选效果比较。唯一执行计划见
> [`NCF_EGF_EQUIVALENCE_PLAN_CN.md`](NCF_EGF_EQUIVALENCE_PLAN_CN.md)。

## 1. 当前结论

此前“群速度已经验证”的表述过强。重新审计得到更精确的结论：

- `pair_ftan` 的群速度包络算子在常速度和人工规定色散的解析 Green function 上通过严格真值测试；
- 当前 1-D active EGF 的 2–12 Hz 相位传播与 QEDisp 一致；
- 当前两层模型在 2–4 Hz 有很强的基阶 Rayleigh 群速度曲率，600 m 有限距离的 Gaussian
  窄带包络不能在整个低频段贴合微分意义的瞬时群速度；
- 4–12 Hz 同时满足 active 波形、群速度和相速度门槛，因此冻结为后续所有双台方法的公共频带；
- 这不是把低频曲线“修好看”，而是保留失败区、给出原因并排除不合格测量区。

核心入口为 `wavedispx.processing.pair_ftan`。NCF 正负支必须在外部拆分、统一传播方向后分别送入，
FTAN 内部不平均两支。

## 2. 物理定义

群速度图使用 Gaussian 窄带滤波和解析信号包络：

\[
H(f;f_c)=\exp\left[-\alpha(f-f_c)^2/f_c^2\right],\qquad
U=r/t_{\mathrm{env}} .
\]

相速度主图使用频域远场相位相干：

\[
E_c(T,c)=\cos\left[-\left(2\pi f r/c+\pi/4\right)-\phi_{\mathrm{obs}}(f)\right].
\]

对 correlation 输入先执行 `imag(hilbert(CF))` 转为 EGF-like 波形，符号与提供的
`EGFAnalysisTimeFreq.m` 一致。相速度图必然存在相差 (2\pi) 的多条等价脊线；绝对支需要独立
先验、人工可信点或跨台距相位连续性。主图不画自动拾取线。

## 3. QEDisp 原生群速度

QEDispInv 的 `sregn96`/`slegn96` Fortran 求解器会通过能量积分直接返回 `cg`，但上游
`forward` 命令行只写出相速度，原 C++ 包装层没有暴露 `cg`。当前本地适配做了两件事：

1. `scripts/install_qedispinv_local.py` 在复制完整 QEDispInv 后，用其原始 Fortran 源码编译本地
   `libqedisp_group.dylib` 或 `libqedisp_group.so`；
2. `run_qedisp_forward(..., compute_group=True)` 通过本地动态库调用 `sregn96`/`slegn96`，返回
   `rayleigh_group_mps` 和 `love_group_mps`。

理论群速度不再依赖相速度数值微分。交叉核对表明，在当前密集频率网格上，原生 `cg` 与
`U=(dk/d\omega)^{-1}` 的中位差为 0.013 m/s、90% 差为 0.061 m/s；因此旧微分不是当前
2–4 Hz 偏差的主因，但原生值消除了理论来源争议。

QEDispInv 完整目录及编译产物位于 Git 忽略的 `.wavedispx-tools/QEDispInv/`，不随本项目提交。
重新安装命令为：

```bash
conda run -n seisloc python scripts/install_qedispinv_local.py /path/to/QEDispInv-main
```

分发 QEDispInv 或其派生动态库前必须单独复核上游许可证；当前用途是用户本机验证。

## 4. G0-A：解析真值

严格回归不使用连续脊线自动拾取，而逐周期取群速度能量最大位置，与已知真值比较。验收门槛为：

- 中位绝对误差不超过一个速度网格（2 m/s）；
- 90% 绝对误差不超过两个速度网格（4 m/s）。

测试覆盖：

- 常速度 (c=U=300 m/s)；
- 人工规定的连续色散 (c(T)) 及由其解析传播谱产生的 EGF；
- 正负极性；
- EGF 与 CF→EGF 两种输入；
- 300、600、900 m 距离；
- Gaussian `alpha=20,40,80`；
- Hilbert 符号、Nyquist、无相位锚点不得报告有效绝对相速支。

当前常速度误差基本为 0；人工色散的中位误差约 0.75 m/s，90% 误差约 1.74 m/s。

## 5. G0-B：单一 1-D active EGF

固定配置：

- 模型：20 m 厚 `Vs=220 m/s` 层覆盖 `Vs=440 m/s` 半空间；
- active：vertical unit impulse；
- 分量：vertical–vertical（ZZ）；
- 台距：600 m；
- 候选频带：2–12 Hz；
- FTAN：2 m/s 速度网格，`alpha=120`；
- 理论：QEDisp 基阶 Rayleigh 相速度和原生群速度；
- 主图：波形、(T-U)、(T-c)，只画理论线。

不依赖 tracked picker 的结果：

| 测量 | 频带 | 中位绝对误差 | 90% 绝对误差 | 4 m/s 内比例 |
|---|---:|---:|---:|---:|
| 群速度能量最大位置 | 2–12 Hz | 3.18 m/s | 16.83 m/s | 53.2% |
| 群速度能量最大位置 | **4–12 Hz** | **0.91 m/s** | **2.54 m/s** | **100%** |
| 离理论最近的相位亮脊 | 2–12 Hz | 1.01 m/s | 2.89 m/s | 98.5% |
| 离理论最近的相位亮脊 | **4–12 Hz** | **0.61 m/s** | **1.17 m/s** | **100%** |
| active 阵列相速度主峰 | 2–12 Hz | 0.60 m/s | 1.09 m/s | 99.9% |

诊断排除了以下主因：

- 分量索引错误：HDF5 索引 4 明确为 vertical–vertical；
- 群速度理论微分错误：QEDisp 原生 `cg` 与微分结果一致；
- 高阶模态主导：active 阵列和 pair FTAN 的主要能量均为基阶 Rayleigh；
- FFT 零填充不足：8192 与 262144 点结果不变；
- active 正演相位错误：阵列相速度谱在 2–12 Hz 与 QEDisp 高度一致。

支持的解释是时频分辨率限制：2–4 Hz 正好跨越该模型群速度从约 323 m/s 急降到约
172 m/s 的区间。收窄 Gaussian 频带会改善 3.5–4 Hz，却因时间展宽损害 2–3 Hz；不存在一个
固定 `alpha` 能让 600 m pair 在整个 2–12 Hz 同时通过。4–12 Hz 则稳定通过。

复算命令：

```bash
conda run -n seisloc python scripts/plot_active_pair_ftan_qc.py
```

输出目录：`output_python/pair_ftan_gate0/`，包含 PNG/PDF/SVG/TIFF、逐周期 source-data CSV 和
JSON Gate 摘要。

## 6. 后续使用约束

- 后续 linear、RMSR、temporal reweighting、NNWS 和随机对照全部固定使用 4–12 Hz；不得按方法
  改频带或重新调 `alpha`。
- 所有正式双台图均同时展示对应 NCF 波形、群速度图和相速度图；理论线来自同一个 QEDisp 模型。
- 2–4 Hz 可以保留在诊断图中，但只能标为未通过的候选区，不能进入方法优劣指标。
- `alpha=120` 是本次 600 m、4–12 Hz Gate 0 的仪器标定，不是可直接推广到其他站距的经验公式。
- 若未来要求保留 2 Hz，必须在主计划中另立变更：重新设计更平滑的单一 1-D 模型、增加有效台距/
  记录长度，或改用经独立真值验证的多分辨率群速度方法，不能直接放宽误差门槛。
- 实际 NCF 的高 SNR、图面更亮或自动线更平滑仍不等于 EGF 更正确。
