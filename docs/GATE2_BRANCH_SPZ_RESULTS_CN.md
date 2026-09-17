# Gate 2 v5.2 正负支 stationary-phase oracle 结果

## 1. 结论

v5.2 的 truth-only `oracle_branch_spz` 比 v5.1 的共同角向均匀化明显更接近正确机制：四个场景的
symmetric waveform CC 全部提高，单位范数 L2 和 pre-arrival 能量全部下降，并且四个场景都优于
20 次匹配时间置乱。near12 的 positive branch 和 mirror168 的 negative branch 也按预期得到主要
改善。

但是，这个结果仍未通过预注册联合门：far75 和 mirror168 各有一个相反支的 CC 下降，far75 和
near12 的 wrapped phase RMS 增加；四个场景的 FTAN 都没有同时改善群速度图、相速度图、QEDisp
群速度理论线能量和正确相速度周期支优势。由于连 source-truth oracle 都未通过，v5.2 不运行
`beam_branch_spz`，不运行盲测种子 5701–5705，也不进入真实资料。

准确的降级结论是：**正负支独立的 SPZ 时间窗权重能改善 symmetric NCF 的总体波形相似度和
precursor，但在当前 25 s 多事件混合窗上不能保证两支、相位和正确频散周期支同时保真。** 不能把
它称为提高了完整 EGF 等价性。

## 2. 冻结协议与符号约定

- 复用 v5.1 开发种子 5501 的同一批 600 个正/负支短 CCF、事件真值方位谱和 active EGF；
- 0°（东侧）源固定对应 positive branch，180°（西侧）源固定对应 negative branch；该约定由两个
  单一延迟波形的相关算子单元测试在运行前确定；
- 每支目标为相应 endfire 方位 ±20° 的归一化 top-hat；正则项 0.05；
  `N_eff >= 0.5M`；
- 正、负权重独立优化、分别叠加，再形成 symmetric branch；
- event truth 只进入 oracle，active EGF 和 QEDisp 仅评价；
- 每支 20 次独立时间置乱；FTAN 不使用自动拾取。

四个场景的正、负支 `N_eff` 都达到预注册下限 300/600。这说明目标很强，优化器已被有效窗约束
限制；本结果没有在看到输出后放松该约束或调 SPZ 宽度。

## 3. 波形、两支和置乱对照

成功方向为 symmetric CC 增、L2 降、phase RMS 降、pre-arrival 不增，且 positive/negative 两支
CC 都增。下表均为 `oracle_branch_spz - linear`：

| 场景 | Δ symmetric CC | Δ L2 | Δ phase RMS (rad) | Δ pre-arrival | Δ positive CC | Δ negative CC | 优于匹配置乱 | 联合结果 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| uniform | +0.029848 | -0.057433 | -0.125182 | -0.003112 | +0.062888 | -0.012544 | 是 | 失败 |
| far75 | +0.020707 | -0.034714 | +0.045714 | -0.015223 | +0.021686 | -0.002682 | 是 | 失败 |
| near12 | +0.049029 | -0.071227 | +0.047610 | -0.011627 | +0.124820 | +0.008538 | 是 | 失败 |
| mirror168 | +0.016497 | -0.026035 | -0.026262 | -0.012175 | -0.027596 | +0.068293 | 是 | 失败 |

near12 的两支 CC 都提高，但 phase RMS 增加；mirror168 的目标 negative branch 明显提高，但
positive branch 下降；far75 的 symmetric CC/L2/precursor 变好，却同时损害 phase 和 negative
branch。因此这里确有可重复的局部机制，不是随机权重，但还不是完整 EGF 改善。

## 4. FTAN 独立门

四项指标均相对 linear；成功需要全部非劣/改善：

| 场景 | Δ group image CC | Δ phase image CC | Δ group theory energy | Δ correct-branch advantage | 结果 |
|---|---:|---:|---:|---:|---|
| uniform | -0.028296 | +0.038938 | -0.001842 | -0.002379 | 失败 |
| far75 | +0.174163 | +0.095983 | +0.044164 | -0.002615 | 失败 |
| near12 | -0.054195 | -0.008205 | -0.001817 | -0.000268 | 失败 |
| mirror168 | -0.036450 | -0.006792 | +0.004073 | -0.002612 | 失败 |

far75 是最容易被“图变好看”误判的场景：群/相速度图像相关和 QEDisp 群速度理论线能量都明显
提高，但正确相速度周期支相对相邻 ±1 周支的优势下降，且 phase RMS 和 negative branch 退化。
因此仍判失败。near12 的 symmetric CC 增量最大，但 FTAN 四项都下降，也不能用单一波形 CC
替代频散保真。

## 5. 为什么停止 beam 与盲测

v5.2 预注册顺序是 truth-only oracle → beam proxy → blind。oracle 的作用是先回答：即使知道每窗
真实事件方位，只靠整窗 branch-SPZ 权重是否足以恢复 EGF。答案是“只能部分改善”。此时运行 beam
只会把一个更不准确的方位代理放到已经失败的控制变量上，无法支持方法有效性，所以按计划停止。

源审计图还显示，positive/negative 权重虽分别增强 0°/180°，但同一 25 s 窗内其他方向事件会被
一起放大或压制。near12 的强 12°事件也会通过窗共现影响 negative 权重。这是 whole-window
identifiability 的限制，不是继续调正则项即可解决的问题。

若后续继续 source-aware 路线，必须换研究单元：保存 4 s beam snapshot 对应的目标台对原始波形，
在相同 snapshot 上构造 CCF/事件级权重，或使用能在窗内分离相干源的协方差子空间、盲分离或
相关源校正。这个更改会重定义短 CCF 样本及随机对照，必须作为新版本预注册，不能在 v5.2 内
偷偷缩窗。

## 6. 可追溯输出

- `output_python/gate2_branch_spz_development/gate2_branch_spz_summary.json`；
- `output_python/gate2_branch_spz_development/summary/gate2_branch_spz_complete_evidence.csv`；
- `output_python/gate2_branch_spz_development/summary/gate2_branch_spz_evidence_summary.json`；
- `output_python/gate2_branch_spz_development/figures/`：每场景正/负支波形、group/phase FTAN、源场
  和分支权重审计，SVG/PDF/PNG/TIFF 四格式。

