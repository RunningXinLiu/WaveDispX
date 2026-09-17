# Gate 2 v5.0 稀疏事件双台筛选盲测结果

## 结论先行

Gate 2 v5.0 的合成基准有效、方向簇污染确实降低 NCF–active EGF 一致性，而且同一组短时 CCF
在真值知情的非负权重下数学上可以重组到接近 active EGF。然而，RMSR-SS、StackMaster robust、
Li temporal 单台对适配和 Fan 原版 NNWS **没有任何一个通过预注册的多指标、五盲种子判据**。

因此本轮结果不是“筛选有效但图不够好”，而是更严格的负结论：在当前 600 m、4–12 Hz、
one-bit + whitening、约 10,000 个背景瞬态事件并叠加约 10% 定向事件的双台问题中，这些只看
短时双台 CCF 的方法没有可靠学会压低定向污染源。当前不应把任何方法直接用于真实资料并宣称
提高了 EGF 质量。

## 1. 为什么从 v4.9 改为 v5.0

v4.9 让 576 个背景源在每个 25 s 窗中连续同时激发。它能证明持续方向偏置会损害 NCF，但每个
窗口都混有几乎同样的源群，缺乏 Xie (2020) 和 Fan (2026) 合成试验中的“可按时间选择的稀疏
episode”。重新核对正文、补充材料和公开代码后，v5.0 在运行前冻结为文献一致的稀疏瞬态设计：

- 600 个互不重叠的 25 s 窗，共 15,000 s；
- 576 个均匀背景位置，三个偏置场各增加 58 个 12° 扇区位置；
- 背景期望 10,000 个 Poisson-Ricker 事件，扇区期望约 1,007 个事件；开发实现为 10,145 +
  1,039 个事件；
- 8 Hz Ricker 主频，生成带 3.5–12.5 Hz，统一评价带 4–12 Hz；
- 两台记录分别 one-bit，再谱白化、互相关；
- 垂直力源、ZZ 基阶 Rayleigh、600 m 站距和 Gate 0 已验收的单一 1-D 模型；
- far75、near12、mirror168 三个方向簇与 uniform 控制；
- 对每个偏置场保存完全配对的无扇区事件 `paired_clean_counterfactual`；
- 开发种子 5301；冻结后盲测种子 5401–5405，NNWS 初始化 1001–1005；
- 每个非线性方法固定 20 次同权重分布的时间置乱对照。

v4.9 继续保留为“持续混合源负控制”，v5.0 才是对 Xie/Fan 时间窗筛选机制的主检验。两个结果
并不矛盾。

## 2. 先验证实验本身是否成立

预注册基线要求 uniform 和每个 paired-clean symmetric linear 同时满足 active-EGF
`CC >= 0.65`、wrapped phase RMS `<= 1.2 rad`。开发种子结果为：

| 基线 | waveform CC | phase RMS (rad) | 判定 |
|---|---:|---:|---|
| uniform linear | 0.863 | 0.840 | 通过 |
| paired clean（各偏置场相同背景实现） | 0.761 | 0.887 | 通过 |

加入方向簇后，far75、near12、mirror168 的 linear waveform CC 分别降至 0.732、0.557、
0.689。也就是说，基准波场能恢复 active EGF，而方向簇确实造成可测量退化；本试验不是在一个
本来就无效的 NCF 上比较筛选器。

冻结的五个盲测种子也全部通过该基线门：uniform linear 的 CC 最小值为 0.813、phase RMS
最大值为 0.653 rad；paired-clean 的 CC 最小值为 0.824、phase RMS 最大值为 0.703 rad。
盲测汇总中的 `baseline_gate_passed=true` 因此不是由开发种子外推得到。

## 3. 五个盲测种子的主要结果

下表只列可部署方法。数值为 `method - linear` 的盲测中位差；CC 为正、L2/phase 为负才是改善。
“改善种子”按相应单项计数，不代表同时满足全部 Gate。

| 场景 | 方法 | 成功运行种子 | ΔCC | CC/L2 同向改善种子 | Δphase (rad) | phase 改善种子 | 扇区/背景事件权重比 |
|---|---|---:|---:|---:|---:|---:|---:|
| far75 | RMSR-SS | 5 | +0.00028 | 3/5 | -0.0162 | 4/5 | 1.0004 |
| far75 | robust | 5 | -0.0173 | 1/5 | +0.0069 | 2/5 | 0.9886 |
| far75 | temporal adapter | 5 | -0.1680 | 0/5 | +0.3697 | 0/5 | 1.0154 |
| far75 | NNWS upstream | 1 | +0.00069 | 1/1 | -0.0043 | 1/1 | 0.9999 |
| near12 | RMSR-SS | 5 | +0.00061 | 3/5 | -0.0048 | 4/5 | 0.9996 |
| near12 | robust | 5 | -0.0319 | 1/5 | +0.0468 | 2/5 | 1.0580 |
| near12 | temporal adapter | 5 | -0.0274 | 1/5 | +0.1114 | 0/5 | 0.9493 |
| near12 | NNWS upstream | 4 | +0.00010 | 3/4 | -0.0051 | 4/4 | 1.0010 |
| mirror168 | RMSR-SS | 5 | -0.00414 | 2/5 | +0.0080 | 2/5 | 0.9993 |
| mirror168 | robust | 5 | -0.00979 | 1/5 | +0.0682 | 2/5 | 1.0434 |
| mirror168 | temporal adapter | 5 | -0.1533 | 0/5 | +0.3299 | 0/5 | 0.9899 |
| mirror168 | NNWS upstream | 2 | -0.00006 | 0/2 | -0.0036 | 2/2 | 1.0002 |

uniform 控制也必须非退化：RMSR、robust、temporal 的 median ΔCC 分别为 -0.00280、
-0.00348、-0.0753；NNWS 的成功运行近似线性，但只有 3/5 个种子成功。

### 3.1 原版 NNWS 的失败率

原版 NNWS 在 20 个场景×种子组合中失败 10 次：

- uniform：2/5；
- far75：4/5；
- near12：1/5；
- mirror168：3/5。

失败表现为非有限或全零权重，结果文件保留 `status=failed`，图中留空并写明没有替代波形。
`nnws_population_eps_diagnostic` 将权重标准差改为带 epsilon 的 population 口径后仍失败 4 次，
因此不能把原版失败简单归结为一个标准差梯度奇点，更不能用诊断版替换原版结果。

### 3.2 为什么 RMSR 的微小正差不能称为成功

far75/near12 的 RMSR ΔCC 只有约 3e-4/6e-4，跨种子不一致；其扇区/背景事件有效权重比约为
1，并没有压低加密方位。FTAN 图像相关也不一致：far75 的 group 图中位差为 -0.00236、phase
为 +0.00773；near12 的 group 为 +0.00055、phase 为 -0.00012。mirror168 和 uniform 还出现
CC 退化。它最多说明某些随机实现下 leave-one-out SNR 判据带来极小波形变化，不能说明恢复了
更真实的 EGF。

### 3.3 robust 与 temporal 的边界

robust 在部分场景明显优于自己的匹配置乱，说明它产生了非随机权重；但这些权重对 active EGF
往往是负作用。Li temporal adapter 在多数场景明显退化，且这是单台对归一化适配，不等价于
Li (2020) 用全台网 antisymmetry 和自相关能量约束的网络级实现。因此本结果既不能宣称 robust
有效，也不能用于否定 Li 的网络级方法；它只否定当前这两个双台实现作为本问题的可靠解。

## 4. FTAN 与源事件审计怎么读

正式 QC 同时输出：

1. 4–12 Hz、`-d/dt` NCF 校准后的双台波形；
2. period–group velocity FTAN，叠加 QEDisp 原生基阶 Rayleigh 群速度；
3. period–phase velocity FTAN，叠加 QEDisp 基阶 Rayleigh 相速度；
4. 逐周期归一化版，仅看脊线形态；
5. 每个方法一个全局最小二乘波形尺度、每行统一色标的 common-scale 版；one-bit 数据不解释物理
   振幅；
6. raw、RMSR retained/rejected、robust/temporal/NNWS effective source-event 分布和方位响应。

主图不画自动拾取，只画独立 QEDisp 理论曲线。连续权重方法用“effective contribution”表示，
不伪装成 retained/rejected；只有 RMSR 的 0/1 mask 才有字面意义上的 retained/rejected。

固定的第一盲测种子 5401 是预注册代表图，不是从五个种子中挑出的最好结果：

- `output_python/gate2_sparse_event_blind/figures/near12_seed5401_waveform_ftan_normalized.png`
- `output_python/gate2_sparse_event_blind/figures/near12_seed5401_ftan_common_scale.png`
- `output_python/gate2_sparse_event_blind/figures/near12_seed5401_source_event_audit.png`
- `output_python/gate2_sparse_event_blind/summary/gate2_sparse_event_blind_paired_metrics.png`

所有图同时导出 PNG、PDF、SVG 和 TIFF。完整逐种子指标见：

- `output_python/gate2_sparse_event_blind/summary/gate2_sparse_event_complete_evidence.csv`
- `output_python/gate2_sparse_event_blind/summary/gate2_sparse_event_paired_metrics.csv`
- `output_python/gate2_sparse_event_blind/summary/gate2_sparse_event_ftan_metrics.csv`
- `output_python/gate2_sparse_event_blind/summary/gate2_sparse_event_aggregate.csv`

## 5. Oracle 可达性与正确解释

`oracle_active_common_nnls` 直接读取 active EGF，拟合正、负支共用的非负窗口权重并约束
`N_eff >= 0.1M`。开发种子的有效窗约为 83–95，五个盲测种子均能把 waveform CC 推到约 1。
它证明“信息就在这 600 条短时 CCF 中，合适的时间权重可以重组出来”，从而排除了“任何时间
加权都不可能”的解释。

但该权重使用了答案本身，不能部署，也不能作为 NNWS 成功证据。相反，它把下一步问题缩小为：
缺少一个不接触 active truth、却能识别污染方向/传播物理的稳定观测量。

`oracle_zero_sector` 只保留完全没有扇区事件的约 94–127 个窗口，通常比 linear 更差。原因是它
丢失过多背景相干信息；这也说明“把所有含污染事件的窗都删掉”不是合理上限。

## 6. 是否进入真实资料

不进入“真实资料 efficacy 验证”。真实资料没有 exact EGF；若一个方法在合成真值下已失败或
50% 数值失效，再展示更平滑的实测频散图，只能证明图面变化，不能证明 EGF 更真实。

后续若只为工程故障审计，可以在真实资料上记录运行失败率、有效窗数、日期/站对稳定性和外部
active/MASW 一致性，但必须预先改计划并明确标为 feasibility/failure audit，不得包装为方法成功。

## 7. 下一条有证据价值的路线

当前结果不支持继续微调 NNWS epoch、RMSR 阈值、FTAN `df/dv` 或挑选种子。最有价值的改进是
增加独立的源方向观测约束：

1. 有阵列时，逐窗 beamforming/tau-p 估计方位与慢度，按每个站对 stationary-phase zone 做
   source-aware selection；
2. 有可靠 3C 时，用偏振/分量相干估计主方位，再做方位选择或连续权重；
3. 有多站对时，实施 Li (2020) 的网络级 temporal reweighting，而不是单站对 adapter；
4. 若仍限定为单站对，开发带稳定正则、严格训练/盲测分离的自监督或交叉验证权重，但不能再让
   active truth 进入损失；active NNLS 只能用来设定可达性上限。

这四条路线中，前三条引入了当前双台 CCF 缺失的物理信息，证据优先级高于继续调现有超参数。

## 8. 可复现入口

```bash
conda run -n seisloc python scripts/run_gate2_sparse_event_selection_experiment.py \
  --output-directory output_python/gate2_sparse_event_development \
  --wavefield-seeds 5301 --nnws-seeds 1000
conda run -n seisloc python scripts/run_gate2_sparse_event_selection_experiment.py \
  --output-directory output_python/gate2_sparse_event_blind \
  --wavefield-seeds 5401,5402,5403,5404,5405 \
  --nnws-seeds 1001,1002,1003,1004,1005
conda run -n seisloc python scripts/plot_gate2_sparse_event_qc.py \
  --run-directory output_python/gate2_sparse_event_blind --seed 5401
conda run -n seisloc python scripts/summarize_gate2_sparse_event_results.py \
  --run-directory output_python/gate2_sparse_event_blind
```

计划基线见 `docs/NCF_EGF_EQUIVALENCE_PLAN_CN.md`。本结果文档报告已经运行的证据，不修改
v5.0 冻结设计和判据。
