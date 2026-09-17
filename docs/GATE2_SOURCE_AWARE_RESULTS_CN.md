# Gate 2 v5.1 辅助阵列 source-aware 权重结果

## 1. 结论

v5.1 的 13 台辅助 Z 分量小阵列能够在保留原垂直力事件目录和 600 m 目标站对的前提下，可靠地
识别人为增加的噪声源方向簇；但是，把每个 25 s 时间窗作为整体、以其方位谱均匀化为目标的
`beam_uniformity` 权重，并没有提高目标双台 NCF 与 active EGF 的等价性。

这个结论不是由“图是否更亮”得到的：开发种子 5501 同时检查了无时移波形 CC、单位范数 L2、
wrapped phase RMS、pre-arrival 能量、匹配时间置乱、群速度 FTAN、相速度 FTAN 正确周期支，
并保留 uniform 场作为非退化控制。`beam_uniformity` 在三个偏置场和 uniform 场均未通过联合门，
因此冻结盲测种子 5601–5605 未运行，真实资料 efficacy 验证也未获授权。

更重要的是，使用事件坐标真值构造、但采用同一“整窗角向均匀化”目标的
`oracle_event_uniformity` 同样失败且通常退化更强。因此当前负结果应归因为：**共同时间窗的角向
均匀化目标不足**，而不是简单归因为辅助阵列方位估计不准。

## 2. 冻结实验

- 目标：`(-300, 0)`–`(300, 0)` m 双台 ZZ NCF；4–12 Hz；与 active EGF 和 QEDisp 基阶
  Rayleigh 群/相速度比较；
- 辅助观测：中心附近 13 台 irregular sunflower Z 阵列，仅用于每窗方位谱，不参与目标 NCF；
- beam：4 s Hann snapshot、50% overlap，4.5–11.5 Hz，5° 方位网格，160–440 m/s 速度网格；
- 权重：非负方位均匀化，固定正则项并强制 `N_eff >= 0.5M`；
- 场景：uniform、far75、near12、mirror168；开发种子 5501；
- 对照：linear、paired-clean counterfactual、20 次时间置乱、事件真值 oracle；
- FTAN 主图只叠加 QEDisp 理论群/相速度，不画自动拾取线。显示图为逐周期归一化版，只用于
  脊线/周期支评价，不作为绝对振幅或 SNR 证据。

## 3. 方位门

解析 plane-wave 单元测试覆盖 12°/75°/168°、200/280/360 m/s 和带噪条件并通过。DWM 开发场
的独立方位门如下：

| 场景 | Spearman（扇区功率–事件数） | 扇区存在 AUC | 平均 excess 峰值 | 峰值误差 | 结果 |
|---|---:|---:|---:|---:|---|
| far75 | 0.5703 | 0.8029 | 75° | 0° | 通过 |
| near12 | 0.5973 | 0.8295 | 10° | 2° | 通过 |
| mirror168 | 0.6270 | 0.8708 | 165° | 3° | 通过 |

这证明辅助阵列确实观测到了注入的方向簇，因而后续 EGF 失败不能用“beam 没看到异常源”解释。

## 4. 波形联合门

下表给出 symmetric branch；括号中为相对 linear 的差值。成功方向应为 CC 增、L2 降、phase
RMS 降且 pre-arrival 不增。

| 场景 | 方法 | CC（Δ） | L2 Δ | phase RMS Δ (rad) | pre-arrival Δ | 联合结果 |
|---|---|---:|---:|---:|---:|---|
| uniform | beam | 0.848360 (-0.001259) | +0.002290 | +0.001690 | +0.000050 | 失败 |
| far75 | beam | 0.810141 (-0.001447) | +0.002352 | -0.011605 | -0.008004 | 失败 |
| near12 | beam | 0.728935 (-0.009009) | +0.012339 | +0.062672 | +0.006799 | 失败 |
| mirror168 | beam | 0.790173 (-0.000727) | +0.001123 | +0.006554 | +0.002399 | 失败 |
| uniform | event oracle | 0.834604 (-0.015014) | +0.026726 | +0.020466 | -0.003545 | 失败 |
| far75 | event oracle | 0.772869 (-0.038719) | +0.060130 | +0.102189 | -0.017067 | 失败 |
| near12 | event oracle | 0.712614 (-0.025330) | +0.034181 | +0.228010 | +0.103423 | 失败 |
| mirror168 | event oracle | 0.759432 (-0.031467) | +0.046955 | +0.184721 | +0.025492 | 失败 |

`beam_uniformity` 只有 far75 在 matched-permutation 联合比较中通过；其余场景均未通过。正、负
支联合要求也未全部满足。不能把 far75 的 phase/pre-arrival 局部改善称为方法有效，因为同一场景
CC/L2 变差，且这种方向不在其他场景重复。

## 5. 群/相速度 FTAN 独立审计

FTAN 评价不使用自动 ridge pick，而使用四个预注册量：群速度图与 active 的图像相关、相速度图
与 active 的图像相关、QEDisp 群速度理论线能量、正确相速度周期支相对相邻 ±1 周支的优势。
下表为相对 linear 的差值；四项必须同时非劣/改善才通过。

| 场景 | 方法 | Δ group image CC | Δ phase image CC | Δ group theory energy | Δ correct-branch advantage | 结果 |
|---|---|---:|---:|---:|---:|---|
| uniform | beam | -0.001212 | -0.001541 | -0.000109 | +0.000004 | 失败 |
| far75 | beam | +0.013409 | +0.033137 | +0.010207 | -0.000419 | 失败 |
| near12 | beam | -0.024857 | +0.008119 | -0.000571 | -0.000130 | 失败 |
| mirror168 | beam | -0.008029 | +0.010648 | -0.000512 | -0.001449 | 失败 |
| uniform | event oracle | +0.011550 | -0.010876 | -0.002429 | -0.000129 | 失败 |
| far75 | event oracle | -0.061758 | +0.055139 | +0.033303 | -0.002639 | 失败 |
| near12 | event oracle | -0.091694 | -0.009711 | -0.008772 | +0.000156 | 失败 |
| mirror168 | event oracle | -0.156434 | +0.021172 | -0.005909 | -0.004809 | 失败 |

far75 的 beam 群/相图像相关和理论线能量看似变好，但正确相速度周期支优势下降，且波形 CC/L2
同时变差；这正是“图局部更好看但 EGF 没有更真”的例子。near12 和 mirror168 也只有部分 FTAN
量变好，不能通过联合门。

## 6. 物理解释与下一步

每个 25 s 窗平均混合约 17 个背景事件和约 2 个扇区事件。一个窗权重会同时改变该窗内所有方位、
传播路径和相位的贡献。即使知道事件真值，也无法只削弱异常扇区而保留同窗内有利的
stationary-phase 事件，所以“整窗源方位均匀化”不是足够精细的控制变量。

若继续，最有物理依据的下一候选不是调正则项或阈值，而是先做一个 **truth-only branch-specific
stationary-phase oracle**：正、负时延支分别按对应 endfire/stationary-phase 区事件构造权重，再
分别叠加后组合。只有这个 oracle 能改善 active-EGF 波形、正确相速度周期支和 precursor，才值得
用辅助阵列的方向谱替代真值；否则应停止这条 source-direction 时间筛选路线。该设计必须在运行
前升版并写入 canonical plan。

## 7. 结果文件

- `output_python/gate2_source_aware_development/gate2_source_aware_summary.json`：开发门状态；
- `output_python/gate2_source_aware_development/summary/gate2_source_aware_complete_evidence.csv`：
  波形与 FTAN 完整证据；
- `output_python/gate2_source_aware_development/summary/gate2_source_aware_evidence_summary.json`：
  联合门及失败分类；
- `output_python/gate2_source_aware_development/figures/`：每个场景的波形–FTAN 和源方向审计图，
  同时导出 SVG、PDF、PNG、TIFF。

