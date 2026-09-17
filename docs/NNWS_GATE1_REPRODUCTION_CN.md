# Fan（2026）/Xie（2020）Gate 1 常速度合成复现审计

## 1. 结论先行

在作者公开的四套 200 km、3 km/s 常速度合成 CCF 上，NNWS 的**主要机制结论可以复现**：

- 准均匀源场没有被破坏；
- 35–40°、60–65°和 85–90°三个持续定向源情景中，作者保存的 NNWS 权重均比 linear 和
  RMSR-SS 更接近理论 CCF；
- 最困难的 35–40°情景中，RMSR 仍保留大量含定向源的时间窗，而 NNWS 连续权重降低该扇区的
  有效贡献并增加台站轴向源贡献；
- 不读取作者保存权重、用公开网络重新训练时，四个情景各 5 个初始化种子全部成功，改善方向
  保持一致；
- 参考速度误差 ±10% 时，NNWS 仍优于 RMSR，但高估 10% 时波形 CC 明显下降，说明它不是
  不依赖先验的万能筛选器。

因此，当前证据支持：**在这套已知真值、与训练目标一致的合成数据上，NNWS 确实改变了有效源
照明并改善波形与群速度，不只是让图更平滑。** 它尚不能证明真实资料的 EGF 已恢复，也不能证明
绝对振幅、衰减或任意介质/频带下的 Green function 等价性。Gate 1 仍为“主要机制通过、作者消融
复现未完全闭合”。

## 2. 固定输入与范围

| 项目 | 固定值 |
|---|---|
| NNWS 上游提交 | `04af38f042ec594c0a093178a16c1f72bd0e3380` |
| StackMaster 参考提交 | `66909971f69df60cb25d9e8b732a102d939cd2aa` |
| 介质与站距 | 2-D homogeneous，3.0 km/s，200 km |
| CCF | 每情景 600 × 1001 点，1 s 采样，±500 s |
| 频带/FTAN | period 5–16 s，23 个周期，`alpha=2.5` |
| RMSR 主阈值 | 合成试验统一 `G=1+5/M` |
| NNWS | 作者保存权重；另以 epochs=200、Adam 0.001、0.8:0.2:0.01、CPU、seed 0–4 重训 |
| Selective | StackMaster 算法、linear reference、CC ≥ 0.2、[-85, 85] s |

本审计只回答作者尺度下的双台短时 CCF 选择机制。它不继承 WaveDispX Gate 2 的 600 m、
4–12 Hz、QEDisp 色散模型，也不评价子台阵 FJ/PS。

## 3. 作者保存权重的确定性复现

下表全部由同一公开 `.npy` 数据、公开权重和公开指标代码重算。`group MAE` 是 5–16 s 相对
3.0 km/s 的平均绝对百分比误差；`N_eff` 对连续权重按
`(sum(w)^2/sum(w^2))` 计算，RMSR 为实际保留窗数。

| 情景 | 方法 | SNR | 全窗 CC | 无时移相对 L2 | 对称损失 | precursor RMS 比 | group MAE (%) | N_eff/保留数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | linear | 3.74 | 0.507 | 0.862 | 0.689 | 0.348 | 0.334 | 600 |
| uniform | RMSR | 5.79 | 0.611 | 0.791 | 0.550 | 0.256 | 0.383 | 506 |
| uniform | NNWS | 98.04 | **0.904** | **0.427** | **0.085** | **0.031** | **0.257** | 193.3 |
| azi40 | linear | 5.04 | 0.382 | 0.924 | 0.637 | 0.347 | 1.003 | 600 |
| azi40 | RMSR | 7.51 | 0.462 | 0.887 | 0.607 | 0.235 | 1.010 | 552 |
| azi40 | NNWS | 120.87 | **0.849** | **0.529** | **0.281** | **0.068** | **0.304** | 193.1 |
| azi65 | linear | 3.47 | 0.422 | 0.907 | 0.645 | 0.585 | 1.005 | 600 |
| azi65 | RMSR | 6.09 | 0.488 | 0.873 | 0.482 | 0.304 | 0.876 | 506 |
| azi65 | NNWS | 99.43 | **0.805** | **0.594** | **0.287** | **0.114** | **0.296** | 206.4 |
| azi90 | linear | 3.38 | 0.474 | 0.881 | 0.765 | 0.610 | 1.480 | 600 |
| azi90 | RMSR | 9.44 | 0.570 | 0.822 | 0.587 | 0.354 | 1.235 | 509 |
| azi90 | NNWS | 119.67 | **0.765** | **0.644** | **0.326** | **0.118** | **0.342** | 301.7 |

SNR 与 symmetry 是训练目标或近邻量，不能作为独立成功证据。更有说服力的是：未参与训练的
理论 CCF 全窗 CC、无时移 L2、群速度误差和事后源场机制审计同时改善，且不是只改善一支。

Selective（CC ≥ 0.2）在 uniform 和 azi90 的波形 CC 可达 0.758 和 0.722，但 uniform 的 group
MAE 为 0.944%，比 linear 和 NNWS 更差。这正好说明“波形看起来更相干”与“所有频散指标都
更真”不是同一件事。

## 4. retained/rejected source 审计

源位置和激活时刻只在方法完成以后用于解释结果，没有进入 RMSR 或 NNWS。按作者机制代码的
严格 `1001 * segment` 边界：

| 情景 | 方法 | 台站轴向 ±10°有效源比例 | 附加扇区有效源比例 |
|---|---|---:|---:|
| uniform | raw | 0.115 | — |
| uniform | RMSR retained | 0.136 | — |
| uniform | NNWS weighted | **0.262** | — |
| azi40 | raw | 0.104 | 0.099 |
| azi40 | RMSR retained | 0.111 | 0.095 |
| azi40 | NNWS weighted | **0.257** | **0.055** |
| azi65 | raw | 0.104 | 0.096 |
| azi65 | RMSR retained | 0.125 | 0.068 |
| azi65 | NNWS weighted | **0.245** | 0.078 |
| azi90 | raw | 0.104 | 0.097 |
| azi90 | RMSR retained | 0.123 | 0.082 |
| azi90 | NNWS weighted | **0.205** | 0.071 |

azi40 是关键反例：RMSR 仅把附加扇区比例从 0.099 降到 0.095，说明 SNR-only 筛选基本没有
识别出靠近真实到时的污染；NNWS 将其降到 0.055，同时把轴向贡献增至 0.257。波形伪峰、群速
误差和源场变化因而形成了相互一致的证据链。

NNWS 没有 hard rejection 规则，因此图中只把 RMSR 分成 retained/rejected；NNWS 只展示连续
effective-source contribution。任意按权重分位数制造“NNWS rejected sources”都会改变原算法，
本审计没有这样做。

## 5. 网络初始化与参考速度敏感性

公开 NNWS 代码以 seeds 0–4 在四个情景各训练 200 epochs，20/20 次均产生有限权重：

| 情景 | 全窗 CC 中位数（范围） | group MAE 中位数 (%) | N_eff 中位数 |
|---|---:|---:|---:|
| uniform | 0.863 (0.854–0.890) | 0.324 | 193.3 |
| azi40 | 0.807 (0.777–0.845) | 0.237 | 202.3 |
| azi65 | 0.833 (0.819–0.857) | 0.699 | 216.1 |
| azi90 | 0.831 (0.811–0.845) | 0.272 | 218.4 |

这说明结论不是作者某一个保存初始化的偶然结果，但不同种子对 SNR 和 group MAE 的离散不可
忽略，正式比较必须报告分布而不是只选最佳网络。

在最困难的 azi40 上，将参考速度乘以 0.9/1.0/1.1，并相应反比缩放训练到时窗：

| 方法 | 速度比例 | 全窗 CC | SNR | 对称损失 | group MAE (%) |
|---|---:|---:|---:|---:|---:|
| RMSR | 0.9 | 0.522 | 5.08 | 0.559 | 0.793 |
| RMSR | 1.0 | 0.462 | 7.51 | 0.607 | 1.010 |
| RMSR | 1.1 | 0.417 | 5.72 | 0.547 | 1.026 |
| NNWS（5 seeds 中位） | 0.9 | 0.824 | 48.49 | 0.060 | 0.654 |
| NNWS（5 seeds 中位） | 1.0 | 0.807 | 56.03 | 0.055 | 0.237 |
| NNWS（5 seeds 中位） | 1.1 | 0.708 | 45.78 | 0.053 | 0.708 |

NNWS 对 ±10% 误差仍优于 RMSR，但正偏差 10% 时 CC 明显降低；对称损失仍很好，却不能阻止
群速误差和波形误差上升。这再次说明 symmetry 不能替代外部真值。

### 5.1 DOWS 的透明重建，不冒充精确复现

SI 给出了 DOWS 的参数化、损失、Adam 0.001 和 200 epochs，但没有发布实现或 raw-weight
初始化。本项目按文字定义重建直接权重优化，并明确采用 `raw ~ Normal(0, sigma)`。在
`sigma=1`、5 个种子下：

| 情景 | DOWS 全窗 CC 中位数（范围） | group MAE 中位数 (%) | N_eff 中位数 |
|---|---:|---:|---:|
| uniform | 0.550 (0.526–0.562) | 0.360 | 510.3 |
| azi40 | 0.407 (0.388–0.451) | 0.903 | 512.9 |
| azi65 | 0.451 (0.438–0.488) | 0.784 | 513.4 |
| azi90 | 0.527 (0.508–0.530) | 1.219 | 511.3 |

azi40 的初始化尺度从 0.1/1/3 增大时，CC 中位数由 0.442 降为 0.407/0.363，`N_eff` 由
593.3 降为 512.9/392.8。这个重建版没有复现 NNWS 的改善，而且对未公开初始化敏感；它只能
证明“按 SI 文字仍不足以唯一重建 DOWS”，不能用来断言论文 DOWS 无效。

## 6. 论文—数据—代码差异

1. 论文 uniform 的 linear/RMSR/NNWS SNR 为 9.61/14.94/252.46，公开仓库同一绘图入口和数据
   重算为 3.74/5.79/98.04；CC 则由论文 0.48/0.58/0.87 变为 0.507/0.611/0.904。
   因此可以复现公开仓库，不可声称逐数值复现正文。
2. SI 规定所有合成 RMSR 使用 `G=1+5/M`；公开 `test_uni` 代码却只在频散面板改用
   `G=1+1/M`。本审计主结果统一使用 `5/M`，`1/M` 只保留为 legacy sensitivity。
3. 论文写 1000 s segment；机制绘图代码用严格的 `1001*segment` 激活边界。两种映射约有
   3,100–3,600 个源被分配到不同 CCF。改用 1000 s 后，NNWS 提高轴向贡献、降低 azi40 扇区
   的方向不变，但绝对比例发生变化，例如 azi40 轴向比例由 0.257 降到 0.230。
4. uniform 理论波形在作者代码中使用 `wavelet='pi'`，三个定向情景使用 `sigma2`。本审计按
   作者各图原样复现并显式记录，不能把跨情景 CC 的细小差异全部解释为源方位效应。
5. DOWS 的初始化细节和实现源码未随公开仓库提供。本项目已把 SI 文字重建和初始化尺度扫描
   单独标为 reconstruction；其数值不得冒充 Figure S8 的精确复现。
6. tf-PWS 已使用固定 StackMaster 提交 `66909971...` 的原始 `tfpws(..., p=2)` 和固定
   `stockwell==1.2` 运行；没有使用 scipy STFT 或普通 PWS 替代。四情景全波形 CC 为
   0.500/0.375/0.422/0.468，与 linear 的 0.507/0.382/0.422/0.474 基本相当，显著低于保存
   NNWS 权重的 0.904/0.849/0.805/0.765。它不能提供 retained/rejected source 审计，因此只
   是非线性叠加对照，不是源筛选机制证据。
7. 公开数据只有预计算 CCF 和源表，没有生成这些 CCF 的连续合成记录，因此无法忠实重做
   Figure S18 的 overlap/resegmentation 敏感性。

## 7. 对“方法是否可靠”的当前评价

### 合成数据

**有条件可靠。** 在公开数据、公开损失和作者尺度内，结果同时通过波形、双支、precursor、群速
和源场机制审计，多初始化也可重复。这比只看 SNR 或漂亮频散图可靠得多。

但它仍是自监督目标在同分布合成上的验证。网络以 SNR 和 symmetry 为目标，可能把真实但不对称
的介质响应、方向性散射或振幅信息当成污染。当前 one-bit/whitened 合成只支持归一化波形与频散
结论，不支持绝对振幅或 Q。

### 实际数据

**尚不能据此判定真实 EGF 被恢复。** 实测没有 exact GF，短期资料与本方法自身长期栈更相关也
存在循环。后续必须冻结参数后做未见日期、未见站对、多方位验证，并与长期 linear/robust、
独立 active/MASW/已有频散和正负支分别比较。若只有 SNR、对称性和显示改善，结论只能写成
“提高短时 NCF 的可见性与稳定性”。

## 8. 复现命令与输出

```bash
conda run -n seisloc python scripts/run_nnws_gate1_upstream_reproduction.py
conda run -n seisloc python scripts/run_nnws_gate1_training_sensitivity.py
conda run -n seisloc python scripts/run_nnws_gate1_dows_reconstruction.py
conda run -n seisloc python scripts/run_nnws_gate1_tfpws.py
```

主要输出：

- `output_python/nnws_gate1_upstream_reproduction/gate1_source_selection_audit.png`
- `output_python/nnws_gate1_upstream_reproduction/gate1_waveform_group_velocity_audit.png`
- `output_python/nnws_gate1_upstream_reproduction/gate1_window_weight_audit.png`
- `output_python/nnws_gate1_upstream_reproduction/gate1_metrics.csv`
- `output_python/nnws_gate1_upstream_reproduction/gate1_source_audit.csv`
- `output_python/nnws_gate1_upstream_reproduction/gate1_source_mapping_sensitivity.csv`
- `output_python/nnws_gate1_training_sensitivity/gate1_nnws_initialization_sensitivity.png`
- `output_python/nnws_gate1_training_sensitivity/gate1_reference_velocity_sensitivity.png`
- `output_python/nnws_gate1_training_sensitivity/gate1_training_sensitivity_metrics.csv`
- `output_python/nnws_gate1_dows_reconstruction/gate1_dows_reconstruction_metrics.png`
- `output_python/nnws_gate1_dows_reconstruction/gate1_dows_initialization_scale_sensitivity.png`
- `output_python/nnws_gate1_tfpws/gate1_stackmaster_tfpws_audit.png`
- `output_python/nnws_gate1_tfpws/gate1_tfpws_metrics.csv`

当前 Gate 1 判定：**主要机制检查通过，P2 正式关闭。DOWS 已以不确定重建方式处置，tf-PWS
已按固定上游实现完成且没有表现出 NNWS 式 EGF 改善。唯一未能执行的 overlap/resegmentation
敏感性是因为上游没有发布连续合成记录；这被登记为数据可用性限制，而不是用推测结果补齐。**
