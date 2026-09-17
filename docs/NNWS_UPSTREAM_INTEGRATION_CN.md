# Fan 等（2026）NNWS v1.0 接入说明

## 来源与许可证

- 论文：Fan, Yu, Wang & Wei (2026), *JGR: Solid Earth*,
  DOI `10.1029/2026JB033827`。
- 软件归档：Zenodo `10.5281/zenodo.21534727`。
- 上游仓库：
  `https://github.com/hanyue0001/A-Neural-Network-Weighting-Stacking-method-for-cross-correlation-functions`。
- 接入提交：`04af38f042ec594c0a093178a16c1f72bd0e3380`，2026-07-24。
- 仓库内 `LICENSE` 为 MIT；Zenodo API 元数据当前显示 CC BY 4.0，与论文和仓库
  声称的 MIT 不一致。本项目按仓库内 MIT 文本保留版权和许可声明。

`python_src/nnws/` 是上游 v1.0 核心包的未修改快照，包括模型、损失、训练、指标和
加权叠加代码；未复制论文绘图代码和约 20 MB 的合成数据。许可证与引用文件随包保存。

## WaveDispX 适配边界

`wavedispx.nnws` 负责四件事：

1. 把 WaveDispX 保存的 positive/negative 分支还原为负时延到正时延的完整 CCF；
2. 调用上游 `nnws.train` 学习每个短时 CCF 的连续权重；
3. 对每个有身份的接收器对独立训练，并按其实际权重计算 `N_eff`；
4. 只在训练完成后按距离分箱，并进入现有频散与主动参考 EGF 评价。

当前主研究入口是严格的逐台站对应用：

```text
scope = pair_resolved
```

旧的 `train_nnws_window_strategy` 单距离道阵列代理仍保留用于消融，并固定标记
`array_proxy_from_one_distance_bin`；不得把它与逐对结果合并报告。主流程在每个独立噪声窗
内先计算不分箱的 scalar ZZ CCF，保留 `pair_i/pair_j/distance_m`，训练完成后才按距离平均。
当前仅支持竖直噪声源；3C 的 RR/TT 需要逐对旋转缓存后另行实现。

## 环境与调用

PyTorch 是可选依赖，默认线性栈和源侧筛选不受影响：

```bash
python -m pip install -e '.[nnws]'
```

逐对核心入口：

```python
from wavedispx.nnws import NNWSConfig, train_pairwise_nnws_strategy

result = train_pairwise_nnws_strategy(
    cached_pair_windows,
    dt_s=0.01,
    distance_bin_m=5.0,
    minimum_velocity_mps=150.0,
    maximum_velocity_mps=500.0,
    maximum_period_s=1.0 / frequency_min_hz,
    config=NNWSConfig(epochs=200, seed=0),
)
```

完整正演入口：

```bash
conda run -n seisloc wavedispx run examples/passive_nnws_pairwise_quick.toml
```

`[nnws]` 默认 `enabled=false`。启用时，最大时延自动扩展到至少
`aperture/vmin + 4/fmin`，以容纳论文 reported-SNR 的远端尾窗；若它超过独立噪声窗，运行会
明确报错。缓存按 float32 预估并受 `maximum_cache_mb` 硬门槛限制。

程序先用距离、速度范围、最大周期和实际 lag 轴判定台站对是否能完整容纳 pre-signal、signal、
post-signal 和 reported-SNR remote-tail 窗。几何不合格对单独记录 `eligible=false`，不计入训练
失败或覆盖率分母。合格对的训练成功比例必须达到 `minimum_pair_coverage`（默认 1.0），结果才
标记为 sufficient。失败对保留零权重和错误文本，不允许暗中回退到 baseline。

公平对照均限制在相同的 NNWS 成功台站对：

- `nnws_common_linear`：同一对集、全部窗单位权重；
- `nnws_random_neff_control`：每对独立置换 NNWS 权重与时间窗的对应，精确保留权重分布和
  `N_eff`；
- `nnws_common_oracle_*` / `nnws_common_random_*`：启用源侧筛选时，将其逐窗权重应用到同一
  台站对集合。

每个策略在 HDF5 中保存自己的 `distance_m/pair_count`，分析时不再错误复用全阵列距离轴。
HDF5 保存完整 `[window, pair]` 权重矩阵；CSV 保存逐对摘要。

主动参考 EGF 只能用于训练后的独立评价，不得进入 NNWS 损失或选择校准距离。

## 当前复现审计

在 `seisloc` 环境（Python 3.12.12、PyTorch 2.10.0）中：

- 上游测试：15 passed，另有 3 个 subtests passed；
- 上游 `test_uni` 数据、seed 0、CPU、200 epochs：约 3.34 s；
- 本次训练的 reported SNR 为 62.93，和仓库已保存权重所得波形的 CC 为 0.987；
- 仓库自带 `Figs/test0_analysis_figure.pdf` 标注的 linear/RMSR_SS/NNWS SNR
  分别为 3.74/5.79/98.04；
- 论文正文相应数字为 9.61/14.94/252.46。

因此，目前只能确认源码可运行、权重学习行为与保存结果高度相似，不能声称精确复现论文
SNR。后续比较必须统一数据版本、噪声窗定义、归一化和随机初始化，并以理论 EGF 的相位、
走时、波形与频散误差作为主要终点。

适配测试还发现：当输入 CCF 仅有幅度缩放、权重初始方差接近零时，上游
`WeightStdLoss = -torch.std(weights)` 可能产生未定义梯度并输出 NaN。适配层不会修改上游
公式，但会在权重进入累计器前显式拒绝非有限值。实际实验必须报告这种失败，不能回退到
全窗口线性栈。

此外，上游 `torch.std(weights)` 默认使用样本标准差，而论文式（5）写的是除以 `M` 的总体
标准差；这是代码与论文公式之间的一个小但真实的差异。对大量窗口影响通常有限，但在少窗
或低方差情况下可能放大数值不稳定。

为把“复现”和“数值稳定性实验”严格分开，`nnws.std_mode` 有三个显式值：

- `upstream_sample`：未经修改的上游 v1.0 样本标准差与损失；
- `population_eps`：按论文总体标准差并给 RMS/标准差平方根加 epsilon 的消融；
- `disabled`：关闭权重标准差项的消融。

后两者不是上游原算法，输出会写入 `std_mode`，论文比较必须分组报告。

## WaveDispX 首次正演冒烟结果

使用 `examples/passive_source_selection_anisotropic_quick.toml` 的同一批 60 个短时 NCF，
选择 150 m 的 ZZ 距离道训练；到时窗为 0.30–1.00 s，最大周期为 0.286 s。主动参考 EGF
只在训练完成后评价。

50 epochs、CPU、seed 0 的结果：

| 指标 | linear | NNWS array proxy |
|---|---:|---:|
| 训练窗 RMS 比 | 4.6893 | 4.7867 |
| 与主动 EGF 的无时移波形 CC | 0.81429 | 0.81406 |
| 相位 RMS（rad） | 0.56753 | 0.57036 |
| pre-arrival ghost fraction | 0.03563 | 0.03578 |
| `N_eff` | 60 | 59.994 |

权重基本退化为等权，EGF 指标没有改善。由于当前最大时延不足以容纳论文定义的远端 tail
noise window，`reported_snr` 不可用并明确返回 NaN。继续按上游默认训练到 200 epochs 时，
权重变成 NaN，适配层按设计中止。

这不是“NNWS 已被否定”，而是说明按距离分箱后再选单道学习全阵列权重的旧代理不成立，且
这一小型正演的窗间波形多样性不足。现已实现 pair-resolved 缓存、逐台站对训练、远端尾窗
所需时延和完整审计；NNWS 仍保持默认关闭，必须在更长、具有时变源分布的预注册合成实验上
比较 linear、随机同预算、Oracle 源筛选与 NNWS，主要终点为无最佳时移的 EGF 相位/走时/波形
一致性，SNR 只作次要指标。

## 逐台站对端到端 quick 验证

`examples/passive_nnws_pairwise_quick.toml` 使用 4 个接收器、6 个台站对、4 个独立 10 s 窗，
以 `population_eps` 消融训练 5 epochs。该算例只验证接线和审计，不足以评价方法效果：

- 3 个近距对因 pre-signal 窗不足被判为几何不合格；其余 3/3 训练成功，覆盖率 1.0；
- 权重范围 0.4821–0.4981，各合格对 `N_eff` 均约 4.0，仍接近等权；
- 在共同的 120/180 m 距离支持上，无时移 EGF CC：linear 0.066845，NNWS 0.066859；
- phase RMS：linear 1.866702 rad，NNWS 1.866664 rad；
- pre-arrival ghost fraction：linear 0.078161，NNWS 0.078174。

三项 EGF 变化都极小，不能据此宣称改善。共同支持只有 2 个距离道，三种对照的
`ambiguous_frequency_fraction` 都是 1.0 且没有有效自动拾取；频散结果不可解释。

## 60 窗时变各向异性源 benchmark

两个 benchmark 使用完全相同的模型、噪声种子、24 个源的预注册活动表、21 台接收器和
60×10 s 独立窗；210 个总台站对中，21 个 150–200 m 台站对满足完整 NNWS 时间窗，并形成
6 个共同距离道。训练采用 CPU、seed 0、上游默认 200 epochs。

`population_eps` 稳定性消融的 21/21 对全部成功；权重范围约
`1.6e-10–0.9340`，中位 `N_eff=24.18`。共同支持的 symmetric EGF 指标为：

| 策略 | 无时移 CC ↑ | phase RMS rad ↓ | ghost fraction ↓ | 谱形 log-RMS ↓ |
|---|---:|---:|---:|---:|
| `nnws_common_linear` | 0.679092 | 0.781216 | 0.037004 | 0.644352 |
| `nnws_random_neff_control` | 0.677126 | 0.846224 | 0.043951 | 0.631751 |
| `nnws_pairwise` | **0.797089** | **0.658485** | **0.006248** | **0.574192** |
| `nnws_common_oracle_hard_mask` | 0.485214 | 1.032122 | 0.055907 | 0.745697 |
| `nnws_common_random_neff_control` | 0.674379 | 0.780114 | 0.023664 | 0.639099 |

随机置换相同 NNWS 权重没有复现改善，表明在这个单一合成实现中，改善来自权重与具体时间窗的
对应，而不只是较小的 `N_eff`。这句话只适用于 6 个远距道上的 EGF 波形指标，不代表频散已经
恢复。正、负支也均改善，但负支仍明显弱于正支。

未经修改的 `upstream_sample` 精确损失在 200 epochs 时有 6/21 个合格对产生非有限权重，覆盖率
仅 0.714，低于预注册的 1.0 门槛，因此所有 exact-upstream 策略均标记 insufficient，不输出
频散/EGF 优劣数字。这复现了样本标准差损失在低方差权重处的数值风险；不能拿稳定性消融结果
冒充原算法结果。

上述波形改善没有通过整体有效性判据：共同支持只有 6 个远距道，理论 EGF、linear、随机对照和
NNWS 的 `ambiguous_frequency_fraction` 均为 1.0，因此这些频散图不可解释。程序另外输出显式
全孔径混合策略：对可训练台站对使用 NNWS/置换权重，对几何上不可训练的近距对保留单位线性
权重；它不改变任何上游训练权重，也不属于“失败对静默回退”。同一实现结果为：

| 全孔径策略 | 无时移 CC ↑ | phase RMS rad ↓ | ghost fraction ↓ | 谱形 log-RMS ↓ | 频散歧义率 ↓ |
|---|---:|---:|---:|---:|---:|
| `baseline_linear` | **0.791885** | 0.614742 | 0.032908 | **0.856386** | 0.8780 |
| `nnws_hybrid_random_neff_control` | 0.789412 | 0.627954 | 0.034525 | 0.856294 | 0.8211 |
| `nnws_hybrid_full_aperture` | 0.723082 | **0.588607** | **0.020100** | 0.948125 | 0.8211 |

混合 NNWS 的相位与到时前能量改善，但波形相关和谱形变差，仍为 **overall fail**。此外，将无噪声
主动 EGF 输入同一频散算子时，200 m 全孔径本身仍有约 80% 歧义频点；NCF 频散图相对 EGF 频散
图的相关系数虽由 linear 的 0.9833 小幅升至混合 NNWS 的 0.9855，但当前几何没有足够分辨率把
这种相对变化解释为可靠频散恢复。

`scripts/scan_egf_dispersion_aperture.py` 的无噪声 EGF 上限扫描显示，10 m 道间距下孔径从
100/200/300/400/600 m 增大时，歧义率约为 100%/80.5%/50.0%/37.0%/10.2%，中位半功率脊宽
约为 276/144/88/66/43 m/s。必须先用至少 400–600 m 的几何建立可分辨的频散基准，再做多噪声
种子统计；重复当前 200 m 失分辨率设置不能回答频散是否改善。

完整 QC 由 `scripts/plot_nnws_qc.py` 生成。`nnws_effectiveness_gate.json` 要求 EGF 分辨率门槛、
频散图相对 EGF 非劣，以及无时移波形相关、相位、pre-arrival 和谱形全部非劣；当前远距共同支持
与全孔径混合比较均为 `overall_pass=false`。目前只有一个噪声种子和一份源活动表；在新几何通过
分辨率门槛之前，不报告跨种子“有效率”。
