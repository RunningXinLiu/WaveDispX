# 双台 NCF 时间窗筛选与 EGF 等价性主推进计划

> **文档地位：唯一执行基线（canonical plan）**
>
> **当前版本：v5.2**
>
> **设计冻结日期：2026-08-15**
>
> **当前主线：双台短时 CCF 的选择/加权；子台阵 FJ/PS 暂停**

## 0. 计划治理

本文件是噪声源分布、短时 CCF 筛选、NCF 叠加和 NCF–EGF 等价性验证的唯一主推进计划。
其他 README、实验说明、结果审计和旧验证协议只能补充实现细节，不能覆盖本文件的阶段顺序、
科学问题、成功判据或禁止事项。

后续若更改以下任一内容，必须在改代码或运行正式实验之前（最迟与代码同一次变更）更新本文件：

- 研究问题或允许的结论强度；
- 主方法、对照组或 Oracle 定义；
- 介质模型、源场、观测系统、分量或候选频带；
- 训练/开发/盲测划分、随机种子或有效样本量约束；
- FTAN、波形、频谱或源场评价指标；
- Gate 的阶段顺序、进入条件或成功/失败判据；
- 双台与子台阵方法的范围边界。

每次变更必须同步更新文末“计划变更记录”，写明日期、原决定、新决定、证据或原因、受影响的
配置/代码/结果。禁止只在聊天、代码注释或结果目录中改变计划。

## 1. 科学问题与允许的结论

### 1.1 主研究问题

在噪声源方位分布随时间变化且存在持续定向源时，能否仅依据一个站对的短时 CCF，选择或加权
时间段，使最终双台 NCF 比全窗口线性叠加和公平随机对照更接近同一介质中的独立主动源
Green function？

### 1.2 必须分开的两个结论层级

1. **归一化波形/相位/频散恢复**：允许使用 one-bit 和谱白化，评价走时、相位、波形形态与
   频散；不能解释绝对振幅和衰减。
2. **振幅保持的 EGF 恢复**：使用 raw CCF 或仅线性预处理，评价相对振幅、距离衰减、相位和
   完整波形；不能用 phase-only/PWS 的高 SNR 代替振幅保真。

最终报告必须明确结果属于哪一层。除非振幅保持轨道也通过独立真值验证，否则不得宣称恢复了
“完整 EGF”或绝对振幅 Green function。

### 1.3 不充分证据

下列现象单独出现时，只能说明可见性、稳定性或特定任务改善，不能证明 NCF 更接近 EGF：

- SNR 增加；
- 正负支更对称；
- 频散能量图更亮或更平滑；
- 自动拾取更接近理论曲线；
- 更多周期通过 SNR 门槛；
- 短期结果与同一种方法的长期结果更相关；
- 单个随机种子、单个站距或单一源方位取得改善。

## 2. 文献证据与本计划的继承关系

### 2.1 Xie et al. (2020)：RMSR-SS

- 观测单元是一个站对的短时 CCF，而不是子台阵 gather。
- 用参考群速度定义 signal、precursor 和 tail 窗。
- leave-one-out 比较删除某条短时 CCF 前后的 signal/noise RMS ratio。
- 合成试验绘制 retained/rejected sources；实测用约 15 d 结果与长期线性栈相速度比较。
- 主要支持 SNR、收敛和相速度稳定性；不能单独保证相位、振幅或双支保真。

来源：<https://doi.org/10.1093/gji/ggaa232>

### 2.2 Weaver–Yoritomo 与 Li et al. (2020)：temporal reweighting

- 对时间块 CCF 求非负权重，最小化 antisymmetry 或主到时前 acausality 相对信号能量。
- 评价因果性、正负支对称性、方位振幅均匀性和相对振幅稳健性。
- 证据强于仅优化 SNR，但只能使有效源场“更接近各向同性”，不能凭空补齐长期缺失方位。

来源：<https://doi.org/10.1121/1.5043406>；<https://doi.org/10.1093/gji/ggaa001>

### 2.3 Fan et al. (2026)：NNWS

- 输入是一个站对的一组短时 CCF，输出每条 CCF 的 0–1 连续权重。
- 损失联合 inverse RMS ratio、causal/anticausal antisymmetry 和弱权重分散正则项。
- 合成包含均匀源以及 35–40°、60–65°、85–90°持续定向源。
- 当污染到时靠近真实面波时，SNR-only RMSR 可能失败；对称性提供额外判据。
- 合成用 theoretical CCF、全窗 waveform CC、SNR、群速度误差和加权源 KDE；实测没有 exact GF。

来源：<https://doi.org/10.1029/2026JB033827>

### 2.4 Yang et al. (2023)：stacking 基准

- linear、robust、selective、cluster、PWS、tf-PWS、Nth-root 和 covariance filter 对相位与振幅
  的保存能力不同。
- 高 SNR 或清楚首波不意味着适合振幅、衰减或时间变化研究。
- raw NCF 的 robust stack 是必须纳入的通用基线。

来源：<https://doi.org/10.1093/gji/ggac410>

### 2.5 Stationary-phase 与阵列方法的范围边界

Snieder 的 stationary-phase 理论解释哪些源方向可相长贡献双台 Green function；它不是自动
选择器。Ning et al. 的 beamforming/in-line source selection 面向子台阵 virtual shot gather，
不是 Xie/Fan 的 pair-local CCF weighting。

来源：<https://doi.org/10.1103/PhysRevE.69.046610>；
<https://doi.org/10.1007/s10712-022-09723-2>

由此冻结的主线是：**先完成双台 CCF 选择/加权的证据闭环，再单独研究子台阵 FJ/PS。**

## 3. 全局执行顺序

```text
Gate 0  FTAN 与相位算子已知真值验证
   ↓
Gate 1  原样复现 Xie/Fan 双台选择机制
   ↓
Gate 2  单一 1-D 色散模型 + active EGF + 受控源方位试验
   ↓
Gate 3  多方法、多随机种子和预处理轨道比较
   ↓
Gate 2-S  辅助小阵列 source-aware 权重；目标仍为同一双台 CCF
   ↓
Gate 4  实际双台资料时间外/站对外验证
   ↓
Gate 5  独立的子台阵 FJ/PS 研究（当前暂停）
```

上一个 Gate 没有通过时，不得用下游更复杂场景掩盖问题，也不得把下游图件作为当前方法有效的
证据。

## 4. Gate 0：FTAN 与相位算子真值验证

### 4.1 目的

在任何筛选比较之前，排除群速度包络、相位符号、距离/时间单位、Hilbert 变换、远场相移、
周期顺序和整数周跳错误。FTAN 是测量仪器；仪器未合格时不评价筛选器。

### 4.2 G0-A：常速度解析 Green function

- 构造已知 (c=U) 的带限解析或半解析 Green function。
- 群速度必须由窄带解析信号包络到时 (t_g) 得到：(U=r/t_g)。
- 相速度必须来自传播相位；明确处理 (2\pi) 周期分支和所用二维/三维远场相位常数。
- 正、负时延支分开测试，不先对称化。
- 至少测试正负极性、CF→EGF 校准、不同距离和不同 Gaussian 带宽。

验收值由离散速度网格和滤波分辨率决定并在运行前写入测试；原则上中位误差不得超过一个速度
网格，至少 90% 的有效周期不得超过两个网格。

### 4.3 G0-B：单一 1-D 模型 active EGF

- 只使用一个简单、正常频散的层状模型；不比较低速夹层或复杂多模型。
- QEDisp 基阶 Rayleigh (c(T)) 由 `forward` 计算，(U(T)) 由其原生 `sregn96` 能量积分直接
  计算；相速度微分只保留为交叉核对，不再作为正式群速度真值。
- active-source 正演产生独立 Green function，不使用全窗口 NCF 代替真值。
- active 波形、群速度 (T-U) 能量谱和相速度 (T-c) 能量谱必须同时输出。
- 主图仅叠加 QEDisp 理论曲线，不画自动拾取；拾取只允许作为通过单元测试后的补充数值审计。

### 4.4 公共有效频带

2–12 Hz 是候选目标，不是预先保证的有效范围。公共有效频带必须由以下条件一次性冻结：

- active EGF 在该频点有足够频谱能量；
- 群速度包络在物理到时窗内可分辨；
- 相速度正确周期支可由独立相位约束确定；
- 数值采样、时间窗和边界反射合格；
- 若进入阵列阶段，还必须满足空间 Nyquist 和孔径分辨率。

冻结后的频带掩膜对所有方法相同，禁止按方法选择更好看的频段。

**Gate 0 冻结结果（2026-08-15）：** 2–12 Hz 相速度通过，但 600 m 双台群速度包络在 2–4 Hz
无法解析当前模型的急剧群速度曲率。正式 waveform/group/phase 公共频带冻结为 **4–12 Hz**，
速度网格为 2 m/s，Gaussian `alpha=120`。2–4 Hz 只保留在诊断图中并标为不合格候选区；不得
用于方法比较。上述参数在 Gate 1–3 不按方法重调。

### 4.5 参考实现

实现需同时对照已提供的 `EGFAnalysisTimeFreq.m` 定义和发表的 FTAN 流程；任何与参考实现的
差异必须写入 `docs/PAIR_FTAN_VALIDATION_CN.md`。该文档已记录重新验收、QEDisp 原生群速度、
2–4 Hz 失败原因和 4–12 Hz 公共频带；旧自动拾取误差不再作为验收依据。

## 5. Gate 1：Xie/Fan 机制复现

### 5.1 目的

先在作者使用的简单常速度条件下隔离验证选择/加权实现，避免把 NNWS/RMSR 实现问题与近地表
频散、数值正演或 FTAN 问题混在一起。

### 5.2 复现场景

按 Fan 论文建立：

1. 准均匀多源基准；
2. 35–40°持续定向源；
3. 60–65°持续定向源；
4. 85–90°持续定向源。

Gate 1 使用作者公开的原始常速度合成数据与已保存 NNWS 权重，固定为：二维均匀介质
3.0 km/s、站距 200 km、600 条 1001 点短时 CCF、采样间隔 1 s、总模拟时长 600,000 s、
5–16 s period band、FTAN `alpha=2.5`。均匀情景含 10,000 个环带源；三个偏置情景各含
11,000 个源，其中额外 1,000 个源位于对应方位扇区。该论文尺度只用于机制复现，**不继承**
Gate 0/2 的 600 m、4–12 Hz、`alpha=120` 近地表设置。

源数、站距、记录长度、period band、segment length、预处理、损失系数和随机初始化应尽可能
忠实于论文；无法完全一致的项目必须在结果中逐项列明。作者公开代码用激活时刻的
`1001 * segment` 边界把源映射到 CCF，而论文文字写 1000 s segment；主复现按代码的 1001
边界执行，并把 1000 s 边界作为源图敏感性审计。

### 5.3 复现方法

- `linear_stack`；
- Xie `rmsr_ss`；
- Fan `nnws_upstream`；
- Fan 补充材料的 `dows`、`selective`、`tf_pws` 作为复现消融。

RMSR 主结果统一使用 Fan 补充材料为合成试验规定的
`G = 1 + 5/M`。作者 `test_uni` 绘图代码的波形面板使用 `5/M`，但频散面板改用 `1/M`；
该不一致不得混入主比较，只作为明确标注的 legacy sensitivity 输出。

### 5.4 复现证据

- theoretical CCF 与各方法最终波形；
- 全时窗 waveform CC、SNR、antisymmetry 和 precursor energy；
- 群速度误差；
- 原始源、有效加权源 KDE、retained/rejected source map；
- 参考群速度 ±10%、segment length、overlap 和随机初始化敏感性。

必须观察到：均匀源不被明显破坏；远离 stationary-phase zone 的污染可被 RMSR/NNWS 抑制；
靠近真实面波到时的污染能暴露 SNR-only 与 symmetry-aware 方法的差异。否则 Gate 1 失败。

## 6. Gate 2：单一 1-D 色散模型主合成试验

### 6.1 固定项

- 一个经 Gate 0 合格的简单 1-D 模型；
- ZZ 分量和基阶 Rayleigh 波；
- 固定 600 m 主站对、源时函数统计、时间段数、窗口长度、4–12 Hz 公共有效频带和
  FTAN `alpha=120`；
- 独立 active Green function 与 QEDisp 理论曲线；
- 源数由 ideal-uniform NCF 的收敛测试确定，不按图面密度任意选择。

Love 波、RR/TT、复杂多模态和多速度模型均不进入主合成试验。

### 6.2 四类源场

1. **准均匀多源控制**：检验方法不会把合理 NCF 优化坏。
2. **远离 stationary-phase zone 的间歇性扇区增强**：SNR 型方法的容易情景。
3. **靠近 stationary-phase zone 的间歇性扇区增强**：污染与真实波列接近的困难情景。
4. **第 3 类的镜像方位**：将污染从一支换到另一支，排除方法固有支偏好。

### 6.3 每类偏置源场的两个版本

- **paper-faithful/additional-power**：在均匀背景上额外增加扇区源，复现 Xie/Fan 的能量增强。
- **constant-total-power**：每个时间段总注入源功率完全相同，只改变方位密度。

第二个版本用于排除“方法只是删除高能时段”的替代解释。

### 6.4 Oracle 的使用边界

真实源位置、源活动和 stationary-phase 标签只用于合成后的机制审计与上限对照。它们不得进入
RMSR、temporal reweighting、NNWS 的训练、权重、阈值或参考到时。

### 6.5 Gate 2 源数收敛与几何预注册

在查看任何 Gate 2 方法优劣前，先只对准均匀控制场运行源数收敛，固定如下：

- 双台位于 `(-300, 0)` 与 `(300, 0)` m；源位于以台对中点为圆心、1,500–1,800 m 的远场环带；
- 采用 Gate 0 的 20 m/半空间简单 1-D 模型、垂直力源、ZZ 和基阶 Rayleigh；
- `dt=0.01 s`、每窗 25 s、120 个互不重叠独立窗；源谱为 3.5–12.5 Hz Tukey 平顶带，正式
  评价仍只用 4–12 Hz；
- 候选均匀背景源数固定为 72/144/288/576/1152，每个源数使用 3 个预先固定的独立噪声种子；
- 选择相对 1152 源参考，active-EGF waveform CC 中位差不超过 0.02、wrapped phase RMS 中位差
  不超过 0.05 rad，且同一条件在相邻更高源数不反转的最小源数；不满足则选 1152，并明确报告
  没有观察到平台；
- 收敛判据只在 `spectral_whitening` phase/dispersion 轨道运行，不能据此解释绝对振幅。

源数冻结后，三个间歇扇区场共享同一批准均匀背景坐标和同一 120 窗活动序列。扇区源数固定为
背景源数的 25%，扇区宽 12°；远离 SPZ 场中心为 75°，靠近 SPZ 场为 12°，镜像场为 168°。
扇区幅度状态固定为 0/0.75/1.5/2.5，各出现 30 窗后由固定 schedule seed 置乱。additional-power
保留背景振幅为 1；constant-total-power 在每个窗按源振幅平方和缩放全部源，使总注入功率等于
准均匀背景。两个版本必须共用噪声随机数、坐标和活动次序，形成配对反事实。

### 6.6 v5.0：文献一致的稀疏瞬态 episode 主试验

v4.8–v4.9 的开发场让所有 576 个背景源在每个 25 s 窗内同时产生连续平稳随机噪声。paired-clean
证明方位偏置确实降低 active-EGF 一致性，但 RMSR、NNWS 和 temporal 均无法稳定识别应降权窗。
重新核对 Xie (2020) 与 Fan (2026) 后确认，这两篇论文的合成证据是约 10,000 个瞬态 Ricker
源按激发时刻分散到 500/600 个 segment，每段只含约 17–20 个背景事件；Fan 的方向簇约为背景
事件数的 10%。因此 v4.9 被保留为“持续混合源场负控制/适用边界”，不再作为 Xie/Fan 时间筛选
主验证。

新的 Gate 2 主试验在运行前冻结为：

- 600 个互不重叠的 25 s 窗，共 15,000 s；600 取自 Fan 合成的 segment 数，25 s 在当前
  4–12 Hz 缩尺模型内包含 100–300 个周期；
- 576 个均匀背景候选位置仍使用已通过空间离散审计的 1,500–1,800 m 环带；三个偏置场各增加
  58 个位于 12° 宽扇区的候选位置，中心仍为 75°、12° 和 168°；
- 使用项目已有的 `poisson_ricker` 源模型，所有候选位置共享
  `event_rate_hz = 10000 / (576 × 600 × 25)`。因此背景期望事件数为 10,000，58 个扇区位置
  期望约 1,007 个事件；每次事件随机正负号、振幅均匀分布于 0.5–1.0；
- Ricker 主频 8 Hz，生成带 3.5–12.5 Hz，正式评价仍为 4–12 Hz；垂直力、ZZ、600 m 站对和
  Gate 0 的 1-D 模型不变；
- 每窗先对两台记录分别 one-bit，再 spectral whitening 后互相关，忠实对应 Xie/Fan 的相位/
  频散预处理轨道；该轨道不解释绝对振幅；
- 方向簇不再使用人为 0/0.75/1.5/2.5 全源幅度状态，时间变化完全来自离散事件出现与否、位置、
  激发时刻、极性和振幅。additional/constant-power 重复场从主试验移除；
- 保存每个事件的 `window_index/source_index/time/weight/source_group/azimuth`，方法窗权重回填到
  每个事件后绘制 raw/effective/retained/rejected 源图；
- 每个偏置场以同一 634 源几何、同一随机数流重新合成仅将 58 个扇区源活动置零的 600 窗
  `paired_clean_counterfactual`。这保证背景事件逐个配对；
- 统一源场和每个 paired-clean 的 symmetric linear 必须达到 active-EGF CC ≥ 0.65 且 wrapped
  phase RMS ≤ 1.2 rad，否则先判定事件合成基准不合格，不评价选择器；
- 除 v4.9 已冻结的 linear/RMSR/robust/temporal/原版 NNWS/NNWS 数值诊断外，新增
  `oracle_active_common_nnls` 可达性诊断：用 active EGF 真值拟合一组正、负支共用的非负窗权重，
  并强制 (N_{eff} ≥ 0.1M)。它只回答“当前短 CCF 集合能否通过时间加权接近真值”，绝不是可
  部署方法；
- 开发波场/网络种子为 5301/1000；参数冻结后盲测波场种子为 5401–5405，网络种子为
  1001–1005。每个非线性/筛选方法仍保留 20 次匹配权重置乱对照。

### 6.7 v5.1：辅助二维小阵列的 source-aware 双台权重

v5.0 已证明 pair-local RMSR、robust、temporal adapter 与原版 NNWS 不能稳定识别方向簇，而
active-common NNLS 又证明同一组短 CCF 存在可达的非负组合。v5.1 不再调这些方法的超参数，
而是检验缺失的信息是否就是独立源方位观测。

范围冻结如下：

- 研究终点仍是 `(-300,0)`–`(300,0)` m 目标双台的 ZZ NCF、active EGF 和 4–12 Hz 双台
  group/phase FTAN；辅助阵列不参与 NCF 叠加，也不做 FJ/PS；
- 在台对中点放置 13 个确定性的 irregular sunflower 小阵列台：半径由
  `45*sqrt(k/12)` m、方位由 `k*137.507764°` 定义，`k=0..12`。连同目标双台共 15 个 Z 台；
  方位估计只使用 13 个辅助台，避免 600 m 长基线产生空间混叠；
- 噪声源、介质、垂直力、Poisson-Ricker 事件、600×25 s、one-bit→whitening 目标 CCF 与
  far75/near12/mirror168 场保持 v5.0 定义。增加接收器不得改变源随机数流；每个种子的事件目录
  必须与同种子双台实现逐事件一致；
- beamforming 使用每个 25 s 窗的原始 Z 波形，4 s Hann snapshot、50% overlap；频率固定为
  4.5:1:11.5 Hz，backazimuth 网格 0:5:355°，相速度搜索为 160:40:440 m/s。每个 snapshot/
  frequency 先做 unit-phase normalization，再形成 phase-only conventional beam power；速度维取
  最大值，snapshot/frequency 求均值，得到每窗归一化 72 方位功率。算法不知道源坐标、事件标签、
  QEDisp 精确曲线或 active EGF；
- `beam_uniformity` 权重通过非负优化，使 600 窗加权后的 72 方位功率接近固定均匀目标。目标函数
  冻结为 `72*sum((P^T w/sum(w)-1/72)^2) + 0.05*mean((w-1)^2)`，随后只允许与单位权重凸混合以
  强制 `N_eff >= 0.5M`；不设置按结果调节的 hard threshold；
- `oracle_event_uniformity` 用事件真值的 5° 方位直方图代替 beam power，使用完全相同的优化与
  `N_eff` 约束，只作“方位均匀化本身是否可达”的真值诊断；它不属于可部署方法；
- 开发波场种子固定为 5501；只有下述方位门与 EGF 联合门均通过，才运行冻结盲测种子
  5601–5605。每个连续权重固定 20 次时间置乱对照；查看开发结果后不得改阵列、网格、正则项、
  `N_eff` 或 Gate 阈值，除非先升计划版本。

方位估计必须先过两层门：

1. 解析 plane-wave 单元测试覆盖 12°/75°/168°、200/280/360 m/s 和带噪条件；backazimuth
   中位误差不超过 5°、90% 不超过 10°，速度误差不超过一个 40 m/s 网格；
2. DWM 稀疏混合开发场中，beam 估计的扇区功率与真实 `sector_event_count` 的 Spearman 相关
   在三个偏置场均不低于 0.25，识别“该窗是否含扇区事件”的 ROC AUC 均不低于 0.65；各场景
   600 窗平均方位谱的最大方向簇必须落在预设扇区中心 ±10°。真值只用于这一步审计，不进入权重。

若方位门通过，`beam_uniformity` 仍必须满足第 11.1 节全部 waveform/phase/precursor/FTAN/
匹配置乱/uniform 非退化判据；仅改善源图、SNR 或单个场景不能进入盲测。若
`oracle_event_uniformity` 也不能改善 EGF，则判定“角向均匀化目标不足”；若 oracle 改善但 beam
失败，则判定“当前小阵列/beam 特征不足”。这两个失败原因必须分开报告。

### 6.8 v5.2：正负支分离的 stationary-phase 方向目标

v5.1 已证明辅助阵列能够观测方向簇，但把每个窗整体推向全方位均匀并不能改善 EGF；事件真值
oracle 同样失败，说明下一步不能继续调均匀化正则项或阈值。v5.2 检验更具体的物理假设：正、
负时延支应分别由对应 endfire/stationary-phase 方向的源贡献构造，而不是先用同一组全方位权重
混合两支。

范围在运行前冻结如下：

- 只复用 v5.1 开发种子 5501 已保存的 600 个 `positive_windows`、`negative_windows`、辅助阵列
  beam 方位谱、事件真值方位谱、active EGF、paired-clean 和固定 FTAN 参数；不重算波场，不改变
  速度模型、源数、事件率或目标站对；
- 先用两个单一远场源的解析/数值单元测试确定“源位于 0°/180°时对应哪一个 CCF 支”的符号约定；
  映射写入测试，禁止查看方法结果后交换分支；
- 每个分支的固定方向目标为以其 endfire 方位为中心 ±20° 的归一化 top-hat。20° 来自 4–12 Hz、
  150–500 m/s 和约 1.6 km 源半径下的保守 Fresnel/stationary-phase 角宽，不按 near12 或结果调节；
- 新通用算子 `angular_target_weights(P, q)` 使用与 v5.1 相同的非负目标函数、正则项 0.05 和
  `N_eff >= 0.5M` 约束。`oracle_branch_spz` 对正负支分别输入事件真值角谱；
  `beam_branch_spz` 分别输入观测 beam 角谱。两者的正、负权重独立拟合，分别叠加后再形成
  symmetric branch；
- source truth 只允许进入 `oracle_branch_spz` 和符号/机制审计。`beam_branch_spz` 不得读取源坐标、
  事件标签、active EGF 或 QEDisp；
- 每个正/负权重分别做 20 次匹配时间置乱，并用相同的分支组合形成 random control；禁止把
  oracle 权重或 active 波形用于阈值、正则项或方向宽度选择；
- Gate 顺序为：符号约定测试 → `oracle_branch_spz` waveform/两支/precursor/FTAN/uniform 联合门
  → `beam_branch_spz` 同一联合门。若 oracle 失败，判定“整窗 branch-SPZ 选择仍不足”并停止该
  路线；若 oracle 通过而 beam 失败，才判定方位谱代理不足；
- 开发仍只用 5501。只有 oracle 和 beam 都通过才生成并运行冻结盲测种子 5701–5705；未通过前
  不进入真实资料。主 FTAN 图仍只画 QEDisp 理论线，不画自动拾取。

## 7. Gate 3：正式方法、控制与多随机种子

### 7.1 主比较方法

1. `linear_stack`；
2. `robust_stack`；
3. `rmsr_ss`；
4. `temporal_reweight_antisymmetry`；
5. `nnws_upstream`；
6. 与硬筛选保留数或连续权重 (N_{eff}) 严格匹配的 random/permuted control；
7. `oracle_zero_sector`，仅作为真值知情的干净窗子集；因只保留 30/120 窗，**不再称为上限**。

PWS、tf-PWS、Nth-root 和 cluster 只作二级应用敏感性，不能替代主方法。

### 7.2 先验与防止真值泄漏

- RMSR 和 NNWS 可以使用一个宽松、带预先设定误差的参考群速度窗，因为原方法需要此先验。
- QEDisp 精确曲线和 active EGF 只能用于最终评价，不得生成筛选窗或调参。
- 阈值、损失系数、随机种子范围和频带在开发集冻结；盲测后不得改动。
- 划分单位是完整 source episode 或随机种子，不能把重叠窗口随机拆到训练与测试两侧。

### 7.3 多随机种子

- 至少一个开发种子用于排错和冻结参数；
- 至少五个未参与调参的随机种子作正式配对验证；
- 报告所有种子及配对差值，不只展示最佳结果；
- stochastic NNWS 还需区分源实现随机性与网络初始化随机性。

### 7.4 Gate 2 方法参数与随机对照预注册

均匀源收敛已经按 6.5 的判据完成，背景源数冻结为 **576**：三种子中位 active-EGF waveform
CC/phase RMS 为 0.750/0.848 rad；1152 源为 0.732/0.919 rad。源数增加并不单调改善，因此该
决定只表示 576 与最高源数已在预设容差内，不表示理想控制已成为 exact EGF。

在运行偏置场开发种子前，方法参数冻结如下：

- RMSR-SS：使用 Xie/Fan 合成口径 `G=1+5/M`，信号先验只用预注册 150–500 m/s 宽窗和
  `tau=0.5/4 s`，不读取 QEDisp 精确曲线；
- NNWS：作者 v1 逐窗标量权重、200 epochs、`lr=1e-3`、0.8/0.2/0.01 损失和
  `upstream_sample` 标准差口径；网络初始化种子与波场种子分开记录；
- robust：固定 StackMaster 提交 `66909971...` 的原始 Pavlis–Vernon/Yang `robust`，全 lag
  求权重后将同一权重应用于正负支；
- temporal antisymmetry：Li scheme V 的**单台对归一化适配**，在 0–5 s 正负支差的 L2
  二次型上求非负、和为 1 的权重。Li 的网络级 scheme V 以所有台对 antisymmetry 为噪声、
  以自相关幅度和的平方为信号；本双台合成没有独立自相关幅度集合，故以已白化短 CCF 的单位
  和约束替代信号归一化。输出必须标为 adapter，不能称完整复现 Li 网络级实现；
- oracle：仅偏置扇区幅度为 0 的窗权重为 1；只作真值知情的干净窗子集，不参与调参且不称为
  性能上限；
- 对 RMSR、robust、temporal、NNWS 和 oracle 各生成 20 个固定种子的时间置乱权重对照，严格
  保持权重分布、硬筛选窗数和 (N_{eff})。主图中的 random control 取其 active-EGF CC 最接近
  20 次中位数的一次，不取最佳一次；定量表保留全部 20 次；
- 开发波场种子固定为 4301，只用于排错和冻结实现；盲测波场种子固定为
  4401/4402/4403/4404/4405。开发结果检查完成后若没有书面 v4.9 变更，不得再调方法参数。

### 7.5 开发种子后的 v4.9 数值修复与诊断边界

v4.8 开发运行已完成且结果已查看，因此以下变更在任何重跑或盲测前显式冻结；源几何、活动序列、
频带、窗口、RMSR 阈值、robust 参数和所有评价指标均不改变：

- Li 单台对 adapter 的二次型目标在当前白化 CCF 上约为 (10^{-13})，SLSQP 因绝对终止容差在
  均匀初值处一次迭代即停止。实现改为用“均匀权重目标值”缩放二次型，使均匀目标为 1；该正
  常数缩放不改变理论 minimizer，只恢复数值尺度不变性。必须增加 CCF 振幅整体缩放下权重不变
  的回归测试；
- 原版 `nnws_upstream` 在 far75/near12 的四个场景出现全零或非有限权重，仍作为正式失败保留，
  禁止被替换。新增 `nnws_population_eps_diagnostic`，唯一差异是把权重标准差的零方差梯度改为
  带 epsilon 的 population 口径；它只用于判断故障是否来自上游数值奇点，不得称为原版 NNWS
  成功；
- `oracle_zero_sector` 的标签为兼容已生成文件暂不改名，但语义改为“真值知情干净窗子集”，
  不能称 upper bound。每个偏置场另外使用相同 720 源几何、相同随机数流和相同 120 窗，仅将
  扇区活动置零，生成 `paired_clean_counterfactual`；它是机制真值，不是可部署选择器；
- additional-power 与 constant-total-power 在 spectral-whitening 轨道得到完全相同 CCF，符合
  逐窗全局幅度被白化消除的预期。这两个版本仍保留并报告等价性，不重复解释为两份独立证据；
- v4.9 修复先只重跑开发种子并做权重—真值、active-EGF、匹配置乱和 paired-clean 审计。若没有
  新的书面计划版本，不再依据开发结果改变任何方法参数，然后才可运行五个盲测种子。

## 8. 预处理双轨

### 8.1 Phase/dispersion track

按 Xie/Fan 的论文条件使用 one-bit 和 spectral whitening，主要评价：

- 波形形态和相位；
- 群/相速度频散；
- precursor、双支对称性和收敛；
- 不解释绝对振幅。

### 8.2 Amplitude-preserving track

使用 raw CCF 或仅 demean/detrend/taper/bandpass 等线性预处理，重点比较 linear、robust 和
temporal reweighting，并评价：

- 固定全局比例后的波形和频谱；
- 相对振幅与距离衰减；
- 相位、前到时能量和双支保真。

两条轨道必须分开出图、分开下结论。

## 9. 评价指标层级

### 9.1 独立真值主要终点

- 与 active EGF 的全时窗 waveform correlation；
- 不允许最佳时移的 normalized (L_2) error；
- active–NCF 复交叉谱 wrapped phase residual；
- 固定全局标量后的相对振幅/谱形误差；
- precursor、tail 和零时延附近的非目标能量；
- causal 与 reversed-anticausal 两支分别相对 active EGF 的误差；
- 群速度 (T-U) 图与 active/QEDisp 的一致性；
- 相速度正确 (2\pi) 周期支相对相邻假支的能量优势。

### 9.2 机制指标

- 扇区原始源功率与有效权重后源功率；
- retained/rejected source distribution；
- stationary-phase 与非 stationary-phase 源的保留率；
- 权重与真实源标签的关系，仅用于合成 audit；
- (N_{eff}=(\sum w)^2/\sum w^2)。

### 9.3 内部/循环指标

SNR 和 antisymmetry 是 RMSR/NNWS/temporal reweighting 的目标或近邻指标，因此必须报告，但不能
单独作为独立成功终点。

## 10. 正式图件

### Figure 1：源场与选择机制

- 全部源、台站、stationary-phase zone 和时间激活；
- 原始、加权后、retained、rejected 源分布；
- 各方位扇区的原始和有效源功率；
- source truth 只出现在 synthetic audit 图，不进入算法输入。

### Figure 2：波形—群速度—相速度证据链

每种源场一张 3 行 × 6 列图：

- 列：active EGF、linear、RMSR、temporal reweighting、NNWS、equal-budget random/permuted；
- 第一行：对应双台 NCF 波形，正负支分开并叠加 active EGF；
- 第二行：群速度 (T-U) 能量谱与 QEDisp (U(T))；
- 第三行：相速度 (T-c) 能量谱与 QEDisp (c(T))。

主 FTAN 图不得画自动拾取线。必须输出：

1. 使用固定全局振幅标定和公共色标的绝对/相对能量版；
2. 仅供观察脊线形态的逐周期归一化版。

逐周期归一化版不能用于证明振幅或 SNR 改善。

### Figure 3：多随机种子配对结果

显示每个盲测种子的 `method - linear` 和 `method - matched random` 配对差值、中心统计和明确的
种子离散度；不得只画均值或挑选代表种子。

### Figure 4：实际双台资料

- 短期资料各方法波形和长期参考；
- sliding-window 时间稳定性；
- 正负支、precursor、相位和频散覆盖；
- 多站距、多方位和留出日期结果；
- 若存在独立 active/MASW 结果，单独作为外部参考。

## 11. 成功与失败判据

### 11.1 合成成功必须同时满足

- 相对 linear 和匹配 random control，active-EGF waveform CC 提高且 (L_2) error 降低；
- wrapped phase residual 降低；
- causal 与 anticausal 两支不是只改善一支；
- precursor energy 不增加；
- 群速度能量更接近 active/QEDisp；
- 相速度增强的是正确周期支而不是相邻假支；
- 均匀源场没有实质退化；
- 在预先冻结的多数盲测种子上方向一致。

振幅保持轨道还必须满足相对振幅和谱形非劣性。

### 11.2 降级结论

- 只改善 SNR/显示：称为“提高可见性或收敛”。
- 改善相位/频散但损害 precursor 或振幅：称为“适用于相位/频散任务”。
- 合成通过、实测稳定但无外部 GF：称为“合成真值支持、实测一致性支持”。
- 只有合成 active 真值、振幅保持和实测外部参考均通过，才讨论更强的 EGF 等价性。

## 12. Gate 4：实际双台资料

真实资料没有 exact EGF，因此不设置“真实 EGF 已恢复”的判据。流程为：

1. 在开发站对/日期冻结所有参数；
2. 在未见日期、未见站对和不同方位上运行；
3. 与长期 linear/robust reference、正负支一致性和外部 active/MASW/已有频散比较；
4. 报告失败站对和不足窗口，禁止回退成全窗口后仍标记为方法结果；
5. 只将结果解释为稳定性、可重复性与外部一致性证据。

## 13. Gate 5：子台阵 FJ/PS（当前暂停）

子台阵路线不属于当前双台验证。重新开启前必须单独更新本计划并定义新的研究问题。

合法路线只有：

1. 每个站对独立选择/加权后，再组装 virtual-source gather；或
2. 使用 beamforming、tau-p、3C coherency/polarization 等阵列级源方位选择器。

不得把一个站对的时间 mask 直接称为整个子台阵的 stationary-phase 筛选。FJ/PS 重新启动时必须
先完成 `active array → FJ/PS → QEDisp phase velocity` 验收，并明确空间 Nyquist、孔径分辨率和
公共有效频带。FJ/PS 只评价相速度，不得冒充群速度 FTAN。

## 14. 实施票据与当前状态

### 14.1 P1 / Gate 0 结果冻结

- 常速度解析 GF：正负极性、EGF/CF 输入、300/600/900 m 和多 Gaussian `alpha` 的群速度能量
  真值回归通过；
- 人工色散解析 GF：不使用 tracked picker 时，中位/90% 群速度误差约 0.75/1.74 m/s；
- QEDisp 原生群速度已通过本地 `sregn96`/`slegn96` 接入，原生值与相速度微分值的中位差仅
  0.013 m/s；
- 600 m active ZZ：2–12 Hz 群速度中位/90% 误差为 3.18/16.83 m/s，故全带失败；4–12 Hz 为
  0.91/2.54 m/s，相位最近理论亮脊为 0.61/1.17 m/s，故公共频带通过；
- active 阵列相速度在 2–12 Hz 相对 QEDisp 的中位/90% 误差为 0.60/1.09 m/s，说明正演波场和
  理论相位一致，低频失败来自 pair group-FTAN 分辨率而非速度模型相速度错误；
- Gate 0 主图已改为 active 波形 + (T-U) + (T-c)，只画理论线，不画自动拾取。

### 14.2 P2 / Gate 1 当前结果

- 作者公开提交、四套 CCF/源表、保存权重和 StackMaster 参考提交已固定并记录 SHA-256；
- linear、`G=1+5/M` RMSR、保存 NNWS 权重和 selective CC≥0.2 已在四情景统一重算；
- 作者保存权重的 uniform/azi40/azi65/azi90 全窗 CC 分别为
  0.904/0.849/0.805/0.765，均优于对应 linear 与 RMSR；
- azi40 中，附加扇区有效源比例由 raw 0.099、RMSR 0.095 降到 NNWS 0.055，台站轴向 ±10°
  比例增至 0.257；该源场变化与伪峰、precursor、波形和群速同时改善一致；
- 不使用保存权重的公开网络重训在四情景各 5 个初始化种子全部成功；azi40 CC 范围为
  0.777–0.845，高于 RMSR 的 0.462；
- azi40 的 ±10% 参考速度测试仍优于 RMSR，但 +10% 时 NNWS 中位 CC 从 0.807 降至 0.708，
  已登记为真实敏感性；
- 1000 s 论文映射与 1001 s 代码映射改变绝对源比例，但不改变 NNWS 增加轴向、压低 azi40
  扇区的方向；
- 正文 SNR 无法从公开数据逐数值复现；DOWS 初始化/源码未公开，故只能作为透明重建和不确定
  性扫描，不能冒充 Figure S8 精确复现。
- DOWS 已按 SI 文字定义透明重建并扫描未公开初始化尺度；`Normal(0,1)` 下四情景 CC 中位数为
  0.550/0.407/0.451/0.527，且 azi40 对初始化尺度敏感。该结果只说明公开信息不足以唯一复现
  Figure S8，不作为“DOWS 无效”的证据。
- 固定 StackMaster `66909971...`、`stockwell==1.2` 和 `tfpws(..., p=2)` 的原版 tf-PWS 已完成；
  uniform/azi40/azi65/azi90 的全窗 CC 为 0.500/0.375/0.422/0.468，与 linear 基本相当，远低于
  保存 NNWS 权重；tf-PWS 不产生逐窗权重，不能进行 retained/rejected source 机制审计。
- overlap/resegmentation 因作者未发布连续合成记录而不可忠实执行；该不可运行项登记为上游数据
  可用性限制。其余可运行复现和消融均已完成，因此 Gate 1/P2 正式关闭，结论仍限定为有条件的
  合成机制可靠，不外推实际资料 EGF。

完整结果和限制见 `docs/NNWS_GATE1_REPRODUCTION_CN.md`。

### 14.3 Gate 2 开发审计（v4.9 负控制与 v5.0 稀疏事件）

- v4.9 持续混合源负控制的 paired-clean symmetric CC 为 0.682；加入 far75/near12/mirror168
  扇区后 linear CC 分别降到 0.544/0.434/0.342，证明方位偏置足以破坏 NCF；
- 同一负控制中 RMSR 几乎保留全部 120 窗，原版 NNWS 在 far75/near12 数值失败、在其余场景近
  常权；robust 和 temporal 也未同时优于 linear 与匹配置乱。该结果支持“持续同时混合源不能仅靠
  时间删窗修复”的适用边界，不用于否定文献的稀疏 episode 结论；
- v5.0 开发种子实际生成 10,145 个背景事件和每个偏置场 1,039 个扇区事件。uniform linear 的
  CC/phase RMS 为 0.863/0.840 rad，三个 paired-clean 为 0.761/0.887 rad，满足预注册
  CC ≥ 0.65、phase RMS ≤ 1.2 rad 的事件基线门槛；
- far75/near12/mirror168 污染后的 linear CC 为 0.732/0.557/0.689。开发种子中 NNWS 仅变为
  0.736/0.557/0.688，RMSR、robust 与 temporal 均未形成跨指标优势；各可部署方法的平均扇区/
  背景事件权重比约为 0.98–1.07，没有显示系统方向簇抑制；
- active-EGF common NNLS 真值诊断以约 83–95 个有效窗可达到 CC≈1，证明当前 CCF 集合在数学上
  存在接近真值的非负时间组合；这不是可部署方法，也不能用来证明 NNWS/RMSR 有效；
- 五个冻结盲测种子 5401–5405 已全部完成。原版 NNWS 在 20 个场景×种子组合中失败 10 次：
  uniform 2/5、far75 4/5、near12 1/5、mirror168 3/5；失败均按 method-case 失败保留，没有
  回退、替换或从汇总中隐藏。`nnws_population_eps_diagnostic` 仍失败 4 次，说明故障不能只归因于
  sample-standard-deviation 的零方差梯度；
- RMSR 在 far75/near12 的 active-EGF CC 中位增量仅为 +0.00028/+0.00061，且各只有 3/5 种子
  同时改善 CC/L2；mirror168 与 uniform 的中位 CC 增量为 -0.00414/-0.00280。其扇区/背景事件
  有效权重比中位数均约为 1，未实现方向簇抑制；
- robust 在 far75/near12/mirror168 的中位 CC 增量为 -0.0173/-0.0319/-0.00979；temporal
  adapter 分别为 -0.1680/-0.0274/-0.1533，并在 uniform 退化 -0.0753。原版 NNWS 的成功运行
  仅产生约 10^-4–10^-3 的差值且有效源比约为 1，不能与 50% 运行失败率分开解释；
- 所有可部署方法均未同时满足 waveform CC/L2、phase、precursor、匹配置乱、FTAN 与 uniform
  非退化的预注册成功判据。Gate 2/3 的 phase/dispersion 轨道因此判为**完成但未通过**；不进入
  amplitude-preserving 轨道，也不进入真实资料 efficacy 验证。active-common NNLS 在全部盲测
  种子仍可达到 CC≈1，只证明当前短 CCF 集合存在真值知情的可达权重，不证明任何可部署筛选器
  有效；
- 完整结果、逐种子配对证据、FTAN 和源事件审计见 `docs/GATE2_SPARSE_EVENT_RESULTS_CN.md`。

### 14.4 P8 / Gate 2-S v5.1 source-aware 结果

- 13 台辅助 Z 小阵列的解析 plane-wave 单元测试通过；DWM 开发场 far75/near12/mirror168 的
  扇区功率–事件数 Spearman 为 0.5703/0.5973/0.6270，扇区存在 AUC 为
  0.8029/0.8295/0.8708，平均 excess 峰误差为 0°/2°/3°，方位门通过；
- `beam_uniformity` 在 uniform/far75/near12/mirror168 的 symmetric waveform CC 相对 linear
  分别变化 -0.00126/-0.00145/-0.00901/-0.00073，L2 均增加；只有 far75 的 phase 和
  pre-arrival 局部改善，四个场景均未通过 waveform 联合门；
- 独立 FTAN 审计不使用自动拾取。far75 的 group/phase image CC 和 QEDisp group-line energy
  局部提高，但正确相速度周期支优势下降；near12、mirror168 和 uniform 也没有四项联合改善；
- 使用事件真值但保持同一“整窗角向均匀化”目标的 `oracle_event_uniformity` 同样在全部场景
  失败且 waveform 退化更强。因此失败分类为“共同时间窗角向均匀化目标不足”，而不是当前
  beam 方位观测不足；
- 冻结盲测 5601–5605 未获授权，真实资料 efficacy 验证仍不进入。完整证据见
  `docs/GATE2_SOURCE_AWARE_RESULTS_CN.md`。

### 14.5 P9 / Gate 2-S v5.2 branch-SPZ oracle 结果

- 单一延迟波形单元测试将 0°/180°源固定映射到 positive/negative branch，未按结果交换符号；
- truth-only `oracle_branch_spz` 在 uniform/far75/near12/mirror168 的 symmetric CC 相对 linear
  分别增加 +0.02985/+0.02071/+0.04903/+0.01650，L2 和 pre-arrival 在四场景均下降，且均优于
  匹配时间置乱；
- 仍有 uniform/far75/mirror168 至少一个单侧 branch CC 下降，far75/near12 phase RMS 增加；
  因此两支波形联合门未通过；
- FTAN 无自动拾取。far75 虽改善 group/phase image CC 和 QEDisp group-line energy，但正确相速度
  周期支优势下降；其他场景也未实现四项联合改善；
- oracle 联合门失败，失败分类为“25 s 多事件混合窗的 whole-window branch-SPZ 选择不足”。按
  oracle-first 规则不运行 `beam_branch_spz`、5701–5705 盲测或真实资料。完整证据见
  `docs/GATE2_BRANCH_SPZ_RESULTS_CN.md`。

| 票据 | 内容 | 状态 |
|---|---|---|
| P0 | 固化 canonical plan、旧协议状态和变更治理 | 已完成 |
| P1 | 重新审计群/相 FTAN；常速度解析 + 单一 1-D active Gate 0 | **已完成；4–12 Hz 通过** |
| P2 | 忠实复现 Xie/Fan 常速度合成和作者消融 | **已完成；overlap 因上游无连续记录而登记为不可运行限制** |
| P3 | 生成 Gate 2 四类源场及配对反事实 | **已完成；v4.9 为持续源负控制，v5.0 稀疏事件基线通过** |
| P4 | 接入 linear/robust/RMSR/temporal/NNWS/random/oracle 公平比较 | **已完成；负结果，无可部署方法通过** |
| P5 | 固定开发/盲测划分并运行多随机种子 | **已完成；5401–5405 全部盲测，未挑种子** |
| P6 | 实际双台资料时间外和站对外验证 | **未进入：Gate 2/3 无通过方法，禁止作为 efficacy 证明** |
| P7 | 子台阵 FJ/PS 独立研究 | 暂停 |
| P8 | 辅助二维小阵列 beamforming + 双台方位均匀化权重 | **已完成开发门：方位通过，EGF/FTAN 联合门失败；未运行盲测** |
| P9 | 正负支分离的 stationary-phase 方向目标 | **已完成 truth-only oracle：总体 CC/L2/precursor 改善，但两支/phase/FTAN 联合门失败；停止 beam 与盲测** |

当前代码已有 active EGF、正负支 NCF、多策略权重、NNWS 适配、QEDisp、FTAN 和若干源场/图件。
这些是可复用组件，不代表相应 Gate 已通过。此前 V2/V3、低速夹层、FJ/PS 和自动拾取结果均视为
探索性历史结果，不能进入新的正式证据链，除非按本计划重新运行并通过验收。

## 15. 可追溯与交付要求

- 每个时间窗保存 `window_index/start_time/end_time`、方法权重、mask、分数和版本；
- 每个方法的波形、FTAN、指标、窗数和权重来自同一 mask；
- 空权重或低 (N_{eff}) 报告 insufficient，禁止回退；
- 保存解析配置、随机种子、Git revision、理论后端、预处理轨道和公共频带；
- 定量图输出 source-data CSV/JSON 与 PNG/PDF/SVG/TIFF；
- 正式运行后更新本文件的票据状态和结果判定；
- 修改代码后运行相关单元测试、全套测试、图件 QA、`git diff --check` 和 CodeGraph 同步。

## 16. 计划变更记录

| 日期 | 版本 | 变更 | 原因/证据 | 影响 |
|---|---|---|---|---|
| 2026-08-15 | v5.2 | 在 v5.1 方向门通过但共同角向均匀化及其 event-truth oracle 均失败后，新增正负支独立的 endfire/SPZ 固定方向目标；冻结 ±20°、原正则项、原 `N_eff`、符号约定单元测试、oracle-first Gate 和新盲测种子 5701–5705 | v5.1 证明方向簇可观测，但一个 25 s 窗混合多方位事件；共同权重无法只削弱异常方向，也把两个 CCF 支混在同一目标中。source-direction 选择与孤立源研究要求按 stationary-phase 方向及 causal/acausal 支分别处理 | 先用 source-truth branch-SPZ oracle 判断“整窗分支选择”本身是否有上限；oracle 不通过即停止，不允许转入真实资料或继续调参 |
| 2026-08-15 | v5.1 | 在 pair-local v5.0 全部可部署方法失败后，新增辅助 13 台二维小阵列的 phase-only beamforming 方位门，并将方位谱均匀化权重只施加于原 600 m 目标站对 CCF；冻结解析/DWM 方位验收、优化目标、有效窗约束和新开发/盲测种子 | v5.0 的扇区/背景有效权重比约为 1，但 active-common NNLS 可达到 CC≈1，说明权重表达能力存在而 pair-local CCF 缺少可靠的源方向观测。直接使用现有 3C 随机力混合器会新增独立 Fx/Fy/Fz 事件并改变 v5 物理问题；Z 小阵列可保持垂直力、ZZ 和事件目录不变 | 先验证方位估计，不通过则停止；通过后才评价 `beam_uniformity`，再通过才运行 5601–5605。该阶段不是 FJ/PS，不授权真实资料 efficacy 结论 |
| 2026-08-15 | v5.0 | 将 Gate 2 主验证从“每窗所有源连续同时激发”改为文献一致的 600 窗稀疏 Poisson-Ricker episode；固定期望 10,000 背景 + 约 1,000 扇区事件、one-bit→whitening、逐事件源审计、paired-clean 和 active-NNLS 可达上限 | Xie 2020 明确为 10,000 瞬态源/500 段、每段约 20 事件；Fan 2026 为 10,000 背景瞬态源/600 段并加约 10% 方向簇。v4.9 连续源场虽被方向偏置破坏，但没有可按时间分离的稀疏 episode，方法负结果不能检验论文所声称机制 | v4.9 输出保留为持续混合源负控制；新脚本和新随机种子独立运行。只有 v5.0 事件基准先通过，才评价方法并进入五种子盲测 |
| 2026-08-15 | v4.9 | 在已查看 v4.8 开发结果后，修复 temporal adapter 目标的纯数值尺度；保留原版 NNWS 失败并新增零方差梯度诊断轨道；将 30 窗 oracle 降级为干净子集，并新增同随机数 120 窗 paired-clean 反事实 | v4.8 中 temporal 在约 (10^{-13}) 目标上一次迭代停于均匀权重；原版 NNWS 在 far75/near12 全零或非有限；30 窗 oracle 的采样方差使其并非性能上限；两个功率版本经逐窗白化后完全相同 | 这些是实现尺度修复和证据标签纠正，不改变物理场景或评价判据；必须重跑开发审计，原版 NNWS 失败不得被诊断版本覆盖，随后参数冻结进入盲测 |
| 2026-08-15 | v4.8 | 按预注册收敛结果冻结 576 个背景源；在查看偏置场结果前冻结 RMSR、NNWS、StackMaster robust、Li scheme-V 单台对适配、oracle 和 20 次匹配置乱对照；固定 1 个开发与 5 个盲测波场种子 | 72/144 明显不足，288 未进入高源数平台，576 与 1152 在预设 active-EGF CC/phase 容差内；Li 2020 明确 scheme V 的网络级噪声/信号定义，但当前双台白化合成没有独立的全阵自相关信号项，需透明记录适配边界 | P3 背景源数已冻结；可运行开发种子。任何看过开发结果后的参数更改必须先升版并说明，不能静默优化 |
| 2026-08-15 | v4.7 | 在 Gate 2 方法运行前预注册双台几何、120 窗、平顶源谱、72–1152 源的三种子收敛判据，以及 75°/12°/168° 三个间歇扇区和 additional/constant-power 配对构造 | 旧 V3 的 216 源和 3.5–9.5 Hz 是探索参数；正式源数必须由理想均匀场相对 active EGF 的收敛决定，且恒功率反事实可区分方位偏置与简单能量剔除 | P3 先执行均匀场收敛并冻结背景源数，之后才生成偏置场；不得因某方法图面更好而回改源数、扇区或活动序列 |
| 2026-08-15 | v4.6 | 完成固定 StackMaster 原版 `tfpws(..., p=2)` 四情景对照；正式关闭 Gate 1/P2；将 overlap/resegmentation 明确登记为上游连续记录缺失导致的不可运行项 | tf-PWS 四情景 CC 与 linear 基本相当且无逐窗源审计能力；全部其他可运行主复现、重训和消融均已完成 | 不能把 tf-PWS 视觉锐化作为 EGF 恢复证据；主线获准进入 Gate 2，Gate 1 结论仍只支持作者尺度上的有条件合成机制 |
| 2026-08-15 | v4.5 | 用户授权继续推进后，固定 `stockwell==1.2` 为 StackMaster tf-PWS 的可选依赖；tf-PWS 必须调用固定提交中的原始 `tfpws(..., p=2)`，不得用 scipy STFT/PWS 替代 | StackMaster 1.2.1 源码直接依赖 PyPI `stockwell`，当前环境此前缺失；固定版本和原函数可消除实现替换歧义 | P2 可完成最后一个可运行 SI 消融；仍无法复现的 overlap 仅因上游未发布连续合成记录，作为数据可用性限制保留 |
| 2026-08-15 | v4.4 | 按 SI 文字重建 DOWS，固定 5 seeds，并扫描未公开的 raw-weight 初始化尺度 0.1/1/3；明确将结果标为 reconstruction 而非 Figure S8 精确复现 | 论文 SI 未提供 DOWS 源码和初始化；实测重建结果对该缺失参数敏感，不能选择单一初始化冒充作者实现 | DOWS 不再是未执行项，但只作为不确定消融；P2 剩余可运行项为获准接入 StackMaster `stockwell` 后的 tf-PWS，overlap 因无连续合成记录保持不可复现 |
| 2026-08-15 | v4.3 | 完成 Gate 1 四情景 linear/RMSR/NNWS/selective 主复现、retained/rejected/effective source audit、5 个初始化种子重训、±10% 参考速度和 1000/1001 s 映射敏感性；P2 保持进行中 | 公开权重和重训均支持 NNWS 机制，但正文 SNR 与公开数据不一致；DOWS 缺精确初始化/源码，tf-PWS 可选依赖未接入；不以未完成消融换取“正式通过” | Gate 1 可用于支持作者尺度的有条件合成机制；不得外推真实 EGF 或进入 Gate 2 正式比较，直至 P2 剩余消融完成或书面降级处置 |
| 2026-08-15 | v4.2 | 冻结 Gate 1 的作者原始数据、200 km/3 km/s/5–16 s/`alpha=2.5` 参数；明确 Gate 1 不沿用 Gate 2 的 4–12 Hz；主 RMSR 固定 `G=1+5/M`，将作者均匀源频散面板的 `1/M` 与 1000/1001 s 映射差异改为显式敏感性 | 上游提交 `04af38f...`、正文、SI 与绘图源码逐项核对后发现两个可影响结果的论文—代码差异；预注册后再运行可避免按结果挑阈值或混用尺度 | P2 复现脚本、源映射、指标表和图件必须同时记录主口径与 legacy sensitivity；Gate 2 仍固定 4–12 Hz |
| 2026-08-15 | v4.1 | 接通 QEDisp 原生 `sregn96/slegn96` 群速度；完成 Gate 0；将 2–12 Hz 候选频带收缩为 4–12 Hz 公共双台频带；固定 600 m、2 m/s 网格和 `alpha=120`；主图禁用自动拾取线 | 解析 GF 严格回归通过；active 相速度 2–12 Hz 通过，但当前模型 2–4 Hz 群速度曲率在 600 m 有限距离下不满足包络分辨率门槛；4–12 Hz 群/相速度均通过 | Gate 2–3 所有方法固定 4–12 Hz 和同一 FTAN 参数；2–4 Hz 仅作失败诊断；P1 完成，下一票为 P2 |
| 2026-08-15 | v4.0 | 将主线冻结为 Gate 0–4 双台 CCF 选择/加权；FJ/PS 后置为独立 Gate 5；加入 Xie/Fan 复现、Li temporal reweighting、raw/one-bit 双轨、active FTAN 真值和恒定总功率反事实控制 | 重新核对 Xie 2020、Fan 2026 正文与 SI、Li 2020、Yang 2023、Ning 2022；这些方法的核心证据单元是 pair-local short CCF，且 SNR 不足以证明 EGF 保真 | 旧 V3 subarray 主协议暂停；现有 FTAN 和所有探索结果需按新 Gate 重验 |
