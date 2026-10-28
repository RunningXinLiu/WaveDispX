"""Structured Chinese parameter guidance for the native desktop editor.

Every known TOML scalar has a user-facing name, unit, hard constraint,
recommended starting point, adjustment consequence and (where applicable)
enumerated choices.  The editor uses the same information for table labels,
rich hover tips, the persistent help card and enum combo boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from typing import Any


Choice = tuple[str, str]


@dataclass(frozen=True)
class ParameterSpec:
    label: str
    unit: str
    summary: str
    constraint: str
    recommended: str
    effect: str
    choices: tuple[Choice, ...] = ()


def H(
    label: str,
    unit: str,
    summary: str,
    constraint: str,
    recommended: str,
    effect: str,
    choices: tuple[Choice, ...] = (),
) -> ParameterSpec:
    return ParameterSpec(label, unit, summary, constraint, recommended, effect, choices)


BOOL_CHOICES: tuple[Choice, ...] = (("true", "开启"), ("false", "关闭"))

SECTION_HELP = {
    "run": ("运行与输出", "算例名称、模式、结果目录、CPU 并行和正演内存/精度参数。"),
    "model": ("速度—密度模型", "一维水平层状 Vp、Vs、密度和 Q；最后一层必须是半空间。"),
    "geometry": ("观测系统 Geometry", "接收器线阵、规则二维面阵或任意二维 CSV。"),
    "das": ("DAS 有限标距", "把点接收器中心转换为有方向、有标距的轴向应变或应变率通道。"),
    "active_source": ("主动源与子波", "炮点、载荷分量、Ricker/单位脉冲子波和主动记录采样。"),
    "ambient_sources": ("环境噪声源", "被动源分量、空间分布、数量和空间随机种子。"),
    "noise": ("噪声时间模型与 PSD", "有限记录时长、独立窗、采样、功率谱和时间随机种子。"),
    "source_selection": ("噪声源筛选", "按已知正演源照明生成可审计的 oracle 权重与公平随机对照。"),
    "nnws": ("逐台站对 NNWS", "Fan 等（2026）神经网络加权叠加的默认关闭研究流程。"),
    "processing": ("频散处理与拾取", "互相关归一化、成像核、频率—速度网格、拾取和理论模态。"),
}

GROUP_LABELS = {
    "run": "运行、输出与性能",
    "model": "速度—密度模型",
    "model.layers": "模型分层",
    "geometry": "观测系统 Geometry",
    "geometry.receivers": "接收器",
    "das": "DAS 光纤与测量算子",
    "active_source": "主动源与源时间函数",
    "ambient_sources": "环境噪声源空间分布",
    "noise": "噪声时间模型与 PSD",
    "source_selection": "Oracle 噪声源照明筛选",
    "nnws": "逐台站对神经网络加权叠加",
    "processing": "互相关、频散与拾取",
}


PARAMETER_SPECS: dict[str, ParameterSpec] = {
    # Run and performance.
    "run.name": H("算例名称", "—", "结果和图件使用的算例标识。", "不能为空。", "简短英文、数字和下划线，例如 passive_site01。", "只改变名称和元数据；不会改变数值结果。"),
    "run.mode": H("模拟模式", "—", "选择主动炮集或真实有限记录环境噪声流程。", "只能是 active 或 passive；改变模式时应载入同模式模板。", "主动炮集选 active；背景噪声互相关选 passive。", "切换模式会改变必需的源参数、观测阶段和默认频散方法。", (("active", "主动源炮集"), ("passive", "被动环境噪声"))),
    "run.output_directory": H("输出目录", "路径", "保存 HDF5、CSV、summary、manifest 和可复现 input.toml。", "必须是当前用户可创建或写入的目录。", "每个算例使用独立目录，避免结果互相覆盖。", "不改变计算；同名文件会在保存阶段更新。"),
    "run.workers": H("CPU workers", "线程", "C++正演及支持并行的径向变换所请求的工作线程数，不是多炮/多任务数量。", "整数且≥0；0由内核按硬件并发数自动选择，并受当前任务数限制。", "可先保留模板4；需要腾出CPU时设1或2。0不意味着最快。", "不控制所有NumPy/BLAS、QED或界面线程，不保证全程序CPU占用上限；增加线程不保证线性提速，也不改变模型或源数量。"),
    "run.frequency_block_size": H("频率分块大小", "频点/块", "频散成像等支持分批的步骤每批处理的频点数；不是频率步长df，也不是正演截断频率。", "整数且≥1。", "可先保留模板8；内存紧张时试4。", "影响临时数组大小与调度；不保证块越大越快，不改变请求频率坐标。"),
    "run.source_block_size": H("噪声源分块大小", "源/块", "合成混合噪声记录时每批处理的源数，不是源总数。", "整数且≥1。", "可先保留模板8；主动模式通常不使用此项。", "影响临时内存和累加顺序；可能有浮点舍入差异，不改变源位置、源数或功率定义。"),
    "run.radial_interpolation_step_m": H("径向 Green 插值步长", "m", "源—接收器距离方向的 Green 函数插值间隔。", "必须 >0；被动模式还必须 ≤ 最短 S 波长的 5%。", "先用 0.5 m；高频或低 Vs 模型按审核给出的上限减小。", "减小提高径向插值精度但增加正演时间和内存；过大会失真并被 required 检查阻止。"),
    "run.store_source_responses": H("保存逐源传递响应", "—", "是否把每个被动噪声源的完整传递函数写入结果。", "true/false。", "常规有限记录模拟选 false；仅诊断单源传播时开启。", "开启显著增加 HDF5 和内存，不改变混合记录或频散结果。", BOOL_CHOICES),
    # Layered model.
    "model.layers[].thickness_m": H("层厚", "m", "当前模型层的厚度；末层用无限半空间封底。", "有限层 >0；只有最后一层可填 inf，且最后一层必须为 inf。", "浅层面波模型常从 2–20 m 有限层开始。", "改变界面深度会移动模态拐点和高阶截止频率；增厚通常把层相关特征推向更低频。"),
    "model.layers[].vp_mps": H("P 波速度 Vp", "m/s", "当前层纵波速度，进入刚度、阻抗和 Rayleigh 椭圆极化。", "有限正数，且必须 Vp > Vs。", "无测井时可从合理泊松比约束估计，但应明确其不确定性。", "增大 Vp 主要改变 Rayleigh 椭圆性、近场和体波能量；对 Love 波影响较弱。"),
    "model.layers[].vs_mps": H("S 波速度 Vs", "m/s", "控制 Rayleigh/Love 相速度与最短空间波长的核心模型量。", "有限正数，且必须 Vs < Vp。", "优先使用已有 Vs 先验；处理速度范围应覆盖相邻层 Vs 所对应的模态。", "增大 Vs 会整体提高面波相速度；低 Vs 薄层会增强频散和高阶模态。"),
    "model.layers[].density_kgm3": H("密度 ρ", "kg/m³", "当前层体密度，进入弹性阻抗和模态振幅。", "有限正数。", "土层常见 1600–2200，岩层常见 2200–2800；应以现场资料为准。", "密度反差主要改变阻抗、振幅和模态能量分配，对相速度的影响通常弱于 Vs。"),
    "model.layers[].qp": H("P 波品质因子 Qp", "—", "控制 P 波本征衰减。", "正数或 inf；inf 表示无本征衰减。", "没有可靠衰减资料时用 inf；有限 Q 应做敏感性试验。", "减小 Qp 会增强 P 波衰减和相位色散，并降低远检距高频能量。"),
    "model.layers[].qs": H("S 波品质因子 Qs", "—", "控制 S 波及面波本征衰减。", "正数或 inf；inf 表示无本征衰减。", "没有可靠衰减资料时用 inf；有衰减时 Qs 通常比 Qp 更敏感。", "减小 Qs 会明显压低远检距和高频面波能量，可能让高阶模态更难观察。"),
    # Receiver geometry.
    "geometry.receivers.layout": H("接收器布局", "—", "选择参数化线阵、规则二维面阵或坐标文件。", "只能是 line、grid 或 file。", "常规 MASW 用 line；二维覆盖用 grid；现场坐标用 file。", "布局决定实际检距采样、径向 NCF 分箱和空间混叠特征。", (("line", "水平线阵"), ("grid", "规则二维面阵"), ("file", "CSV 任意二维阵列"))),
    "geometry.receivers.start_x_m": H("x 起点", "m", "线阵或面阵最小 x 坐标。", "有限数；必须小于 x 终点。", "按现场坐标填写；原点可以任意，但源和接收器必须使用同一坐标系。", "与终点共同决定 x 孔径；扩大孔径改善低频分辨率。"),
    "geometry.receivers.stop_x_m": H("x 终点", "m", "线阵或面阵最大 x 坐标。", "有限数且 > x 起点；跨度必须是 x 间距的整数倍。", "让最后一道明确落在终点，避免隐式舍入。", "增大终点扩大孔径和接收器数，改善低频分辨率但增加计算量。"),
    "geometry.receivers.spacing_m": H("线阵道间距", "m", "相邻线阵接收器间距。", ">0，且 (终点−起点)/道距 必须为整数。", "使空间混叠上限 cmin/(2Δx) 高于处理最高频率。", "减小道距提高高频空间采样并减少混叠，但增加道数、内存和互相关对数。"),
    "geometry.receivers.spacing_x_m": H("面阵 x 间距", "m", "规则面阵 x 方向接收点间距。", ">0，且 x 跨度必须是它的整数倍。", "通常与 y 间距相同；高频目标需要更小间距。", "减小会增加 x 方向采样和总台站数，降低空间混叠。"),
    "geometry.receivers.start_y_m": H("y 起点", "m", "规则面阵最小 y 坐标。", "有限数且必须小于 y 终点。", "与 x 坐标使用同一米制坐标系。", "与 y 终点共同决定横向孔径和方位覆盖。"),
    "geometry.receivers.stop_y_m": H("y 终点", "m", "规则面阵最大 y 坐标。", "有限数且 > y 起点；跨度必须是 y 间距整数倍。", "让最后一行接收器明确落在终点。", "扩大 y 孔径改善方位和低频覆盖，但增加台站对数量。"),
    "geometry.receivers.spacing_y_m": H("面阵 y 间距", "m", "规则面阵 y 方向接收点间距。", ">0，且 y 跨度必须是它的整数倍。", "通常与 x 间距相同。", "减小提高 y 方向高频空间采样，但接收器对数量近似按台站数平方增长。"),
    "geometry.receivers.y_m": H("线阵 y 坐标", "m", "整条线阵固定的 y 坐标。", "必须是有限数。", "水平层状模型可取 0；只需与源坐标处于同一坐标系。", "整体平移线阵不改变水平层状响应；相对炮点 y 偏移会改变真实炮检距。"),
    "geometry.receivers.distance_bin_m": H("径向距离箱宽", "m", "把二维/不规则台站对按距离归并为径向 NCF 道集。", "被动 grid/file 必填且 >0。", "先取典型最近邻间距 2–10 m，并检查每箱台站对数量。", "过小会使每箱样本稀疏、频散不连续；过大则平滑空间相位并降低高频分辨率。"),
    "geometry.receivers.coordinates_file": H("接收器坐标文件", "CSV", "读取现场或任意二维接收器位置。", "UTF-8 CSV；至少两行，含 x_m,y_m，有限且不重复。", "使用参数页“接收器 CSV”按钮选择，避免手填路径。", "坐标直接决定每个炮检距或台站对距离；错误单位会整体破坏频散速度。"),
    # DAS finite-gauge acquisition.
    "das.enabled": H("启用 DAS", "—", "在每个接收器中心两侧建立标距端点，用端点轴向位移差模拟真实 DAS 测量。", "true/false；DAS v1 只验证 phase_shift；主动/被动源分量先用 vertical。", "节点检波器基准用 false；DAS 算例用 true 并先检查几何图中的橙色标距段。", "开启后正演端点 Green 函数并输出应变/应变率，而不是对最终频散图做经验滤波。", BOOL_CHOICES),
    "das.geometry_mode": H("DAS 几何方式", "—", "选择统一标距/方位，或从接收器 CSV 逐道读取真实光缆端点。", "uniform 或 channel_file；channel_file 要求接收器 layout=file，CSV 含端点列或逐道标距/方位列。", "规则线阵/面阵用 uniform；弯曲、折返或变标距光缆用 channel_file。", "channel_file 允许每道独立方向和标距，并把文件内容写入 HDF5；它不会把折线自动当成连续有限元光缆。", (("uniform", "统一标距与方位"), ("channel_file", "逐道光缆 CSV"))),
    "das.gauge_length_m": H("标距 Gauge length", "m", "uniform 模式下每道沿光纤方向做空间平均的物理长度，端点位于道中心 ±Lg/2。", "有限且 >0；与道间距相互独立；channel_file 模式改由 CSV 每行定义。", "按仪器设置填写，常见 5–20 m；基准可从 10 m 开始。", "增大可提高长波应变信号但会加强 sinc 空间滤波；第一零点约为 f=c/Lg，可能压低高频或部分模态。"),
    "das.measurement": H("DAS 输出量", "—", "选择有限标距轴向应变，或它对时间的一阶导数应变率。", "strain 或 strain_rate。", "与现场解调输出一致；很多 DAS 地震记录使用 strain_rate。", "strain_rate 在频域额外乘 +i2πf，因此相位旋转且高频相对增强；两者单位不同，不能混称。", (("strain", "轴向应变"), ("strain_rate", "轴向应变率"))),
    "das.fiber_azimuth_deg": H("光纤方位角", "deg", "uniform 模式下所有标距相对全局 +x 轴逆时针的方向；决定水平位移投影。", "任意有限角度；0° 沿 +x，90° 沿 +y；180° 与 0° 的轴向应变等价；channel_file 模式由 CSV 逐道定义。", "线阵通常填阵列走向；规则面阵表示各中心处的平行标距；真实转弯光缆使用 channel_file。", "旋转会改变方向灵敏度；接近波前切向时轴向响应可能很弱，这是物理方向性而非程序丢能量。"),
    # Active source.
    "active_source.x_m": H("炮点 x", "m", "本次主动震源的 x 坐标。", "有限数，且不能与接收器完全重合。", "与接收器使用同一坐标原点；多炮时从 Shot CSV 明确选择一炮。", "改变炮点会改变所有炮检距、近场比例和有效径向采样。"),
    "active_source.y_m": H("炮点 y", "m", "本次主动震源的 y 坐标。", "有限数，且不能与接收器完全重合。", "线阵常取 0；偏移炮点应填写真实横向偏移。", "横向偏移改变真实二维炮检距，可能改善不规则检距采样。"),
    "active_source.component": H("主动载荷分量", "—", "决定计算 Rayleigh、Love 或二者。", "vertical、horizontal 或 both。", "未知目标时选 both；只研究竖直 Rayleigh 可选 vertical 以节省时间。", "both 计算五分量 Green 响应，信息最完整但运行和输出更大。", (("vertical", "竖直载荷 / Rayleigh"), ("horizontal", "水平横向载荷 / Love"), ("both", "两类载荷都计算"))),
    "active_source.wavelet": H("主动源子波", "—", "主动震源时间函数。", "ricker 或 unit_impulse。", "常规炮集用 ricker；验证宽频模型、观测系统和成像器时用 unit_impulse。", "unit_impulse 具有平坦离散振幅谱，只适合作为理想合成校准源。", (("ricker", "Ricker 子波"), ("unit_impulse", "单位脉冲（宽频校准）"))),
    "active_source.dominant_frequency_hz": H("Ricker 主频 / 参考频率", "Hz", "Ricker 时为功率谱主要能量中心；unit_impulse 时仅控制正演时窗与延迟。", ">0，且不能高于正演最高频率。", "Ricker 时放在目标频带中部；unit_impulse 时可取目标频带中部。", "Ricker 主频改变能量谱；unit_impulse 下不改变平坦源谱。"),
    "active_source.maximum_frequency_hz": H("主动正演最高频率", "Hz", "波场正演的频谱截断上限，不是频散图的显示上限。", "≥主频/参考频率，且严格 < 1/(2dt)。", "必须检查子波在截断处的剩余谱幅；不能只比成像上限略高。用固定模型的上限加密对照检查完整波形和频散图。", "截断过低可能产生振铃，并被时域指数校正放大；DAS 应变率尤其敏感。提高上限增加计算量，但不要求把频散图也画到该频率；成像频带另按阵列采样检查。"),
    "active_source.dt_s": H("主动时间采样间隔", "s", "炮集时间序列采样间隔。", ">0，且最高频率 < 1/(2dt)；最短周期建议至少 8 点。", "按 dt ≤ 1/(8 fmax) 选取。", "减小 dt 提高时间采样并扩大 Nyquist，但增加样点数、内存和 FFT 成本。"),
    "active_source.record_duration_s": H("主动记录时长", "s", "去除 DWM 延迟后保存并用于频散成像的炮集时长；1/T 给出有限记录的频率尺度。", "0 表示按群速度、最大炮检距和处理最低频率给出候选值；显式值必须不短于预检查建议。", "自动值不是收敛保证。检查近、中、远道完整波形，同时对记录周期和正演最高频率做敏感性检查。", "改变时长也会改变原生 DWM 频率网格和复频移，并非单纯多保存一段波形。末端振荡时不要只延长记录；FFT 补零不能增加真实记录的信息。"),
    # Ambient source geometry.
    "ambient_sources.component": H("环境源分量", "—", "选择竖直快速路径或物理三分量随机力。", "vertical 或 three_component。", "先用 vertical 调试；需要 RR/TT/ZZ 与 Love/Rayleigh 分量研究时用 three_component。", "three_component 计算 Fx/Fy/Fz 经完整 Green 张量传播，计算和内存明显增加。", (("vertical", "竖直随机力"), ("three_component", "三分量 Fx/Fy/Fz 随机力"))),
    "ambient_sources.layout": H("噪声源空间布局", "—", "选择环形照明、线性道路源或显式坐标。", "annulus、line/road 或 file。", "验证各向同性恢复优先 annulus+stratified；已知现场源位置用 file。", "源布局决定方位照明；严重单侧/线性分布会使 NCF 不对称并偏置频散能量。", (("annulus", "环形方位照明"), ("line", "道路 / 线性噪声带"), ("file", "CSV 显式源位置"))),
    "ambient_sources.count": H("噪声源数量", "个", "同时参与接收端混合的独立噪声源数。", "整数且 ≥1；file 布局应与 CSV 有效行数一致。", "环形正式收敛建议 192–384；quick 可用 24。", "增加源数主要填补方位照明空缺并降低空间采样伪影，但不能替代更长记录。"),
    "ambient_sources.seed": H("源位置随机种子", "—", "控制参数化源环/源线的可复现空间位置。", "整数且 ≥0。", "固定种子做可重复对比；不应挑选只产生漂亮结果的种子。", "改变种子只改变源位置抽样；file 布局下位置由 CSV 决定。"),
    "ambient_sources.inner_radius_m": H("源环内半径", "m", "噪声源环离阵列中心的最小半径。", "≥0，且严格小于外半径；源不能与接收器重合。", "至少大于阵列半孔径，避免源落入阵列内部。", "增大内半径使入射更接近平面波，但传播衰减和正演距离增大。"),
    "ambient_sources.outer_radius_m": H("源环外半径", "m", "噪声源环离阵列中心的最大半径。", "> 内半径。", "可取内半径的 1.2–2 倍，避免所有源处于同一半径。", "增大径向宽度可减弱规则相干伪影，但使远源振幅更弱。"),
    "ambient_sources.angular_sampling": H("源方位采样", "—", "决定源环的方位角生成方式。", "stratified、uniform 或 random。", "正式模拟优先 stratified；uniform 用于完全规则对照。", "random 可能留下大方位空缺；stratified 在每个扇区内随机，通常收敛最稳。", (("stratified", "分层随机方位（推荐）"), ("uniform", "严格等方位"), ("random", "完全随机方位"))),
    "ambient_sources.x_min_m": H("源线 x 起点", "m", "线性噪声带的最小 x 坐标。", "有限数且 < x 终点。", "让源线覆盖阵列孔径之外的足够范围。", "向外延长可增加入射角覆盖，但源数固定时会降低沿线空间密度。"),
    "ambient_sources.x_max_m": H("源线 x 终点", "m", "线性噪声带的最大 x 坐标。", "有限数且 > x 起点。", "与起点共同覆盖目标道路或噪声带。", "增大跨度扩展照明范围；源数不变时单位置采样更稀。"),
    "ambient_sources.y_m": H("源线 y 坐标", "m", "整条线性噪声带固定的 y 坐标。", "必须是有限数。", "填写道路或机械噪声带相对阵列的真实横向距离。", "离阵列更远时波前更接近平面，但振幅变弱、传播时间更长。"),
    "ambient_sources.coordinates_file": H("噪声源坐标文件", "CSV", "显式给出每个被动噪声源位置。", "UTF-8 CSV；含 x_m,y_m，坐标有限且不重复。", "使用“噪声源 CSV”按钮；同时检查 count 是否与文件行数一致。", "文件位置直接决定方位照明，空间 seed 不再改变位置。"),
    "ambient_sources.activity_file": H("逐窗源活动表", "CSV", "为每个独立窗和源指定非负振幅倍数，用于预先定义时变各向异性照明实验。", "UTF-8 长表，含 window_index,source_index,amplitude_multiplier；索引从 0 开始，未列出的源为 0，每窗必须有正功率。", "常规均匀源场不填写；筛选验证应预先声明源活动表，避免事后挑窗。", "改变它会改变同一位置源在各窗的活动强度和 NCF 照明偏差，但不会重算 Green 传递函数。"),
    # Noise time process and PSD.
    "noise.temporal_model": H("噪声时间模型", "—", "生成连续平稳噪声或有离散激发时刻的随机脉冲。", "stationary_gaussian 或 poisson_ricker。", "一般环境噪声用 stationary_gaussian；车辆/敲击事件演示用 poisson_ricker。", "Poisson 模型会生成事件目录；平稳模型没有离散激发时刻表。", (("stationary_gaussian", "平稳高斯连续噪声"), ("poisson_ricker", "Poisson 随机 Ricker 事件"))),
    "noise.seed": H("噪声时间随机种子", "—", "控制源时间函数、事件数、时刻、极性和权重。", "整数且 ≥0。", "固定种子比较模型/阵列差异；收敛研究可用多个种子做重复试验。", "改变它不改变源位置，但会改变有限记录噪声实现和 NCF 随机误差。"),
    "noise.duration_s": H("记录总时长", "s", "所有独立噪声窗累计的有限记录长度，不是NCF lag。", ">0，且为独立窗长整数倍。", "60 s仅调试；自动按钮至少60独立窗且600 s（有activity_file时不扩展）。这是候选，不保证收敛。", "增加时长降低统计方差；固定源场/种子/窗长，比较T与2T。源方位缺口、短孔径或混叠不能靠增加总时长消除。"),
    "noise.dt_s": H("噪声采样间隔", "s", "混合记录和 NCF 的时间采样间隔。", ">0，PSD 上限和处理上限必须严格 < 1/(2dt)。", "按 dt ≤ 1/(8 fmax) 保证最短周期至少约 8 点。", "减小 dt 提高高频时间采样，但成倍增加每窗样点、FFT、内存和互相关成本。"),
    "noise.window_s": H("独立窗长", "s", "每次独立合成并累积互相关的时间窗长度。", ">0、≤总时长，且必须大于最大台间传播时延。", "先满足审核给出的 required lag，再令总时长包含足够多独立窗。", "过短会截断远台传播；过长会减少独立窗数并增大单窗内存。"),
    "noise.maximum_lag_s": H("NCF 最大 lag", "s", "被动互相关图保存并送入频散成像的最大正时延，可同时检查早到体波与较慢面波。", "0 表示按模型最慢群速度与阵列孔径自动设计；显式值 >0 且必须小于独立窗长。", "先用 0；需要固定实测处理窗或做 lag 敏感性时再显式填写。", "增大可保留更晚的慢波/尾波，但增加 NCF 缓存和成像成本；过短会截断远台面波。"),
    "noise.overlap": H("窗重叠率", "—", "相邻合成/互相关窗的重叠比例。", "当前实现必须固定为 0。", "保持 0；每个窗独立生成并由流式互相关器叠加。", "当前版本修改为非零会被校验拒绝。"),
    "noise.psd_shape": H("噪声 PSD 形状", "—", "定义每个背景源的期望功率谱。", "ricker、flat_band 或 file。", "窄带平滑源用 ricker；带内均匀照明用 flat_band；实测谱用 file。", "PSD 直接控制可恢复频带；频带外没有源能量，补零不能恢复。", (("ricker", "Ricker 型平滑谱"), ("flat_band", "带内平坦谱"), ("file", "CSV 外部功率谱"))),
    "noise.dominant_frequency_hz": H("PSD 峰值频率", "Hz", "Ricker 型噪声 PSD 的能量峰值。", "> 最低频率且 ≤ 最高频率。", "放在目标处理频带中部；flat_band/file 时仅保留为配置参考。", "增大把主要源能量推向高频和浅层，但更易受衰减和空间混叠影响。"),
    "noise.minimum_frequency_hz": H("PSD 最低频率", "Hz", "噪声源具有有效能量的低频边界。", "≥0 且 < 峰值频率。", "应不高于处理最低频率，并考虑阵列孔径能否分辨该低频。", "降低它不会自动提升低频分辨率；若孔径不足只会增加不稳定低频能量。"),
    "noise.maximum_frequency_hz": H("PSD 最高频率", "Hz", "噪声源有效能量的高频边界。", "≥峰值，严格 < 1/(2dt)，且处理最高频率不能超过它。", "略高于处理上限，同时满足道距空间混叠和 Green 插值约束。", "增大扩展潜在高频，但显著收紧 dt、道距和径向插值要求。"),
    "noise.psd_file": H("外部 PSD 文件", "CSV", "从文件读取频率—功率谱。", "psd_shape=file 时必填；UTF-8，frequency_hz 严格递增、power 非负。", "使用“PSD CSV”按钮选择并先查看输入审核中的谱形。", "文件谱将替代内置 Ricker/平坦形状；错误频率单位会直接造成频带错误。"),
    "noise.source_amplitude": H("单源相对振幅", "相对量", "每个噪声源时间函数的统一振幅尺度。", "有限正数。", "通常保持 1；只在振幅动态范围试验中修改。", "线性缩放混合记录幅值；归一化频散能量通常不变，但数值动态范围会变化。"),
    "noise.independent_sources": H("各噪声源相互独立", "—", "是否为每个源生成独立随机时间函数。", "true/false。", "真实 Green 恢复和常规被动处理必须选 true；false 仅作相干源敏感性试验。", "关闭会让不同源相干，产生强交叉项和假频，不再满足标准噪声互相关假设。", BOOL_CHOICES),
    "noise.event_rate_hz": H("Poisson 事件率", "事件/(s·源)", "Poisson-Ricker 模型中每个源每秒平均事件数。", ">0；仅 poisson_ricker 模型实际生效。", "从 0.1–0.5 开始，使事件既有重叠又不过度连续。", "增大会使记录更连续、事件数和计算负载增加；过低导致有限窗内大量空源。"),
    # Oracle source-illumination selection.
    "source_selection.enabled": H("启用 Oracle 源筛选", "—", "使用正演中已知的逐窗源系数评估阵列轴向 stationary-phase 照明，并并行生成筛选与随机对照 NCF。", "true/false；当前仅支持线阵，且属于正演上限实验，不能直接用于未知实测源。", "先保持 false 验证基线；研究源分布影响时在独立示例中开启。", "开启会额外输出逐窗源照明指标、四个权重策略和审计表，但不改变 receiver-only NCF 数值核。", BOOL_CHOICES),
    "source_selection.method": H("源筛选方法", "—", "选择源侧评分算法。", "当前只能是 oracle_axis。", "线阵正演实验使用 oracle_axis。", "该方法读取已知源位置与源系数，因此只代表可达到的上限，不能冒充可观测实测筛选器。", (("oracle_axis", "已知源的线阵轴向 Oracle"),)),
    "source_selection.objective": H("源筛选科学目标", "—", "区分端射 stationary-phase 增强与完整各向同性 EGF 恢复，避免用单一分数替代不同物理任务。", "stationary_phase 或 isotropic_egf。", "定向高频频散试验用 stationary_phase；以完整 NCF–EGF 等价为主终点用 isotropic_egf。", "stationary_phase 偏好轴向端射功率；isotropic_egf 偏好高方位熵并保留双向平衡，两者可能给出不同甚至相反的选窗结果。", (("stationary_phase", "端射/SPZ 相位目标"), ("isotropic_egf", "全方位完整 EGF 目标"))),
    "source_selection.frequency_bands_hz[][]": H("源评分频带边界", "Hz", "每个 [最低频率, 最高频率] 子带独立计算源功率、SPZ 占比和双向平衡。", "二维数组；每个上限必须大于下限，并全部位于噪声 PSD 频带内。", "按目标频散频带分成 1–3 个物理子带；不要依据待评价 EGF 误差调频带。", "分带更细可揭示频率相关源偏置，但单带频点过少会增加评分方差。"),
    "source_selection.stationary_phase_half_angle_deg": H("SPZ 半角", "°", "阵列轴正、反延长方向两侧被视为 stationary-phase/endfire 的角度容差。", ">0 且 ≤90°。", "远场线阵先用 15–25°，并按阵列孔径与源距做敏感性分析。", "增大保留更多斜入射源并提高窗数，但会降低对非 SPZ 相位偏置的抑制。"),
    "source_selection.hard_min_endfire_fraction": H("硬筛 SPZ 功率下限", "0–1", "要求每个评分频带内落在正反 endfire 扇区的源功率最低占比。", "0≤值≤1；所有频带均达标才保留。", "从几何上均匀源场的预期扇区占比附近开始，不用 EGF 测试误差反调阈值。", "提高会强化轴向照明但减少有效窗，可能使方差上升或产生空 mask。"),
    "source_selection.hard_min_bidirectional_balance": H("硬筛双向平衡下限", "0–1", "约束正、反 endfire 源功率平衡，1 表示完全相等，0 表示单侧。", "0≤值≤1；所有频带均达标才保留。", "对称 EGF 实验可从 0.6–0.8 开始；只分析单支时另做消融。", "提高通常改善因果/非因果支对称性，但会剔除真实长期单侧照明并减少 N_eff。"),
    "source_selection.hard_min_azimuth_entropy": H("硬筛方位熵下限", "0–1", "isotropic_egf 目标要求的最小归一化方位功率熵，1 接近均匀全方位。", "0≤值≤1；只在 isotropic_egf 目标决定 hard mask。", "从 0.7–0.85 做预注册敏感性分析，并同时报告 N_eff。", "提高会偏好更均匀照明并减少长期源偏差，但可能大幅减少可用窗并提高有限样本方差。"),
    "source_selection.random_seed": H("随机预算对照种子", "—", "控制与 oracle hard 同窗数的随机 mask，以及与 soft 完全同权重分布的随机置换。", "整数且 ≥0。", "固定种子保证公平复现，并在正式结论中增加多个随机种子置信区间。", "改变它只改变随机对照分配，不改变噪声源实现或 oracle 分数。"),
    # Pair-resolved NNWS research workflow.
    "nnws.enabled": H("启用逐对 NNWS", "—", "缓存逐窗、逐台站对 ZZ NCF，并在每个台站对上独立训练权重。", "true/false；仅限 passive + vertical；默认关闭。", "先用 false 建立 baseline，再在独立研究算例中开启。", "开启会增加互相关缓存、PyTorch 训练和审计输出；不会改写 baseline。", BOOL_CHOICES),
    "nnws.epochs": H("NNWS 训练轮数", "epoch", "每个台站对独立优化上游 StackingNet 的轮数。", "整数且 ≥1。", "quick 用 2–10；正式复现从上游默认 200 开始并检查收敛。", "增加通常提高优化充分性，但运行时间按台站对数近似线性增长，也可能暴露数值不稳定。"),
    "nnws.learning_rate": H("NNWS 学习率", "—", "Adam 优化器每步参数更新尺度。", "有限且 >0。", "上游默认 0.001；敏感性试验应预先声明。", "过大会震荡或产生非有限权重；过小可能在固定 epoch 内未收敛。"),
    "nnws.snr_weight": H("SNR 损失权重", "—", "控制信号窗相对噪声窗 RMS 的优化贡献。", "有限且 ≥0；它与对称性权重不能同时为 0。", "上游默认 0.8。", "提高会更偏好高训练窗 RMS 比，但不保证与真实 EGF 更等价。"),
    "nnws.symmetry_weight": H("对称性损失权重", "—", "控制正、负时延支反对称能量的惩罚。", "有限且 ≥0；它与 SNR 权重不能同时为 0。", "上游默认 0.2。", "提高会偏好因果/非因果支相似的窗，但可能抑制真实单侧照明信息。"),
    "nnws.std_weight": H("权重离散度项系数", "—", "鼓励不同短窗获得有区分度的连续权重。", "有限且 ≥0。", "上游默认 0.01；必须与 std_mode 一起报告。", "增大可能强化选择性，也会放大少窗或低方差数据的数值不稳定。"),
    "nnws.std_mode": H("标准差损失模式", "—", "区分上游精确损失和显式数值稳定性消融。", "upstream_sample、population_eps 或 disabled。", "复现上游用 upstream_sample；低方差诊断可另跑 population_eps。", "后两种属于消融，不能与上游精确结果合并报告。", (("upstream_sample", "上游 v1.0 样本标准差"), ("population_eps", "总体标准差 + epsilon 消融"), ("disabled", "关闭标准差项消融"))),
    "nnws.seed": H("NNWS 随机种子", "—", "控制每个台站对神经网络初始化和训练复现。", "整数且 ≥0。", "固定为 0 复现；正式结论增加多种子稳健性。", "改变它会改变优化初值和权重，但不改变噪声实现。"),
    "nnws.random_control_seed": H("随机同 N_eff 对照种子", "—", "逐台站对置换 NNWS 权重与时间窗的对应关系，同时精确保留权重分布和 N_eff。", "整数且 ≥0。", "默认 271828，并与网络初始化种子分开。", "改变它只改变随机权重—时窗配对，不改变 NNWS 权重值、台站对支持或噪声实现。"),
    "nnws.device": H("NNWS 计算设备", "—", "选择 PyTorch CPU、CUDA、MPS 或自动探测。", "auto、cpu、cuda 或 mps；设备必须实际可用。", "通常用 auto；严格跨平台复现可固定 cpu。", "GPU/MPS 可加速大量台站对训练，但浮点实现可能带来微小差异。", (("auto", "自动选择"), ("cpu", "CPU"), ("cuda", "NVIDIA CUDA"), ("mps", "Apple Metal"))),
    "nnws.minimum_effective_windows": H("最小有效窗数", "N_eff", "按权重平方和计算每个台站对可接受的最小有效样本量。", "有限且 >0。", "至少 2；正式实验应明显更高并与 baseline 同报。", "提高会拒绝由极少数窗支配的台站对，降低覆盖率但增强统计保守性。"),
    "nnws.minimum_pair_coverage": H("最小台站对覆盖率", "0–1", "允许 NNWS 结果进入频散/EGF 评价所需的成功台站对比例。", "有限且在 (0,1]。", "默认 1.0，先要求全覆盖。", "降低允许部分台站对失败，但仍要求每个距离箱至少有有限结果，并必须报告缺失几何。"),
    "nnws.maximum_cache_mb": H("逐对缓存上限", "MiB", "限制逐窗×台站对×时延×三分支 float32 NCF 缓存。", "有限且 >0；超限会在生成前中止。", "quick 用 128–512；按台站数、窗数和最大时延估算后设置。", "增大允许更大实验但提高内存风险；不改变数值算法。"),
    "nnws.pair_scope": H("NNWS 台站对范围", "—", "选择训练全部台站对，或只训练包含一个参考接收点的虚拟源道集。", "all 或 reference_receiver。", "普通小阵列用 all；长线阵频散基准可用 reference_receiver 控制为 N-1 个台站对。", "reference_receiver 避免台站对数量平方增长，但结果只代表所选虚拟源。", (("all", "全部台站对"), ("reference_receiver", "单参考接收点虚拟源道集"))),
    "nnws.reference_receiver_index": H("参考接收点索引", "从 0 开始", "pair_scope=reference_receiver 时，指定必须包含在哪些台站对中的接收点。", "整数且 0 ≤ index < 接收点数。", "线阵端点虚拟源通常用 0，以获得最大距离孔径。", "改变索引会改变距离覆盖与照明几何；必须与结果一起保存和报告。"),
    "nnws_v2.enabled": H("启用 NNWS-v2", "—", "启用独立的连续时间块交叉拟合、全道集共享权重实验。", "true/false；仅限 passive + vertical；默认关闭。", "只在独立验证算例中开启，并保留 baseline 与同 N_eff 随机对照。", "v2 不修改上游 v1，但仍是研究消融，不能仅凭训练 SNR 宣称 EGF 改善。", BOOL_CHOICES),
    "nnws_v2.folds": H("连续时间折数", "fold", "把时间窗按顺序分成连续块，每个保存权重都由未见过该块的模型产生。", "整数且 ≥2，且每折必须留下足够训练窗。", "默认 5；很短记录可用 2。", "折数过多会减少每折验证样本并增加训练成本。"),
    "nnws_v2.epochs": H("每折训练轮数", "epoch", "每个交叉拟合模型的 Adam 更新次数。", "整数且 ≥1。", "quick 用 2；正式基准从 200 开始并检查损失曲线。", "增加会近似线性增加运行时间，也可能加剧代理目标过拟合。"),
    "nnws_v2.learning_rate": H("NNWS-v2 学习率", "—", "控制每折神经网络的参数更新尺度。", "有限且 >0。", "默认 0.001。", "过高可能震荡，过低可能未收敛。"),
    "nnws_v2.arrival_weight": H("到时窗外能量权重", "—", "惩罚主到时信号窗之外的对称支 RMS。", "有限且 ≥0。", "默认 0.55，并与前驱、弱对称和 N_eff 项共同报告。", "过高可能压制真实尾波或多模态能量。"),
    "nnws_v2.precursor_weight": H("前驱能量权重", "—", "单独惩罚预测最早到时之前的正、负支能量。", "有限且 ≥0。", "默认 0.20。", "速度窗错误会把真实快速相位误判为前驱。"),
    "nnws_v2.symmetry_weight": H("弱对称约束权重", "—", "在主到时窗内轻度约束正负时延支一致。", "有限且 ≥0。", "默认 0.05，明显低于 v1 的对称项。", "真实单侧照明下过强会制造表面对称而非真实 EGF。"),
    "nnws_v2.effective_count_weight": H("N_eff 惩罚权重", "—", "当有效窗数低于目标比例时惩罚权重集中。", "有限且 ≥0。", "默认 0.20。", "过低会允许少数窗支配；过高会使权重趋近线性叠加。"),
    "nnws_v2.temporal_smoothness_weight": H("时间平滑权重", "—", "约束相邻训练窗权重突变，但不跨越留出块连接。", "有限且 ≥0。", "默认 0.02。", "过高可能抹除短暂但物理有利的源方位变化。"),
    "nnws_v2.minimum_weight": H("最小单窗权重", "—", "给每个窗保留非零权重下限，避免硬剔除和零支持。", "有限且在 [0,1)。", "默认 0.05。", "提高会增加稳健性但降低筛选自由度。"),
    "nnws_v2.minimum_effective_fraction": H("最小有效窗比例", "0–1", "要求 N_eff 不低于总时间窗数的给定比例。", "有限且在 (0,1]。", "默认 0.50；同时报告绝对 N_eff。", "较高更保守，但可能限制短时强改善。"),
    "nnws_v2.seed": H("NNWS-v2 随机种子", "—", "控制各折模型初始化。", "整数且 ≥0。", "默认 0；正式结论应做多种子检验。", "改变种子可能改变权重，稳定性本身就是可靠性指标。"),
    "nnws_v2.random_control_seed": H("v2 随机对照种子", "—", "置换共享权重与时间窗的对应关系，精确保留权重分布和 N_eff。", "整数且 ≥0。", "默认 271828。", "只改变时间对应，不改变权重集合。"),
    "nnws_v2.device": H("NNWS-v2 计算设备", "—", "选择 PyTorch CPU、CUDA、MPS 或自动探测。", "auto、cpu、cuda 或 mps。", "默认 auto；跨平台复现固定 cpu。", "不同设备可能产生微小浮点差异。", (("auto", "自动选择"), ("cpu", "CPU"), ("cuda", "NVIDIA CUDA"), ("mps", "Apple Metal"))),
    "nnws_v2.maximum_cache_mb": H("v2 逐对缓存上限", "MiB", "限制逐窗、逐台站对、逐时延 NCF 缓存。", "有限且 >0。", "默认 512。", "只控制内存保护，不改变方法。"),
    "nnws_v2.pair_scope": H("v2 台站对范围", "—", "选择全部台站对或单参考接收点虚拟源道集。", "all 或 reference_receiver。", "长线阵基准用 reference_receiver。", "单参考点降低内存，但结论只代表该虚拟源。", (("all", "全部台站对"), ("reference_receiver", "单参考接收点虚拟源道集"))),
    "nnws_v2.reference_receiver_index": H("v2 参考接收点", "从 0 开始", "指定共享权重应用的虚拟源道集参考接收点。", "整数且 0 ≤ index < 接收点数。", "端点线阵通常用 0。", "改变索引会改变距离覆盖和代理 CCF。"),
    "nnws_v2.pair_ftan_receiver_index": H("v2 双台 FTAN 接收点", "从 0 开始", "指定与 v2 参考接收点组成双台 FTAN 审计站对的另一接收点。", "整数且 0 ≤ index < 接收点数，且不能等于 reference_receiver_index。", "选择接近冻结主站距的接收点；当前 Gate 0 主站距为 600 m。", "改变索引会改变台距、群/相速度分辨率和对应波形，但不得据此按方法挑选更好结果。"),
    "pair_spz.enabled": H("启用 pair-SPZ Oracle", "—", "用正演已知源位置与精确源随机流计算台站对、频率相关 stationary-phase 照明上限。", "true/false；仅限 passive + vertical；默认关闭。", "只用于正演机制验证；实测必须由 beamforming/偏振估计替代真源输入。", "这是 Oracle 上限，不得作为实际可部署方法直接报告。", BOOL_CHOICES),
    "pair_spz.phase_tolerance_rad": H("SPZ 相位容差", "rad", "把源到台站对的路径差失配换算为频率相关相位误差后的高斯核尺度。", "有限且 >0。", "默认 π/2；需做频率和速度模型敏感性。", "过宽会把离轴源纳入，过窄会减少有效源和窗数。"),
    "pair_spz.off_spz_weight": H("非 SPZ 损失权重", "—", "惩罚加权集合中不能满足 stationary-phase 路径差的源功率。", "有限且 ≥0。", "默认 0.55。", "过大可能牺牲双向平衡和有效窗数。"),
    "pair_spz.bilateral_balance_weight": H("双向照明损失权重", "—", "在整个时间窗集合上约束正、反传播 SPZ 功率平衡。", "有限且 ≥0。", "默认 0.30。", "与逐窗平衡不同，它允许互补单侧窗共同进入叠加。"),
    "pair_spz.effective_count_weight": H("SPZ N_eff 损失权重", "—", "惩罚物理权重的有效窗数低于目标。", "有限且 ≥0。", "默认 0.15，并由硬性混合步骤再次保证下限。", "过高使权重接近线性，过低会由少数照明窗支配。"),
    "pair_spz.minimum_weight": H("SPZ 最低单窗权重", "—", "保留每个时间窗的非零贡献下限。", "有限且在 [0,1)。", "默认 0.05。", "提高增强稳健性但降低源筛选自由度。"),
    "pair_spz.minimum_effective_fraction": H("SPZ 最低有效窗比例", "0–1", "要求物理权重及与 v2 组合权重的 N_eff/N 不低于该值。", "有限且在 (0,1]。", "默认 0.50。", "较高更保守，但可能限制强各向异性源场下的改善。"),
    "pair_spz.maximum_iterations": H("SPZ 优化迭代上限", "iteration", "限制 L-BFGS-B 集合照明优化步数。", "整数且 ≥1。", "默认 500。", "未收敛会记录在审计输出中，不能静默接受。"),
    "pair_spz.random_seed": H("SPZ 对照随机种子", "—", "控制物理权重和组合权重的时间置乱对照。", "整数且 ≥0。", "默认 161803。", "只改变权重—时间窗对应，不改变权重集合。"),
    "pair_spz.pair_scope": H("SPZ 台站对范围", "—", "指定路径差核使用全部台站对或单虚拟源道集。", "all 或 reference_receiver。", "与 NNWS-v2 组合时必须使用相同范围。", "改变范围会改变物理照明目标。", (("all", "全部台站对"), ("reference_receiver", "单参考接收点虚拟源道集"))),
    "pair_spz.reference_receiver_index": H("SPZ 参考接收点", "从 0 开始", "pair_scope=reference_receiver 时指定虚拟源接收点。", "整数且 0 ≤ index < 接收点数。", "端点线阵通常用 0。", "必须与 NNWS-v2 的参考点一致后才能组合。"),
    # Processing and picking.
    "processing.correlation_normalization": H("互相关归一化", "—", "在接收端互相关前控制振幅动态范围。", "spectral_whitening、one_bit 或 none。", "被动环境噪声优先 spectral_whitening；主动炮集用 none。", "whitening 平衡频带，one_bit 强烈压制振幅；两者都会改变能量权重但不改变 Green 正演。", (("spectral_whitening", "频谱白化（被动推荐）"), ("one_bit", "1-bit 归一化"), ("none", "不归一化（主动常用）"))),
    "processing.dispersion_method": H("频散成像方法", "—", "选择相移、FJ/Bessel 或 Hankel 成像核。", "phase_shift、bessel 或 hankel。", "被动径向 NCF 默认 phase_shift；主动炮集默认 hankel；疑似旁瓣时打开三方法对照。", "不同核对有限孔径、近场、体波和不规则检距的旁瓣响应不同，但不会改变输入波形。", (("phase_shift", "相移叠加"), ("bessel", "FJ / Bessel 变换"), ("hankel", "Hankel 出射波变换"))),
    "processing.theory_solver": H("理论频散求解器", "—", "选择理论 Rayleigh/Love 多模态曲线后端。", "disba 或 qedisp。", "常规模型可用 disba；低速夹层、高阶模态或 mode-kissing 审计显式选择 qedisp。", "qedisp 需要先安装本地独立可执行程序；改变理论后端不改变正演波形，只改变理论参考、SPZ 先验及相应质量指标。", (("disba", "disba / CPS-Dunkin"), ("qedisp", "QEDispInv quadratic-extrema"))),
    "processing.compare_methods": H("同时比较三种方法", "—", "在相同频率—速度网格额外计算 phase shift、FJ/Bessel 和 Hankel。", "true/false。", "正式解释旁瓣或假频时开启；常规批量计算可关闭。", "开启约增加 2–3 倍成像计算，并输出 method_comparison 图；不重复正演或互相关。", BOOL_CHOICES),
    "processing.frequency_min_hz": H("处理最低频率", "Hz", "频散图和自动拾取的最低频率。", "≥0 且 < 处理最高频率。", "让阵列孔径在该频率至少覆盖约 2 个基阶波长。", "降低会扩展低频，但孔径不足时只会产生宽能量带、旁瓣和不稳定拾取。"),
    "processing.frequency_max_hz": H("处理最高频率", "Hz", "频散图和自动拾取的最高频率。", ">最低频率、< Nyquist；被动还必须 ≤ PSD 最高频率并满足空间混叠。", "略低于源谱上限和空间混叠上限，避免边缘弱能量。", "增大可看更浅层/高阶信息，但速度网格、道距、dt 和插值精度成本都增加。"),
    "processing.frequency_step_hz": H("频率网格步长 df", "Hz", "频散图输出的频率采样间隔；它是显示/扫描网格，不是物理分辨率。", "0 表示由记录长度与 FFT 补零自动给出；显式值必须 >0。", "点击“自动填入 f/v/df/dv”得到可编辑建议；同时查看审核中的真实 1/T。", "减小会让图更密并增加计算量，但不会突破真实记录时长 T 所决定的频率分辨率。"),
    "processing.velocity_min_mps": H("相速度下限", "m/s", "频散能量扫描的最低候选相速度。", ">0 且 < 相速度上限。", "略低于预计基阶最低相速度，常比最小 Vs 低 10–30%。", "过高会截断低速基阶；过低增加无效网格和低速旁瓣搜索空间。"),
    "processing.velocity_max_mps": H("相速度上限", "m/s", "频散能量扫描的最高候选相速度。", ">相速度下限。", "覆盖目标高阶模态和最大层 Vs 附近速度。", "增大可容纳高速高阶模态，但扩大无效区域并增加计算量。"),
    "processing.velocity_step_mps": H("相速度步长", "m/s", "相速度扫描网格间隔。", ">0。", "quick 用 2–5 m/s；精细拾取常用 1–2 m/s。", "减小使图和拾取更细，但计算量与速度网格点数近似反比增加；不提高阵列真实分辨率。"),
    "processing.distance_taper_fraction": H("距离域 Tukey 窗", "0–1", "在检距/台间距两端渐消，减弱有限孔径截断。", "0≤值≤1。", "主动规则炮集可从 0 开始；被动 NCF 常用 0.3–0.6。", "增大可压低孔径旁瓣，但也削弱近/远端有效道并降低速度分辨率。"),
    "processing.zero_padding": H("时间 FFT 补零倍数", "倍", "在时间轴补零以细化离散频率采样。", ">0。", "主动常用 5，被动常用 8；超过 8–16 通常收益很小。", "增大让频率轴更平滑但不增加真实频率分辨率，并增加 FFT 内存和计算。"),
    "processing.radial_kernel_lookup_step": H("径向核查表步长", "无量纲", "Bessel/Hankel 核的快速查表步长。", "≥0；0 表示逐点精确特殊函数。", "0.002 通常兼顾速度和精度；方法核验证时用 0。", "减小提高核插值精度但变慢；phase_shift 不使用该参数。"),
    "processing.picker_smoothness": H("拾取平滑权重", "—", "动态规划对频率间速度变化的平滑惩罚。", "≥0。", "从 0.3–0.5 开始，并与理论曲线和原始能量图共同检查。", "增大可抑制跳到旁瓣，但过大会抹平真实拐点或跨模态粘连。"),
    "processing.picker_maximum_jump_mps": H("拾取最大跳变", "m/s/频点", "相邻频点允许的最大速度变化。", ">0。", "常用 20–40 m/s；速度步长较大时应相应放宽。", "减小可阻止跨旁瓣跳跃，但过小会中断陡变的真实频散支。"),
    "processing.picker_minimum_energy": H("拾取最低能量", "0–1", "允许输出自动拾取的最低归一化能量。", "0≤值≤1。", "从 0.05–0.15 开始；噪声旁瓣强时提高。", "提高会减少低信噪比误拾取，也可能让弱高阶模态出现缺口。"),
    "processing.rayleigh_modes": H("理论 Rayleigh 模态数", "阶", "独立计算并叠加显示的 Rayleigh 模态数量。", "整数且 ≥1。", "快速检查 3–4 阶；高阶研究按速度窗和频带增加。", "增加只影响理论曲线计算/显示，不会凭空在正演能量中生成高阶模态。"),
    "processing.love_modes": H("理论 Love 模态数", "阶", "独立计算并叠加显示的 Love 模态数量。", "整数且 ≥1。", "主动水平载荷或被动三分量研究用 3–6 阶。", "增加只影响理论 Love 曲线；被动纯竖直分量中它仅作模型参考。"),
}


# Backward-compatible plain summary mapping used by older tests/extensions.
PARAMETER_SPECS.update({
    "active_source.component": H("主动激发模式", "—", "旧模式是单位载荷基响应；vector是一个方向固定的真实力矢量。", "vector当前仅支持节点；DAS混合源尚未验收。", "指定力方向用vector；旧模板保持原模式。", "源方向和源台方位共同控制R/T响应；水平不等于只激发Love。", (("vertical","旧垂直载荷"),("horizontal","旧水平载荷基响应"),("both","旧双载荷基响应"),("vector","矢量力：幅值＋方向"))),
    "active_source.force_amplitude": H("矢量力幅值系数", "倍", "相对于原单位载荷的总力幅值，各分量共用同一子波。", "有限正数；只用于vector。", "1；方向改变时保持总幅值。", "线性缩放XYZ波形；不是功率比例。"),
    "active_source.force_azimuth_deg": H("水平力方位角", "deg", "全局+X到+Y为正，与接收器无关。", "有限数；只用于vector。", "0沿+X，90沿+Y。", "每个源台对单独转换到径向/横向，决定方向性。"),
    "active_source.force_inclination_deg": H("力倾角", "deg", "0为水平，+90沿原生Green的+Z，-90反向；不隐式解释为ENU/NED。", "[-90,90]。", "研究水平源用0，斜向用明确角度。", "相干H/V分量比随倾角变化，不等于接收波场H/V。"),
    "ambient_sources.polarization": H("源力极化 / 功率模式", "—", "区分旧XYZ幅值与固定总源力PSD的新模式。", "新模式需three_component、独立源、stationary_gaussian；逐事件方向尚未实现。", "旧模板保持legacy；研究比例选独立H/V或定向H/V。", "新模式总PSD为一份源PSD；旧等幅XYZ总PSD为三份，不偷偷重标定旧算例。", (("legacy_xyz","旧XYZ等谱幅值"),("independent_hv","独立H/V：水平无优选方向"),("directed_hv","独立H/V：水平指定方向"))),
    "ambient_sources.horizontal_power_fraction": H("源力水平功率占比", "0–1", "pH=(Sxx+Syy)/总源力PSD，垂直为1-pH；不是接收HVSR或辐射波能量比例。", "[0,1]；新极化模式有效。", "先明确实验比例，如0.6；无优选方向时X/Y各0.3。", "振幅按功率权重平方根缩放；pH=0仍可计算XYZ接收响应。"),
    "ambient_sources.horizontal_azimuth_deg": H("噪声水平力方向", "deg", "所有源共用的全局水平力方向；H与V时间过程独立。", "仅directed_hv；有限数。", "0沿+X，90沿+Y。", "不是源位置的方位，也不是光纤方向。逐源/逐事件方向尚未开放。"),
    "experiment.product": H("计算目标", "—", "选择理论、观测能量、节点HVSR或组合。", "理论不运行波场；HVSR必须被动XYZ节点。", "按所需输出选择，旧模板默认兼容流程。", "未请求的产品不生成；理论线不替代缺失能量。"),
    "experiment.waves": H("输出波型", "—", "选择Rayleigh、Love或两者。", "rayleigh、love、both；不改变源或观测分量。", "理论和能量分别核对可用性。", "未选择或不支持的结果页保留解释；HVSR不按波型分离。"),
    "experiment.theory_step_hz": H("理论曲线实际df", "Hz", "仅理论曲线时的真实QED求值步长，与FFT补零无关。", ">0，最多10000频点。", "按目标频带设定并加密检查。", "不是能量图扫描df；叠加模式用能量图实际频率轴。"),
    "experiment.hvsr_smoothing_b": H("HVSR KO平滑b", "—", "Konno–Ohmachi振幅谱平滑；先合并水平谱，再分别平滑H/V谱、相除。", "有限>=0；0关闭平滑。", "40是可改起点，不是收敛认证。", "越大越窄；保存该观测定义。窗使用10%Tukey，排除不足10周期频点。"),
    "experiment.hvsr_floor_ratio": H("HVSR垂直谱相对下限", "比例", "低于每台每窗最大垂直谱的此比例时标无效，不加小常数制造高峰。", "(0,1)。", "1e-6作为数值筛查，检查有效窗数；不是仪器噪声模型。", "结果为逐台站逐窗几何平均与对数散度，至少3个有效窗；不是置信区间。"),
})
PARAMETER_HELP = {path: spec.summary for path, spec in PARAMETER_SPECS.items()}


def canonical_path(path: str) -> str:
    """Normalize numeric list indexes to ``[]`` for help lookup."""
    parts = path.split(".")
    normalized: list[str] = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            if normalized:
                normalized[-1] += "[]"
            continue
        normalized.append(part)
    return ".".join(normalized)


def _fallback(path: str) -> ParameterSpec:
    top = canonical_path(path).split(".", 1)[0]
    section_name, section_text = SECTION_HELP.get(top, ("高级参数", "WaveDispX 完整 TOML 参数。"))
    raw = canonical_path(path).rsplit(".", 1)[-1]
    return H(raw, "—", section_text, "请以配置校验提示为准。", "优先从已验证模板开始。", "修改后必须重新执行输入审核。")


def parameter_spec(path: str) -> ParameterSpec:
    return PARAMETER_SPECS.get(canonical_path(path), _fallback(path))


def has_specific_help(path: str) -> bool:
    return canonical_path(path) in PARAMETER_SPECS


def group_label(path: str, raw_label: str) -> str:
    return GROUP_LABELS.get(canonical_path(path), raw_label)


def parameter_help(path: str) -> str:
    """Return the concise Chinese meaning for backwards compatibility."""
    return parameter_spec(path).summary


def parameter_choices(path: str) -> tuple[Choice, ...]:
    return parameter_spec(path).choices


def _get(mapping: dict[str, Any] | None, dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    if value is None:
        return default
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def value_guidance(path: str, value: Any, mapping: dict[str, Any] | None) -> str:
    """Return non-blocking, value-aware guidance for important coupled fields."""
    normalized = canonical_path(path)
    try:
        if normalized == "run.workers":
            return "当前请求内核自动选线程；不保证最快。" if int(value) == 0 else f"当前C++核请求 {int(value)} 个工作线程；不是全程序线程总上限。"
        if normalized == "run.radial_interpolation_step_m":
            layers = _get(mapping, "model.layers", [])
            noise_fmax = _get(mapping, "noise.maximum_frequency_hz")
            if layers and noise_fmax:
                minimum_vs = min(float(layer["vs_mps"]) for layer in layers)
                limit = minimum_vs / (20.0 * float(noise_fmax))
                relation = "满足" if float(value) <= limit else "超过"
                return f"按当前最小 Vs 和噪声上限计算，5% 波长上限为 {limit:.3f} m；当前值{relation}该上限。"
        if normalized == "ambient_sources.count":
            count = int(value)
            if _get(mapping, "ambient_sources.layout") == "annulus" and count < 192:
                return f"当前 {count} 个环形源只适合流程测试；正式收敛建议至少 192 个。"
            return f"当前设置 {count} 个噪声源。"
        if normalized in {"noise.duration_s", "noise.window_s"}:
            duration = float(_get(mapping, "noise.duration_s", 0.0))
            window = float(_get(mapping, "noise.window_s", 0.0))
            if duration > 0 and window > 0:
                return f"当前约有 {int(round(duration / window))} 个独立窗（{duration:g}/{window:g}）。"
        if normalized in {"noise.dt_s", "noise.maximum_frequency_hz"}:
            dt = float(_get(mapping, "noise.dt_s", 0.0))
            fmax = float(_get(mapping, "noise.maximum_frequency_hz", 0.0))
            if dt > 0 and fmax > 0:
                return f"当前 Nyquist={0.5/dt:.2f} Hz，最短周期约 {1/(dt*fmax):.1f} 个采样点。"
        if normalized in {"active_source.dt_s", "active_source.maximum_frequency_hz"}:
            dt = float(_get(mapping, "active_source.dt_s", 0.0))
            fmax = float(_get(mapping, "active_source.maximum_frequency_hz", 0.0))
            if dt > 0 and fmax > 0:
                return f"当前 Nyquist={0.5/dt:.2f} Hz，最短周期约 {1/(dt*fmax):.1f} 个采样点。"
        if normalized == "geometry.receivers.spacing_m":
            spacing = float(value)
            layers = _get(mapping, "model.layers", [])
            if spacing > 0 and layers:
                minimum_vs = min(float(layer["vs_mps"]) for layer in layers)
                return f"按最小 Vs={minimum_vs:g} m/s 粗估，空间 Nyquist 上限约 {minimum_vs/(2*spacing):.2f} Hz。"
        if normalized in {"das.gauge_length_m", "das.measurement"}:
            if _get(mapping, "das.geometry_mode", "uniform") == "channel_file":
                return "当前标距和方位逐道来自接收器 CSV；这里的统一标距值不会参与正演。"
            gauge = float(_get(mapping, "das.gauge_length_m", 0.0))
            velocity = float(_get(mapping, "processing.velocity_min_mps", 0.0))
            if gauge > 0 and velocity > 0:
                measurement = _get(mapping, "das.measurement", "strain_rate")
                return (
                    f"按当前速度下限粗估，有限标距第一零点约 {velocity/gauge:.2f} Hz；"
                    f"当前输出为 {measurement}。"
                )
        if normalized == "das.fiber_azimuth_deg":
            if _get(mapping, "das.geometry_mode", "uniform") == "channel_file":
                return "当前逐道光缆方位由 CSV 的端点或 fiber_azimuth_deg 列决定。"
            return f"当前光纤切向为相对 +x 轴逆时针 {float(value):g}°；面阵中所有标距均使用此方位。"
        if normalized == "das.geometry_mode":
            return (
                "逐道光缆 CSV 模式：接收器必须为 file 布局，x_m,y_m 是通道中心；"
                "每行再给两个端点，或给 gauge_length_m,fiber_azimuth_deg。"
                if str(value) == "channel_file"
                else "统一模式：所有通道使用 das.gauge_length_m 和 das.fiber_azimuth_deg。"
            )
        if normalized in {
            "geometry.receivers.start_x_m",
            "geometry.receivers.stop_x_m",
            "geometry.receivers.spacing_m",
            "geometry.receivers.y_m",
        } and _get(mapping, "geometry.receivers.layout") == "line":
            start = float(_get(mapping, "geometry.receivers.start_x_m", 0.0))
            stop = float(_get(mapping, "geometry.receivers.stop_x_m", 0.0))
            spacing = float(_get(mapping, "geometry.receivers.spacing_m", 0.0))
            if stop > start and spacing > 0:
                return f"当前线阵孔径 {stop-start:g} m，共约 {int(round((stop-start)/spacing))+1} 道。"
        if normalized in {
            "geometry.receivers.start_x_m",
            "geometry.receivers.stop_x_m",
            "geometry.receivers.spacing_x_m",
            "geometry.receivers.start_y_m",
            "geometry.receivers.stop_y_m",
            "geometry.receivers.spacing_y_m",
        } and _get(mapping, "geometry.receivers.layout") in {"grid", "array"}:
            start_x = float(_get(mapping, "geometry.receivers.start_x_m", 0.0))
            stop_x = float(_get(mapping, "geometry.receivers.stop_x_m", 0.0))
            spacing_x = float(_get(mapping, "geometry.receivers.spacing_x_m", 0.0))
            start_y = float(_get(mapping, "geometry.receivers.start_y_m", 0.0))
            stop_y = float(_get(mapping, "geometry.receivers.stop_y_m", 0.0))
            spacing_y = float(_get(mapping, "geometry.receivers.spacing_y_m", 0.0))
            if stop_x > start_x and stop_y > start_y and spacing_x > 0 and spacing_y > 0:
                count_x = int(round((stop_x - start_x) / spacing_x)) + 1
                count_y = int(round((stop_y - start_y) / spacing_y)) + 1
                return f"当前为 {count_x}×{count_y} 面阵，共 {count_x*count_y} 个接收器。"
        if normalized == "geometry.receivers.distance_bin_m":
            return f"当前二维台站对按 {float(value):g} m 宽度归入径向距离箱。"
        if normalized in {"ambient_sources.inner_radius_m", "ambient_sources.outer_radius_m"}:
            inner = float(_get(mapping, "ambient_sources.inner_radius_m", 0.0))
            outer = float(_get(mapping, "ambient_sources.outer_radius_m", 0.0))
            if outer > inner:
                return f"当前源环半径 {inner:g}–{outer:g} m，径向宽度 {outer-inner:g} m。"
        if normalized in {"processing.frequency_max_hz", "processing.frequency_min_hz"}:
            fmin = float(_get(mapping, "processing.frequency_min_hz", 0.0))
            fmax = float(_get(mapping, "processing.frequency_max_hz", 0.0))
            source_max = _get(mapping, "noise.maximum_frequency_hz", _get(mapping, "active_source.maximum_frequency_hz"))
            suffix = f"；源谱/正演上限 {float(source_max):g} Hz" if source_max is not None else ""
            return f"当前处理带宽 {fmin:g}–{fmax:g} Hz{suffix}。"
        if normalized in {"processing.velocity_min_mps", "processing.velocity_max_mps", "processing.velocity_step_mps"}:
            vmin = float(_get(mapping, "processing.velocity_min_mps", 0.0))
            vmax = float(_get(mapping, "processing.velocity_max_mps", 0.0))
            step = float(_get(mapping, "processing.velocity_step_mps", 1.0))
            if step > 0:
                return f"当前速度窗 {vmin:g}–{vmax:g} m/s，共约 {int(math.floor((vmax-vmin)/step))+1} 个网格点。"
        if normalized == "processing.compare_methods":
            return "当前会输出三种方法对照。" if bool(value) else "当前只计算所选主方法。"
        if normalized == "processing.distance_taper_fraction":
            return f"当前 Tukey 距离窗比例为 {100*float(value):.0f}%。"
        if normalized in {"model.layers[].vp_mps", "model.layers[].vs_mps"}:
            # Resolve the actual indexed layer from paths such as model.layers.[0].vp_mps.
            tokens = path.split(".")
            index = next((int(token[1:-1]) for token in tokens if token.startswith("[") and token.endswith("]")), None)
            layers = _get(mapping, "model.layers", [])
            if index is not None and index < len(layers):
                vp = float(layers[index]["vp_mps"])
                vs = float(layers[index]["vs_mps"])
                return f"当前第 {index+1} 层 Vp/Vs={vp/vs:.3f}。"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return "当前值将在输入审核中与关联参数一起检查。"
    return "修改后请重新执行输入审核，检查模型、Geometry 和采样联动。"


def parameter_detail_html(
    path: str,
    value: Any,
    mapping: dict[str, Any] | None = None,
    *,
    compact: bool = False,
) -> str:
    """Build rich HTML shared by hover tips and the persistent detail card."""
    spec = parameter_spec(path)
    current = escape(str(value))
    choices = ""
    if spec.choices:
        choices = "<br><b>可选值：</b>" + "；".join(
            f"<code>{escape(raw)}</code>＝{escape(label)}" for raw, label in spec.choices
        )
    dynamic = escape(value_guidance(path, value, mapping))
    width = 430 if compact else 760
    return (
        f'<div style="width:{width}px;white-space:normal">'
        f'<div style="font-size:15px;font-weight:700;color:#173b61">{escape(spec.label)}</div>'
        f'<div style="color:#647586"><code>{escape(path)}</code>　单位：{escape(spec.unit)}　当前值：<b>{current}</b></div>'
        f'<hr><b>它控制什么：</b>{escape(spec.summary)}<br>'
        f'<b>允许/约束：</b>{escape(spec.constraint)}{choices}<br>'
        f'<b>推荐起点：</b>{escape(spec.recommended)}<br>'
        f'<b>调大/调小的影响：</b>{escape(spec.effect)}<br>'
        f'<b style="color:#1769c2">当前设置：</b>{dynamic}'
        "</div>"
    )
