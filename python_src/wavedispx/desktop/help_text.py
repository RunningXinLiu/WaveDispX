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
    "active_source": ("主动源与子波", "炮点、载荷分量、Ricker 子波和主动记录采样。"),
    "ambient_sources": ("环境噪声源", "被动源分量、空间分布、数量和空间随机种子。"),
    "noise": ("噪声时间模型与 PSD", "有限记录时长、独立窗、采样、功率谱和时间随机种子。"),
    "processing": ("频散处理与拾取", "互相关归一化、成像核、频率—速度网格、拾取和理论模态。"),
}

GROUP_LABELS = {
    "run": "运行、输出与性能",
    "model": "速度—密度模型",
    "model.layers": "模型分层",
    "geometry": "观测系统 Geometry",
    "geometry.receivers": "接收器",
    "active_source": "主动源与 Ricker 子波",
    "ambient_sources": "环境噪声源空间分布",
    "noise": "噪声时间模型与 PSD",
    "processing": "互相关、频散与拾取",
}


PARAMETER_SPECS: dict[str, ParameterSpec] = {
    # Run and performance.
    "run.name": H("算例名称", "—", "结果和图件使用的算例标识。", "不能为空。", "简短英文、数字和下划线，例如 passive_site01。", "只改变名称和元数据；不会改变数值结果。"),
    "run.mode": H("模拟模式", "—", "选择主动炮集或真实有限记录环境噪声流程。", "只能是 active 或 passive；改变模式时应载入同模式模板。", "主动炮集选 active；背景噪声互相关选 passive。", "切换模式会改变必需的源参数、观测阶段和默认频散方法。", (("active", "主动源炮集"), ("passive", "被动环境噪声"))),
    "run.output_directory": H("输出目录", "路径", "保存 HDF5、CSV、summary、manifest 和可复现 input.toml。", "必须是当前用户可创建或写入的目录。", "每个算例使用独立目录，避免结果互相覆盖。", "不改变计算；同名文件会在保存阶段更新。"),
    "run.workers": H("CPU 并行线程数", "线程", "C++ 正演使用的 CPU 工作线程数。", "整数且 ≥0；0 表示自动。", "一般填 0；需要限制共享机器负载时再指定 1–8。", "增大可缩短正演时间，但超过物理核心后收益很小且可能抢占内存带宽。"),
    "run.frequency_block_size": H("频率分块大小", "频点/块", "C++ 正演一次处理的频率数量。", "整数且 ≥1。", "常用 8–32；内存紧张用 4–8。", "增大可减少调度开销并提高速度，但增加峰值内存。"),
    "run.source_block_size": H("噪声源分块大小", "源/块", "被动正演和混合记录一次处理的源数。", "整数且 ≥1。", "常用 8–32；主动模式基本不受它影响。", "增大通常更快但占用更多内存；不会改变物理结果。"),
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
    # Active source.
    "active_source.x_m": H("炮点 x", "m", "本次主动震源的 x 坐标。", "有限数，且不能与接收器完全重合。", "与接收器使用同一坐标原点；多炮时从 Shot CSV 明确选择一炮。", "改变炮点会改变所有炮检距、近场比例和有效径向采样。"),
    "active_source.y_m": H("炮点 y", "m", "本次主动震源的 y 坐标。", "有限数，且不能与接收器完全重合。", "线阵常取 0；偏移炮点应填写真实横向偏移。", "横向偏移改变真实二维炮检距，可能改善不规则检距采样。"),
    "active_source.component": H("主动载荷分量", "—", "决定计算 Rayleigh、Love 或二者。", "vertical、horizontal 或 both。", "未知目标时选 both；只研究竖直 Rayleigh 可选 vertical 以节省时间。", "both 计算五分量 Green 响应，信息最完整但运行和输出更大。", (("vertical", "竖直载荷 / Rayleigh"), ("horizontal", "水平横向载荷 / Love"), ("both", "两类载荷都计算"))),
    "active_source.wavelet": H("主动源子波", "—", "主动震源时间函数。", "当前仅支持 ricker。", "保持 ricker，通过主频和最高频率调节频带。", "子波谱决定炮集主要能量频带，不改变层状 Green 函数本身。", (("ricker", "Ricker 子波"),)),
    "active_source.dominant_frequency_hz": H("Ricker 主频", "Hz", "Ricker 子波功率谱的主要能量中心。", ">0，且不能高于正演最高频率。", "让主频落在目标频散频带中部。", "增大主频增强高频/浅层分辨率，但更易受道距混叠、衰减和网格采样限制。"),
    "active_source.maximum_frequency_hz": H("主动正演最高频率", "Hz", "波场正演保留的最高频率。", "≥ Ricker 主频，且严格 < 1/(2dt)。", "略高于处理上限并覆盖子波有效尾部，通常约主频的 2–3 倍。", "增大扩大高频带宽和计算频点数；过高会带来时间/空间混叠。"),
    "active_source.dt_s": H("主动时间采样间隔", "s", "炮集时间序列采样间隔。", ">0，且最高频率 < 1/(2dt)；最短周期建议至少 8 点。", "按 dt ≤ 1/(8 fmax) 选取。", "减小 dt 提高时间采样并扩大 Nyquist，但增加样点数、内存和 FFT 成本。"),
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
    # Noise time process and PSD.
    "noise.temporal_model": H("噪声时间模型", "—", "生成连续平稳噪声或有离散激发时刻的随机脉冲。", "stationary_gaussian 或 poisson_ricker。", "一般环境噪声用 stationary_gaussian；车辆/敲击事件演示用 poisson_ricker。", "Poisson 模型会生成事件目录；平稳模型没有离散激发时刻表。", (("stationary_gaussian", "平稳高斯连续噪声"), ("poisson_ricker", "Poisson 随机 Ricker 事件"))),
    "noise.seed": H("噪声时间随机种子", "—", "控制源时间函数、事件数、时刻、极性和权重。", "整数且 ≥0。", "固定种子比较模型/阵列差异；收敛研究可用多个种子做重复试验。", "改变它不改变源位置，但会改变有限记录噪声实现和 NCF 随机误差。"),
    "noise.duration_s": H("记录总时长", "s", "所有独立噪声窗累计的有限记录长度。", ">0，且 ≥ 独立窗长。", "quick 60 s；可用测试 600 s；正式收敛常用 1800 s 或更长。", "增加时长增加独立窗数并降低互相关方差，通常是被动频散连续收敛的关键；计算量近似线性增长。"),
    "noise.dt_s": H("噪声采样间隔", "s", "混合记录和 NCF 的时间采样间隔。", ">0，PSD 上限和处理上限必须严格 < 1/(2dt)。", "按 dt ≤ 1/(8 fmax) 保证最短周期至少约 8 点。", "减小 dt 提高高频时间采样，但成倍增加每窗样点、FFT、内存和互相关成本。"),
    "noise.window_s": H("独立窗长", "s", "每次独立合成并累积互相关的时间窗长度。", ">0、≤总时长，且必须大于最大台间传播时延。", "先满足审核给出的 required lag，再令总时长包含足够多独立窗。", "过短会截断远台传播；过长会减少独立窗数并增大单窗内存。"),
    "noise.overlap": H("窗重叠率", "—", "相邻合成/互相关窗的重叠比例。", "当前实现必须固定为 0。", "保持 0；每个窗独立生成并由流式互相关器叠加。", "当前版本修改为非零会被校验拒绝。"),
    "noise.psd_shape": H("噪声 PSD 形状", "—", "定义每个背景源的期望功率谱。", "ricker、flat_band 或 file。", "窄带平滑源用 ricker；带内均匀照明用 flat_band；实测谱用 file。", "PSD 直接控制可恢复频带；频带外没有源能量，补零不能恢复。", (("ricker", "Ricker 型平滑谱"), ("flat_band", "带内平坦谱"), ("file", "CSV 外部功率谱"))),
    "noise.dominant_frequency_hz": H("PSD 峰值频率", "Hz", "Ricker 型噪声 PSD 的能量峰值。", "> 最低频率且 ≤ 最高频率。", "放在目标处理频带中部；flat_band/file 时仅保留为配置参考。", "增大把主要源能量推向高频和浅层，但更易受衰减和空间混叠影响。"),
    "noise.minimum_frequency_hz": H("PSD 最低频率", "Hz", "噪声源具有有效能量的低频边界。", "≥0 且 < 峰值频率。", "应不高于处理最低频率，并考虑阵列孔径能否分辨该低频。", "降低它不会自动提升低频分辨率；若孔径不足只会增加不稳定低频能量。"),
    "noise.maximum_frequency_hz": H("PSD 最高频率", "Hz", "噪声源有效能量的高频边界。", "≥峰值，严格 < 1/(2dt)，且处理最高频率不能超过它。", "略高于处理上限，同时满足道距空间混叠和 Green 插值约束。", "增大扩展潜在高频，但显著收紧 dt、道距和径向插值要求。"),
    "noise.psd_file": H("外部 PSD 文件", "CSV", "从文件读取频率—功率谱。", "psd_shape=file 时必填；UTF-8，frequency_hz 严格递增、power 非负。", "使用“PSD CSV”按钮选择并先查看输入审核中的谱形。", "文件谱将替代内置 Ricker/平坦形状；错误频率单位会直接造成频带错误。"),
    "noise.source_amplitude": H("单源相对振幅", "相对量", "每个噪声源时间函数的统一振幅尺度。", "有限正数。", "通常保持 1；只在振幅动态范围试验中修改。", "线性缩放混合记录幅值；归一化频散能量通常不变，但数值动态范围会变化。"),
    "noise.independent_sources": H("各噪声源相互独立", "—", "是否为每个源生成独立随机时间函数。", "true/false。", "真实 Green 恢复和常规被动处理必须选 true；false 仅作相干源敏感性试验。", "关闭会让不同源相干，产生强交叉项和假频，不再满足标准噪声互相关假设。", BOOL_CHOICES),
    "noise.event_rate_hz": H("Poisson 事件率", "事件/(s·源)", "Poisson-Ricker 模型中每个源每秒平均事件数。", ">0；仅 poisson_ricker 模型实际生效。", "从 0.1–0.5 开始，使事件既有重叠又不过度连续。", "增大会使记录更连续、事件数和计算负载增加；过低导致有限窗内大量空源。"),
    # Processing and picking.
    "processing.correlation_normalization": H("互相关归一化", "—", "在接收端互相关前控制振幅动态范围。", "spectral_whitening、one_bit 或 none。", "被动环境噪声优先 spectral_whitening；主动炮集用 none。", "whitening 平衡频带，one_bit 强烈压制振幅；两者都会改变能量权重但不改变 Green 正演。", (("spectral_whitening", "频谱白化（被动推荐）"), ("one_bit", "1-bit 归一化"), ("none", "不归一化（主动常用）"))),
    "processing.dispersion_method": H("频散成像方法", "—", "选择相移、FJ/Bessel 或 Hankel 成像核。", "phase_shift、bessel 或 hankel。", "被动径向 NCF 默认 phase_shift；主动炮集默认 hankel；疑似旁瓣时打开三方法对照。", "不同核对有限孔径、近场、体波和不规则检距的旁瓣响应不同，但不会改变输入波形。", (("phase_shift", "相移叠加"), ("bessel", "FJ / Bessel 变换"), ("hankel", "Hankel 出射波变换"))),
    "processing.compare_methods": H("同时比较三种方法", "—", "在相同频率—速度网格额外计算 phase shift、FJ/Bessel 和 Hankel。", "true/false。", "正式解释旁瓣或假频时开启；常规批量计算可关闭。", "开启约增加 2–3 倍成像计算，并输出 method_comparison 图；不重复正演或互相关。", BOOL_CHOICES),
    "processing.frequency_min_hz": H("处理最低频率", "Hz", "频散图和自动拾取的最低频率。", "≥0 且 < 处理最高频率。", "让阵列孔径在该频率至少覆盖约 2 个基阶波长。", "降低会扩展低频，但孔径不足时只会产生宽能量带、旁瓣和不稳定拾取。"),
    "processing.frequency_max_hz": H("处理最高频率", "Hz", "频散图和自动拾取的最高频率。", ">最低频率、< Nyquist；被动还必须 ≤ PSD 最高频率并满足空间混叠。", "略低于源谱上限和空间混叠上限，避免边缘弱能量。", "增大可看更浅层/高阶信息，但速度网格、道距、dt 和插值精度成本都增加。"),
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
            return "当前为自动使用可用核心。" if int(value) == 0 else f"当前最多使用 {int(value)} 个 CPU 线程。"
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
