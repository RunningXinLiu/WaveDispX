"""Reader-facing helpers for the active/passive tutorial notebooks.

The notebooks keep every user-editable value in one ``PARAMETERS`` mapping.
This module only turns that mapping into the same validated ``WaveDispXWorkflow``
used by Streamlit and the CLI; it contains no modelling equations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .app_service import (
    CaseInputError,
    CaseSession,
    explain_input_error,
    load_template_mapping,
    parse_shot_csv,
)
from .config import RunConfig
from .preflight import PreflightReview
from .workflow import WaveDispXWorkflow


@dataclass
class NotebookCase:
    """One prepared tutorial case and its retained interactive session."""

    workflow: WaveDispXWorkflow
    config: RunConfig
    invalidated: list[str]
    input_origin: str
    session: CaseSession | None = None


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_optional_file(
    project_root: Path, table: dict[str, Any], key: str
) -> None:
    value = table.get(key)
    if value in (None, ""):
        table.pop(key, None)
    else:
        table[key] = str(_project_path(project_root, value))


def prepare_notebook_case(
    project_root: str | Path,
    mode: Literal["active", "passive"],
    parameters: dict[str, Any],
) -> NotebookCase:
    """Build a workflow from a single visible notebook parameter mapping.

    When ``input_toml`` is set, the TOML is the sole source of case parameters;
    the remaining cell values are intentionally ignored.  Otherwise the
    selected quick template supplies only schema-safe defaults, and every
    user-facing table in ``parameters`` is applied explicitly.
    """
    root = Path(project_root).expanduser().resolve()
    values = deepcopy(parameters)
    input_toml = values.get("input_toml")
    if input_toml not in (None, ""):
        path = _project_path(root, input_toml)
        workflow = WaveDispXWorkflow.from_toml(path)
        if workflow.config.mode != mode:
            raise CaseInputError(
                f"Notebook 模式是 {mode!r}，但 TOML run.mode 是 "
                f"{workflow.config.mode!r}"
            )
        invalidated = (
            ["forward", "analysis"]
            if mode == "active"
            else ["forward", "observations", "analysis"]
        )
        return NotebookCase(
            workflow,
            workflow.config,
            invalidated,
            f"TOML: {path}",
        )

    template_value = values.get("template", f"examples/{mode}_quick.toml")
    template_path = _project_path(root, template_value)
    mapping = load_template_mapping(template_path)
    run = deepcopy(values["run"])
    run["mode"] = mode
    if run.get("output_directory") not in (None, ""):
        run["output_directory"] = str(
            _project_path(root, run["output_directory"])
        )
    mapping["run"] = run
    mapping["model"] = {"layers": deepcopy(values["model_layers"])}

    receivers = deepcopy(values["receivers"])
    _resolve_optional_file(root, receivers, "coordinates_file")
    mapping["geometry"] = {"receivers": receivers}
    mapping["das"] = deepcopy(values.get("das", mapping.get("das", {})))
    mapping["processing"] = deepcopy(values["processing"])

    if mode == "active":
        active_source = deepcopy(values["active_source"])
        shot_csv = values.get("shot_csv")
        if shot_csv not in (None, ""):
            shot_path = _project_path(root, shot_csv)
            if not shot_path.is_file():
                raise CaseInputError(f"炮点 Shot CSV 不存在：{shot_path}")
            shots = parse_shot_csv(shot_path.read_bytes())
            requested_id = values.get("shot_id")
            if requested_id in (None, ""):
                if len(shots) != 1:
                    available = ", ".join(shot.shot_id for shot in shots)
                    raise CaseInputError(
                        "Shot CSV 含多炮，请在参数总控 Cell 设置 shot_id；"
                        f"可选：{available}"
                    )
                selected = shots[0]
            else:
                selected = next(
                    (shot for shot in shots if shot.shot_id == str(requested_id)),
                    None,
                )
                if selected is None:
                    available = ", ".join(shot.shot_id for shot in shots)
                    raise CaseInputError(
                        f"Shot CSV 中找不到 shot_id={requested_id!r}；可选：{available}"
                    )
            active_source["x_m"] = selected.x_m
            active_source["y_m"] = selected.y_m
        mapping["active_source"] = active_source
        mapping.pop("ambient_sources", None)
        mapping.pop("noise", None)
    else:
        ambient = deepcopy(values["ambient_sources"])
        _resolve_optional_file(root, ambient, "coordinates_file")
        noise = deepcopy(values["noise"])
        _resolve_optional_file(root, noise, "psd_file")
        mapping["ambient_sources"] = ambient
        mapping["noise"] = noise
        mapping.pop("active_source", None)

    session = CaseSession(root)
    config, invalidated = session.apply_mapping(mapping)
    assert session.workflow is not None
    return NotebookCase(
        session.workflow,
        config,
        invalidated,
        f"参数 Cell + template: {template_path}",
        session,
    )


_COMMON_REFERENCE = (
    ("run.name", "—", "非空", "算例名；写入图题、manifest 和输出 TOML。"),
    ("run.output_directory", "路径", "可写目录", "HDF5、CSV、图和配置的输出目录。"),
    ("run.workers", "线程", "0=自动，≥0", "CPU 并行线程数；正式计算通常用 0。"),
    ("run.frequency_block_size", "频点", "整数≥1", "C++ 每批频率数；大值更快但更占内存。"),
    ("run.source_block_size", "源", "整数≥1", "被动源分块；主动模式影响很小。"),
    ("run.radial_interpolation_step_m", "m", ">0，且≤最短波长5%", "径向 Green 插值精度；过大会被 required 检查阻止。"),
    ("run.store_source_responses", "—", "true/false", "是否保存逐噪声源传递函数；通常关闭以节省内存。"),
    ("model_layers[].thickness_m", "m", "有限层>0；末层=inf", "层厚；最后一层必须是半空间。"),
    ("model_layers[].vp_mps", "m/s", "Vp>Vs>0", "P 波速度；影响阻抗与 Rayleigh 椭圆极化。"),
    ("model_layers[].vs_mps", "m/s", ">0", "S 波速度；控制面波相速度和最短空间波长。"),
    ("model_layers[].density_kgm3", "kg/m³", ">0", "密度；进入层状介质刚度和阻抗。"),
    ("model_layers[].qp / qs", "—", ">0 或 inf", "品质因子；inf 表示无本征衰减。"),
    ("receivers.layout", "—", "line/grid/file", "线阵、规则二维面阵或任意二维 CSV。"),
    ("receivers.start/stop/spacing", "m", "跨度可被间距整除", "line 使用 x 与 spacing_m；grid 分别使用 x/y。"),
    ("receivers.coordinates_file", "CSV", "file 时必填", "表头 x_m,y_m，至少两个不重复有限坐标。"),
    ("receivers.distance_bin_m", "m", "被动 grid/file 必填", "二维台站对转为径向 NCF 道集时的距离箱宽。"),
    ("processing.dispersion_method", "—", "phase_shift/bessel/hankel", "频散成像核；被动默认 phase_shift，主动默认 hankel。"),
    ("processing.theory_solver", "—", "disba/qedisp", "理论多模态求解器；低速层/mode-kissing 审计可显式用 qedisp。"),
    ("processing.compare_methods", "—", "true/false", "同时输出三种成像核的同网格对照；默认关闭以节省时间。"),
    ("processing.frequency_min/max_hz", "Hz", "0≤fmin<fmax<Nyquist", "频散显示与拾取频带；被动上限不能超过 PSD 上限。"),
    ("processing.velocity_min/max_mps", "m/s", "0<vmin<vmax", "相速度扫描范围，必须覆盖预期模态。"),
    ("processing.velocity_step_mps", "m/s", ">0", "速度网格步长；更小更细但计算和内存增加。"),
    ("processing.frequency_step_hz", "Hz", "0=自动或>0", "频散图显示/扫描频率步长；不改变真实记录长度决定的 1/T 物理分辨率。"),
    ("processing.distance_taper_fraction", "—", "0–1", "距离域 Tukey taper；减弱有限孔径截断旁瓣。"),
    ("processing.correlation_normalization", "—", "spectral_whitening/one_bit/none", "NCF 归一化；主动通常 none，被动默认 whitening。"),
    ("processing.zero_padding", "倍", ">0", "时间 FFT 补零，只细化频率采样，不提高真实分辨率。"),
    ("processing.radial_kernel_lookup_step", "—", "≥0", "Bessel/Hankel 核查表步长；0 为逐点精确函数。"),
    ("processing.picker_smoothness", "—", "≥0", "自动拾取连续性惩罚；过大可能压制真实快速变化。"),
    ("processing.picker_maximum_jump_mps", "m/s", ">0", "相邻频点允许的最大拾取跳变。"),
    ("processing.picker_minimum_energy", "—", "0–1", "低于该归一化能量的频点不输出拾取。"),
    ("processing.rayleigh_modes/love_modes", "阶", "整数≥1", "独立理论频散计算和叠加的模态数。"),
)

_DAS_REFERENCE = (
    ("das.enabled", "—", "true/false", "开启后在每个道中心建立两个标距端点；端点 Green 位移先做轴向差，再进入记录与互相关。"),
    ("das.geometry_mode", "—", "uniform/channel_file", "uniform 使用统一标距/方位；channel_file 从 receiver CSV 逐道读取端点或标距/方位。"),
    ("das.gauge_length_m", "m", ">0", "uniform 模式物理标距 Lg；channel_file 模式由 CSV 每行定义。有限标距响应为 sinc(k_t Lg/2)。"),
    ("das.measurement", "—", "strain/strain_rate", "轴向应变，或按 NumPy rFFT 约定乘 +i2πf 后的应变率。"),
    ("das.fiber_azimuth_deg", "deg", "有限角度", "uniform 模式相对 +x 轴逆时针；channel_file 模式由 CSV 逐道定义。"),
)

_ACTIVE_REFERENCE = (
    ("shot_csv / shot_id", "CSV / —", "可选；多炮必须选ID", "从 shot_id,x_m,y_m 文件选择一炮；设置后覆盖手填 x/y。"),
    ("active_source.x_m / y_m", "m", "有限数值", "单炮坐标；与接收器的平面距离进入 Green 响应。"),
    ("active_source.component", "—", "vertical/horizontal/both", "Rayleigh、Love 或同时计算。"),
    ("active_source.wavelet", "—", "ricker/unit_impulse", "主动震源时间函数；单位脉冲用于理想宽频校准。"),
    ("active_source.dominant_frequency_hz", "Hz", ">0", "Ricker 主频；单位脉冲时为时窗参考频率。"),
    ("active_source.maximum_frequency_hz", "Hz", "≥主频且<Nyquist", "波场正演频率上限。"),
    ("active_source.dt_s", "s", ">0", "时间采样；Nyquist=1/(2dt)，最短周期建议≥8点。"),
    ("active_source.record_duration_s", "s", "0=自动或>0", "保存炮集并用于成像的真实时长；0 按群速度、最大炮检距和最低处理频率自动设计。"),
    ("noise.maximum_lag_s", "s", "0=自动或 0<lag<window", "被动 NCF 保存并成像的最大正时延；0 按群速度与阵列孔径自动设计。"),
)

_PASSIVE_REFERENCE = (
    ("ambient_sources.component", "—", "vertical/three_component", "竖直快速路径，或物理 Fx/Fy/Fz 经五分量 Green 张量传播并输出 RR/TT/ZZ NCF。"),
    ("ambient_sources.layout", "—", "annulus/line/file", "源环、道路源线或显式噪声源坐标文件。"),
    ("ambient_sources.count", "源", "整数≥1", "参与混合的噪声源数；源环收敛通常建议 192–384。"),
    ("ambient_sources.seed", "—", "整数≥0", "只控制噪声源空间分布，可复现 geometry。"),
    ("ambient_sources.angular_sampling", "—", "stratified/uniform/random", "源环方位采样；stratified 通常最稳。"),
    ("ambient_sources.inner/outer_radius_m", "m", "0≤内半径<外半径", "annulus 源环范围，半径按面积均匀抽样。"),
    ("ambient_sources.x_min/x_max/y_m", "m", "xmax>xmin", "line 源线范围和固定 y。"),
    ("ambient_sources.coordinates_file", "CSV", "file 时必填", "显式 noise source 的 x_m,y_m；count 应与文件行数一致。"),
    ("noise.temporal_model", "—", "stationary_gaussian/poisson_ricker", "平稳高斯连续噪声或具有离散激发时刻的 Poisson 事件。"),
    ("noise.seed", "—", "整数≥0", "控制源时间函数、事件数、时刻、极性和振幅。"),
    ("noise.duration_s", "s", ">0", "有限记录总时长；收敛常用 600–1800 s。"),
    ("noise.window_s", "s", "0<窗长≤总时长", "独立合成/互相关窗；还必须覆盖最大台间传播时延。"),
    ("noise.dt_s", "s", ">0", "时间采样；PSD 最高频率必须小于 1/(2dt)。"),
    ("noise.psd_shape", "—", "ricker/flat_band/file", "噪声功率谱形状；file 读取 frequency_hz,power。"),
    ("noise.minimum/dominant/maximum_frequency_hz", "Hz", "fmin<fdom≤fmax<Nyquist", "噪声能量频带及谱峰。"),
    ("noise.psd_file", "CSV", "PSD=file 时必填", "频率递增、功率非负的外部 PSD。"),
    ("noise.source_amplitude", "相对量", ">0", "单源振幅尺度；影响记录幅值，不改变归一化频散定义。"),
    ("noise.independent_sources", "—", "通常 true", "各源独立是接收端 Green 恢复的标准假设；false 仅作相干敏感性实验。"),
    ("noise.event_rate_hz", "1/s/源", ">0", "poisson_ricker 每个源的平均事件率。"),
    ("noise.overlap", "—", "当前固定 0", "独立合成窗不重叠，由流式互相关器直接叠加。"),
)


def parameter_reference_markdown(mode: Literal["active", "passive"]) -> str:
    """Return a complete Chinese parameter table for display in Jupyter."""
    specific = _ACTIVE_REFERENCE if mode == "active" else _PASSIVE_REFERENCE
    rows = (*_COMMON_REFERENCE, *_DAS_REFERENCE, *specific)
    lines = [
        "### 参数详细说明",
        "",
        "| 参数路径 | 单位 | 必填/约束 | 作用与调整影响 |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        f"| `{path}` | {unit} | {constraint} | {description} |"
        for path, unit, constraint, description in rows
    )
    return "\n".join(lines)


def required_failure_markdown(review: PreflightReview) -> str:
    """Explain every blocking preflight failure in notebook-friendly Chinese."""
    failures = [
        check
        for check in review.checks
        if check.level == "required" and not check.passed
    ]
    if not failures:
        return "✅ **所有 required 检查通过，可以进入正演。**"
    lines = [
        f"❌ **{len(failures)} 个 required 检查失败；正演保持锁定。**",
        "",
    ]
    for index, check in enumerate(failures, start=1):
        lines.extend(
            (
                f"{index}. **{check.name}**",
                f"   - 参数位置：`{check.parameter_path}`",
                f"   - 当前值：{check.value}",
                f"   - 必须满足：{check.criterion}",
                f"   - 怎么改：{check.guidance}",
            )
        )
    return "\n".join(lines)


def input_error_markdown(error: Exception) -> str:
    """Format parser/configuration errors like the Streamlit error cards."""
    issue = explain_input_error(error)
    return "\n".join(
        (
            f"❌ **参数校验失败｜`{issue.field_path}`**",
            "",
            f"- **位置：** {issue.location}",
            f"- **问题：** {issue.problem}",
            f"- **怎么改：** {issue.suggestion}",
            f"- **原始技术信息：** `{issue.technical_message}`",
        )
    )
