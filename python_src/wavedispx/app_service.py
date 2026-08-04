"""UI-neutral case/session services shared by Streamlit and notebooks.

This module deliberately does not import Streamlit.  It owns uploaded-file
staging, TOML parsing/serialization and safe reuse of :class:`WaveDispXWorkflow` so a
future desktop UI can call the same application boundary.
"""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Literal

from .config import RunConfig, config_from_mapping
from .workflow import WaveDispXWorkflow


class CaseInputError(ValueError):
    """Raised when an uploaded or edited case cannot be materialized safely."""


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ShotLocation:
    """One selectable active-shot row loaded by the UI."""

    shot_id: str
    x_m: float
    y_m: float


@dataclass(frozen=True)
class InputIssue:
    """Actionable, UI-neutral explanation of one rejected input.

    ``technical_message`` preserves the exact core error for reproducibility,
    while the other fields tell a non-programmer where to edit the case and
    what change is expected.
    """

    location: str
    field_path: str
    problem: str
    suggestion: str
    technical_message: str


def explain_input_error(error: Exception) -> InputIssue:
    """Translate parser/configuration failures into an actionable UI issue."""
    message = str(error).strip() or type(error).__name__
    lower = message.lower()

    rules = (
        (
            ("toml",),
            "建立算例 → 载入完整 TOML",
            "TOML 文件",
            "配置文件无法读取或缺少必要表。",
            "使用 UTF-8 TOML；至少包含 [run]，也可先载入一个内置模板再修改。",
        ),
        (
            ("模型第", "model.layers"),
            "建立算例 → 速度—密度模型",
            "[[model.layers]]",
            "模型层参数缺失、不是有效数值，或层间约束不成立。",
            "逐行检查厚度、Vp、Vs、密度、Qp、Qs；要求 Vp > Vs > 0，最后一层厚度填 inf。",
        ),
        (
            ("last model layer", "final model layer", "half-space"),
            "建立算例 → 速度—密度模型",
            "末层 thickness_m",
            "末层没有定义为半空间。",
            "把最后一行“厚度 m”改为 inf；前面的有限层不能填 inf。",
        ),
        (
            ("炮点 csv", "shot csv", "shot_id"),
            "建立算例 → 主动源与子波 → 炮点输入",
            "Shot CSV",
            "炮点文件缺失、格式错误或炮号不唯一。",
            "上传 UTF-8 CSV，表头使用 shot_id,x_m,y_m；x_m/y_m 必须为有限数字，shot_id 不能重复。",
        ),
        (
            ("请选择接收器", "receiver coordinate", "geometry.receivers.coordinates_file", "duplicate x_m,y_m"),
            "建立算例 → 观测系统 Geometry",
            "接收器 CSV / geometry.receivers.coordinates_file",
            "接收器坐标文件缺失或内容无效。",
            "选择接收器 CSV；表头为 x_m,y_m，至少两行有限数值坐标，并删除重复坐标。",
        ),
        (
            ("坐标 csv",),
            "建立算例 → 观测系统或噪声源 CSV",
            "x_m,y_m 坐标文件",
            "坐标文件的编码、表头或数值不符合要求。",
            "使用 UTF-8 CSV 和 x_m,y_m 表头；每行填有限数字且不要重复位置。",
        ),
        (
            ("receiver grid", "spacing_x_m", "spacing_y_m"),
            "建立算例 → 观测系统 Geometry → 二维面阵",
            "x/y 起止坐标与间距",
            "面阵范围或网格间距不成立。",
            "保证 x终点>x起点、y终点>y起点，并让两个方向的跨度都能被各自间距整除。",
        ),
        (
            ("receiver span", "stop_x_m", "geometry.receivers.spacing_m", "receiver coordinates"),
            "建立算例 → 观测系统 Geometry → 线阵",
            "起点 x、终点 x、道间距",
            "线阵范围或道间距不成立。",
            "保证终点 x > 起点 x、道间距 > 0，并让 (终点−起点)/道间距 为整数。",
        ),
        (
            ("distance_bin_m",),
            "建立算例 → 观测系统 Geometry",
            "径向距离箱 distance_bin_m",
            "被动二维/文件阵列缺少径向距离箱或其值无效。",
            "在 Geometry 中填写大于 0 的“径向距离箱”；可先取接收点典型间距，通常 2–10 m。",
        ),
        (
            ("请选择噪声源", "source coordinate file", "ambient_sources.coordinates_file", "file source layout"),
            "建立算例 → 环境噪声源空间分布",
            "噪声源 CSV / ambient_sources.coordinates_file",
            "文件型噪声源没有选择有效坐标文件。",
            "上传 UTF-8 噪声源 CSV，表头为 x_m,y_m；源数会按有效数据行自动更新。",
        ),
        (
            ("ambient_sources.outer_radius_m", "inner_radius_m"),
            "建立算例 → 环境噪声源空间分布 → 源环",
            "源环内/外半径",
            "源环半径关系不成立。",
            "设置内半径 ≥ 0，并确保外半径严格大于内半径；同时避免源与接收器重合。",
        ),
        (
            ("road/line sources", "x_min_m", "x_max_m"),
            "建立算例 → 环境噪声源空间分布 → 源线",
            "源线 x 起点、x 终点、y",
            "源线坐标范围不成立。",
            "所有坐标填写有限数字，并保证 x 终点严格大于 x 起点。",
        ),
        (
            ("ambient_sources",),
            "建立算例 → 环境噪声源空间分布",
            "ambient_sources.*",
            "噪声源空间参数缺失或无效。",
            "检查源类型、数量、空间种子、方位采样及该类型对应的半径/线段/CSV 参数。",
        ),
        (
            ("请选择 psd", "noise.psd_file", "psd file"),
            "建立算例 → 噪声时间模型与 PSD",
            "PSD CSV / noise.psd_file",
            "选择 file PSD 后没有提供有效 PSD 文件。",
            "上传 UTF-8 CSV，表头为 frequency_hz,power；频率递增、功率非负。",
        ),
        (
            ("noise.window_s", "window_s must", "correlation window"),
            "建立算例 → 噪声时间模型与 PSD",
            "独立窗长 window_s / 总时长 duration_s",
            "互相关窗长与记录时长或最大传播时延不匹配。",
            "令 window_s ≤ duration_s；若输入审核仍失败，增大 window_s，直到大于表中 required lag。",
        ),
        (
            ("noise frequencies",),
            "建立算例 → 噪声时间模型与 PSD",
            "PSD 最低频率、主频、最高频率",
            "PSD 三个频率的次序不成立。",
            "设置 PSD最低频率 < PSD主频 ≤ PSD最高频率。",
        ),
        (
            ("noise.maximum_frequency_hz", "processing.frequency_max_hz must not exceed noise"),
            "建立算例 → 噪声时间模型与频散计算范围",
            "PSD最高频率 / 处理最高频率 / dt",
            "噪声或处理上限超出源频带或时间 Nyquist。",
            "保证处理最高频率 ≤ PSD最高频率 < 1/(2dt)；可降低频率上限或减小 dt。",
        ),
        (
            ("noise.",),
            "建立算例 → 噪声时间模型与 PSD",
            "noise.*",
            "噪声时间参数缺失或无效。",
            "检查总时长、窗长、dt、PSD 频带、随机种子、源振幅和 Poisson 事件率。",
        ),
        (
            ("active_source.maximum_frequency_hz",),
            "建立算例 → 主动源与子波",
            "Ricker 主频、正演最高频率、dt",
            "主动源频率关系或时间采样不满足要求。",
            "保证正演最高频率 ≥ Ricker 主频且 < 1/(2dt)；可降低最高频率或减小 dt。",
        ),
        (
            ("active_source", "active source"),
            "建立算例 → 主动源与子波",
            "active_source.*",
            "主动源位置、分量或子波参数无效。",
            "检查炮点 x/y、分量、Ricker 主频、最高频率和 dt；坐标必须为有限数字。",
        ),
        (
            ("processing maximum values",),
            "建立算例 → 频散计算范围",
            "处理频率/相速度上下限",
            "处理范围的上限没有大于下限。",
            "保证处理最高频率 > 最低频率，最高相速度 > 最低相速度。",
        ),
        (
            ("processing.frequency_max_hz",),
            "建立算例 → 频散计算范围",
            "处理最高频率 / dt / PSD最高频率",
            "处理最高频率超出可计算范围。",
            "降低处理最高频率，或减小 dt；被动模式还必须不高于 PSD 最高频率。",
        ),
        (
            ("processing.",),
            "建立算例 → 频散计算范围或高级计算参数",
            "processing.*",
            "频散处理参数缺失、越界或组合无效。",
            "检查频率范围、速度范围、步长、taper、归一化、拾取和理论模态参数。",
        ),
        (
            ("radial_interpolation_step_m",),
            "建立算例 → 高级计算参数",
            "径向 Green 插值",
            "径向插值步长相对最短波长过粗。",
            "把“径向 Green 插值”减小到技术信息给出的上限，或降低 PSD 最高频率。",
        ),
        (
            ("run.name",),
            "建立算例 → 基础参数",
            "算例名称 run.name",
            "算例名称为空。",
            "输入非空算例名称；推荐使用简短英文、数字和下划线。",
        ),
        (
            ("run.",),
            "建立算例 → 基础参数或高级计算参数",
            "run.*",
            "运行参数缺失或无效。",
            "检查算例名称、模式、输出目录、workers、分块大小和径向插值步长。",
        ),
    )

    for needles, location, field_path, problem, suggestion in rules:
        if any(needle in lower for needle in needles):
            return InputIssue(location, field_path, problem, suggestion, message)
    return InputIssue(
        "建立算例 → 当前参数组",
        "未能自动定位",
        "输入未通过核心校验。",
        "根据下方原始信息检查对应字段；若仍不明确，可先载入同模式模板，再逐项替换参数。",
        message,
    )


def parse_toml_bytes(content: bytes) -> dict[str, Any]:
    """Parse uploaded TOML bytes and return an independent mutable mapping."""
    try:
        mapping = tomllib.loads(content.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CaseInputError("TOML 文件必须使用 UTF-8 编码") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CaseInputError(f"TOML 解析失败：{exc}") from exc
    if not isinstance(mapping, dict) or "run" not in mapping:
        raise CaseInputError("TOML 必须包含 [run] 表")
    return deepcopy(mapping)


def load_template_mapping(path: str | Path) -> dict[str, Any]:
    template = Path(path).expanduser().resolve()
    if not template.is_file():
        raise CaseInputError(f"模板文件不存在：{template}")
    return parse_toml_bytes(template.read_bytes())


def stage_uploaded_file(
    directory: str | Path,
    filename: str,
    content: bytes,
    *,
    allowed_suffixes: tuple[str, ...],
) -> Path:
    """Persist one upload under a content-addressed, traversal-safe filename."""
    safe_name = _SAFE_NAME.sub("_", Path(filename).name).strip("._")
    if not safe_name:
        raise CaseInputError("上传文件名无效")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {item.lower() for item in allowed_suffixes}:
        allowed = ", ".join(allowed_suffixes)
        raise CaseInputError(f"文件 {filename!r} 类型不支持；允许：{allowed}")
    if not content:
        raise CaseInputError(f"上传文件 {filename!r} 为空")
    digest = hashlib.sha256(content).hexdigest()[:16]
    output = Path(directory).expanduser().resolve() / "uploads"
    output.mkdir(parents=True, exist_ok=True)
    staged = output / f"{digest}_{safe_name}"
    if not staged.exists() or staged.read_bytes() != content:
        staged.write_bytes(content)
    return staged


def coordinate_csv_count(content: bytes) -> int:
    """Validate an uploaded x/y coordinate CSV and return its row count."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CaseInputError("坐标 CSV 必须使用 UTF-8 编码") from exc
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or not {"x_m", "y_m"}.issubset(rows[0]):
        raise CaseInputError("坐标 CSV 必须包含 x_m,y_m 表头和至少一行数据")
    positions: list[tuple[float, float]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            position = (float(row["x_m"]), float(row["y_m"]))
        except (TypeError, ValueError) as exc:
            raise CaseInputError(f"坐标 CSV 第 {row_number} 行不是有效数字") from exc
        if not all(math.isfinite(item) for item in position):
            raise CaseInputError(f"坐标 CSV 第 {row_number} 行必须为有限数值")
        positions.append(position)
    if len(set(positions)) != len(positions):
        raise CaseInputError("坐标 CSV 含有重复的 x_m,y_m")
    return len(rows)


def parse_shot_csv(content: bytes) -> list[ShotLocation]:
    """Parse an active-shot table used by the UI.

    The file must contain ``x_m,y_m``.  ``shot_id`` is optional; when omitted,
    stable row labels ``shot_001`` ... are assigned.  The numerical solver is
    intentionally a single-shot workflow, so the UI lets the user select one
    returned row and persists that row's x/y coordinates in ``[active_source]``.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CaseInputError("炮点 CSV 必须使用 UTF-8 编码") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or not {"x_m", "y_m"}.issubset(reader.fieldnames):
        raise CaseInputError("炮点 CSV 必须包含 x_m,y_m 表头和至少一行数据")
    rows = list(reader)
    if not rows:
        raise CaseInputError("炮点 CSV 必须包含 x_m,y_m 表头和至少一行数据")
    shots: list[ShotLocation] = []
    for row_index, row in enumerate(rows, start=1):
        try:
            x_m = float(row["x_m"])
            y_m = float(row["y_m"])
        except (TypeError, ValueError) as exc:
            raise CaseInputError(f"炮点 CSV 第 {row_index + 1} 行不是有效数字") from exc
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise CaseInputError(f"炮点 CSV 第 {row_index + 1} 行必须为有限数值")
        raw_id = row.get("shot_id")
        shot_id = str(raw_id).strip() if raw_id is not None else ""
        if not shot_id:
            shot_id = f"shot_{row_index:03d}"
        shots.append(ShotLocation(shot_id, x_m, y_m))
    identifiers = [shot.shot_id for shot in shots]
    if len(set(identifiers)) != len(identifiers):
        raise CaseInputError("炮点 CSV 的 shot_id 不能重复")
    return shots


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, Path)):
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return repr(value)
    raise CaseInputError(f"无法写入 TOML 的值类型：{type(value).__name__}")


def toml_dumps(mapping: dict[str, Any]) -> str:
    """Serialize the WaveDispX configuration subset without an extra dependency."""
    lines: list[str] = []
    preferred = (
        "run",
        "model",
        "geometry",
        "active_source",
        "ambient_sources",
        "noise",
        "processing",
    )
    ordered_keys = [key for key in preferred if key in mapping]
    ordered_keys.extend(key for key in mapping if key not in ordered_keys)

    def write_table(prefix: str, table: dict[str, Any]) -> None:
        scalar_items = {
            key: value
            for key, value in table.items()
            if not isinstance(value, (dict, list))
        }
        if scalar_items:
            lines.append(f"[{prefix}]")
            lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in scalar_items.items())
            lines.append("")
        for key, value in table.items():
            child = f"{prefix}.{key}"
            if isinstance(value, dict):
                write_table(child, value)
            elif isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise CaseInputError(f"{child} 只支持表数组")
                    lines.append(f"[[{child}]]")
                    lines.extend(
                        f"{item_key} = {_toml_scalar(item_value)}"
                        for item_key, item_value in item.items()
                    )
                    lines.append("")

    for key in ordered_keys:
        value = mapping[key]
        if not isinstance(value, dict):
            raise CaseInputError(f"顶层 {key} 必须为 TOML 表")
        write_table(key, value)
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class CaseSession:
    """Keep one validated case and dependency-safe staged results in memory."""

    workspace_directory: Path
    mapping: dict[str, Any] | None = None
    workflow: WaveDispXWorkflow | None = None

    def __post_init__(self) -> None:
        self.workspace_directory = self.workspace_directory.expanduser().resolve()
        self.workspace_directory.mkdir(parents=True, exist_ok=True)

    def stage_upload(
        self,
        filename: str,
        content: bytes,
        *,
        kind: Literal["toml", "coordinates", "psd"],
    ) -> Path:
        suffixes = {
            "toml": (".toml",),
            "coordinates": (".csv",),
            "psd": (".csv",),
        }
        return stage_uploaded_file(
            self.workspace_directory,
            filename,
            content,
            allowed_suffixes=suffixes[kind],
        )

    def apply_mapping(self, mapping: dict[str, Any]) -> tuple[RunConfig, list[str]]:
        candidate = deepcopy(mapping)
        config = config_from_mapping(
            candidate,
            base_directory=self.workspace_directory,
        )
        if self.workflow is None:
            self.workflow = WaveDispXWorkflow(
                config,
                source_mapping=candidate,
            )
            invalidated = ["forward", "observations", "analysis"]
        else:
            invalidated = self.workflow.configure(
                config,
                source_mapping=candidate,
            )
        self.mapping = candidate
        return config, invalidated

    @property
    def config_toml(self) -> str:
        if self.mapping is None:
            raise CaseInputError("尚未建立算例")
        return toml_dumps(self.mapping)

    def run_stage(self, stage: Literal["review", "forward", "observations", "analysis", "all"]):
        if self.workflow is None:
            raise CaseInputError("请先建立并校验算例")
        actions = {
            "review": self.workflow.review_inputs,
            "forward": self.workflow.compute_forward,
            "observations": self.workflow.simulate_observations,
            "analysis": self.workflow.compute_dispersion,
            "all": lambda: self.workflow.run_all(save=True),
        }
        return actions[stage]()
