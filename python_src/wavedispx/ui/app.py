"""Four-step local Streamlit UI for WaveDispX.

The UI contains no modelling equations of its own.  Every review, forward,
correlation and dispersion action delegates to the same WaveDispXWorkflow used by the
notebooks and command line.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt

from wavedispx.app_service import (
    CaseInputError,
    CaseSession,
    coordinate_csv_count,
    explain_input_error,
    load_template_mapping,
    parse_shot_csv,
    parse_toml_bytes,
)
from wavedispx.config import ConfigError
from wavedispx.preflight import plot_preflight_review
from wavedispx.workflow_plotting import (
    plot_workflow_dispersion,
    plot_workflow_forward,
    plot_workflow_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = {
    "主动源｜快速检查": PROJECT_ROOT / "examples/active_quick.toml",
    "主动源｜正式两层模型": PROJECT_ROOT / "examples/active_two_layer.toml",
    "主动源｜高阶模态 3C": PROJECT_ROOT / "examples/active_high_modes_3c.toml",
    "被动源｜快速检查": PROJECT_ROOT / "examples/passive_quick.toml",
    "被动源｜正式收敛": PROJECT_ROOT / "examples/passive_two_layer.toml",
    "被动源｜高阶模态物理 3C": PROJECT_ROOT / "examples/passive_high_modes_3c.toml",
    "被动源｜二维面阵": PROJECT_ROOT / "examples/passive_grid_quick.toml",
}

HELP = {
    "template": "载入一套完整且已验证的起始参数。快速模板用于几分钟内检查流程，正式模板用于收敛计算。",
    "case_name": "算例标识，会写入图题、manifest 和默认输出目录。建议只用简短英文、数字和下划线。",
    "mode": "active：计算主动炮集及 Rayleigh/Love；passive：先合成有限时长混合噪声，再在接收端计算 NCF。",
    "output": "HDF5、CSV、汇总图、manifest 和可移交 input.toml 的保存目录。允许绝对路径或相对路径。",
    "thickness": "层厚，单位 m。最后一层必须填 inf，表示向下无限延伸的半空间。",
    "vp": "P 波速度，单位 m/s。必须大于同层 Vs。",
    "vs": "S 波速度，单位 m/s；主要控制 Rayleigh/Love 相速度和空间采样要求。",
    "density": "密度，单位 kg/m³；进入分层刚度和阻抗计算。",
    "qp": "P 波品质因子。inf 表示无 P 波衰减；有限正数越小，衰减越强。",
    "qs": "S 波品质因子。inf 表示无 S 波衰减；有限正数越小，面波衰减越强。",
    "receiver_layout": "参数设计｜线阵：输入起止点和道距；参数设计｜面阵：输入 x/y 网格；文件输入｜CSV：上传任意二维坐标。",
    "line_start": "线阵第一道的 x 坐标，单位 m。",
    "line_stop": "线阵最后一道的 x 坐标，单位 m；与起点之差必须能被道间距整除。",
    "spacing": "道间距，单位 m，控制空间混叠。近似上限 f_alias=c_min/(2Δx)，高频模拟需要更小道距。",
    "line_y": "整条线阵的 y 坐标，单位 m。水平分层模型中改变整体 y 不改变径向响应。",
    "grid_x_start": "二维面阵 x 方向起点，单位 m。",
    "grid_x_stop": "二维面阵 x 方向终点，单位 m；跨度必须为 x 间距的整数倍。",
    "grid_x_spacing": "二维面阵 x 方向间距，单位 m；影响接收器数量和空间混叠。",
    "grid_y_start": "二维面阵 y 方向起点，单位 m。",
    "grid_y_stop": "二维面阵 y 方向终点，单位 m；跨度必须为 y 间距的整数倍。",
    "grid_y_spacing": "二维面阵 y 方向间距，单位 m；与 x 间距共同决定最近邻采样。",
    "distance_bin": "被动二维/不规则阵列把台站对距离归入该宽度的径向箱。过小则每箱对数不足，过大则平滑空间相位。",
    "receiver_file": "上传含 x_m,y_m 表头的 UTF-8 CSV。每行一个接收器，允许任意二维形状，但坐标不能重复。",
    "shot_input": "界面输入适合单炮；Shot CSV 可包含 shot_id,x_m,y_m 多行，但当前每个算例只选择并计算其中一炮。",
    "shot_file": "上传 UTF-8 炮点表。x_m,y_m 必填，shot_id 可选；多炮文件载入后需明确选择本次运行的炮号。",
    "shot_select": "当前算例要计算的炮点。选择结果会固化为 [active_source] 的 x_m,y_m，保证输出 TOML 可独立复现。",
    "source_x": "主动炮点 x 坐标，单位 m。所有主动源检距由它与接收器坐标计算。",
    "source_y": "主动炮点 y 坐标，单位 m。二维阵列会按真实平面距离计算 Green 响应。",
    "component": "vertical：Rayleigh/竖直载荷；horizontal：Love/横向载荷；both：同时计算两类分量。",
    "wavelet": "主动源时间函数。当前稳定实现为 Ricker 子波。",
    "dominant_frequency": "Ricker 子波主频，单位 Hz，决定主要能量中心；应落在处理频带附近。",
    "forward_max_frequency": "正演最高频率，单位 Hz。必须高于主频且低于时间 Nyquist 频率。",
    "dt": "时间采样间隔，单位 s。Nyquist 频率为 1/(2dt)，最高频率必须低于该值。",
    "ambient_component": "vertical：仅 Fz→Z 的快速兼容流程；three_component：每个源独立激发全局 Fx/Fy/Fz，经完整五分量 Green 张量传播，输出 RR/TT/ZZ NCF。",
    "ambient_layout": "源环：围绕阵列照明；源线：模拟道路/线性噪声带；CSV：输入任意噪声源位置。",
    "source_count": "同时参与混合的独立噪声源数量。快速检查可用 24；收敛频散通常建议 192–384。",
    "source_seed": "噪声源空间分布随机种子。相同参数和种子会生成完全相同的 geometry。",
    "angular_sampling": "stratified：每个方位扇区一个随机源，推荐；uniform：等方位；random：完全随机，可能出现方位空缺。",
    "inner_radius": "环形噪声源带内半径，单位 m；必须小于外半径并避免源落入接收阵列。",
    "outer_radius": "环形噪声源带外半径，单位 m。半径按面积均匀抽样。",
    "source_line_start": "线性噪声带 x 起点，单位 m。",
    "source_line_stop": "线性噪声带 x 终点，单位 m，必须大于起点。",
    "source_line_y": "线性噪声带的固定 y 坐标，单位 m。",
    "source_file": "上传含 x_m,y_m 表头的噪声源 CSV。源数由文件行数自动读取。",
    "temporal_model": "stationary_gaussian：平稳随机频谱，推荐环境噪声；poisson_ricker：随机时刻的间歇脉冲事件。",
    "psd_shape": "ricker：Ricker 型谱；flat_band：带内近似平坦；file：读取 frequency_hz,power CSV。",
    "duration": "有限记录总时长。60 s 适合调试；600 s 通常可用；1800 s 更接近稳定收敛。",
    "window": "独立合成与互相关窗长，必须长于最大台间传播时延；当前各窗独立且 overlap 固定为 0。",
    "psd_min": "噪声 PSD 最低有效频率，单位 Hz；低于它的源能量置零。",
    "psd_dominant": "Ricker 型噪声 PSD 的峰值频率，单位 Hz；flat_band/file 时仅保留为配置参考。",
    "psd_max": "噪声 PSD 最高有效频率，单位 Hz；必须低于时间 Nyquist，处理上限不能超过它。",
    "noise_seed": "噪声时间序列随机种子，与空间 geometry 种子相互独立。",
    "source_amplitude": "每个噪声源的相对振幅尺度。频散能量最终归一化，但原始记录幅值会随它变化。",
    "independent_sources": "开启时每个噪声源具有独立随机时间函数，是恢复 Green 函数的标准假设；关闭仅用于相干源敏感性试验。",
    "event_rate": "poisson_ricker 模型中每个源每秒的平均事件数；只影响间歇事件模型。",
    "psd_file": "上传含 frequency_hz,power 表头的 UTF-8 CSV；频率需递增，功率应为非负数。",
    "method": "主动炮集通常用 Hankel/FJ；被动 NCF 默认 phase_shift；Bessel/Hankel 可作方法敏感性对照。",
    "compare_methods": "同时计算 phase_shift、Bessel(FJ) 与 Hankel 三张同网格频散图，用于判断旁瓣或假频是否依赖成像方法。计算时间约增加 2–3 倍。",
    "processing_fmin": "频散图最低频率，单位 Hz。过低时阵列孔径不足会产生不可分辨能量。",
    "processing_fmax": "频散图最高频率，单位 Hz；必须满足时间 Nyquist、空间混叠和源谱上限。",
    "velocity_min": "相速度扫描下限，单位 m/s；应低于预计基阶最低相速度。",
    "velocity_max": "相速度扫描上限，单位 m/s；应覆盖基阶、高阶及理论曲线。",
    "velocity_step": "相速度扫描步长，单位 m/s。更小可使图更细，但计算量和数组大小增加。",
    "distance_taper": "距离域 Tukey 窗比例 0–1。被动推荐 0.5，可减弱有限孔径旁瓣；0 表示不加窗。",
    "workers": "CPU 并行线程数。0 表示自动使用可用核心；正式算例通常保持 0。",
    "frequency_block": "C++ 正演核心每批处理的频率数。较大可减少调度开销，但增加峰值内存；常用 8–32。",
    "source_block": "被动噪声源分块大小。较大通常更快但占用更多内存；主动模式基本不使用。",
    "radial_step": "层状介质径向 Green 插值步长，单位 m。必须不超过最短 S 波波长的约 5%。",
    "store_responses": "保存每个噪声源的完整传递响应会显著增加 HDF5 和内存；常规真实混合噪声应关闭。",
    "correlation_normalization": "spectral_whitening：推荐环境噪声；one_bit：抑制强瞬态；none：不做归一化，主动源通常使用。",
    "zero_padding": "时间 FFT 补零倍数，只提高频率采样密度，不增加真实分辨率。主动常用 5，被动常用 8。",
    "picker_smoothness": "动态规划拾取的平滑惩罚权重。越大越偏好连续曲线，但可能压制真实快速变化。",
    "picker_jump": "相邻频点允许的最大速度跳变，单位 m/s；用于防止拾取跳到旁瓣或其他模态。",
    "kernel_step": "Bessel/Hankel 核查表步长。0 使用逐点精确特殊函数；0.002 通常兼顾速度和精度。phase_shift 不使用。",
    "picker_energy": "自动拾取所需的最小归一化能量 0–1；低于阈值的频点不输出拾取。",
    "rayleigh_modes": "独立理论频散中计算并绘制的 Rayleigh 模态数。",
    "love_modes": "独立理论频散中计算并绘制的 Love 模态数；被动竖直分量仅作模型参考。",
}


def _new_session() -> CaseSession:
    workspace = Path(tempfile.mkdtemp(prefix="wavedispx-ui-"))
    return CaseSession(workspace)


def _ensure_state(st) -> CaseSession:
    if "case_session" not in st.session_state:
        st.session_state.case_session = _new_session()
        st.session_state.case_mapping = load_template_mapping(
            TEMPLATES["被动源｜快速检查"]
        )
        st.session_state.template_name = "被动源｜快速检查"
        st.session_state.case_revision = 0
        st.session_state.last_message = "已载入被动源快速模板；尚未开始正演。"
    return st.session_state.case_session


def _index(options: list[str], value: Any, fallback: int = 0) -> int:
    try:
        return options.index(str(value))
    except ValueError:
        return fallback


def _layer_rows(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer in mapping["model"]["layers"]:
        item = dict(layer)
        thickness = float(item["thickness_m"])
        item["thickness_m"] = "inf" if thickness == float("inf") else str(thickness)
        rows.append(item)
    return rows


def _normalize_layers(edited) -> list[dict[str, float]]:
    records = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
    if not records:
        raise CaseInputError("模型至少需要一层")
    layers: list[dict[str, float]] = []
    for index, row in enumerate(records, start=1):
        raw_thickness = str(row.get("thickness_m", "")).strip().lower()
        thickness = float("inf") if raw_thickness in {"inf", "+inf", "半空间"} else float(raw_thickness)
        try:
            layers.append(
                {
                    "thickness_m": thickness,
                    "vp_mps": float(row["vp_mps"]),
                    "vs_mps": float(row["vs_mps"]),
                    "density_kgm3": float(row["density_kgm3"]),
                    "qp": float(row.get("qp", float("inf"))),
                    "qs": float(row.get("qs", float("inf"))),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CaseInputError(f"模型第 {index} 行含无效数值") from exc
    return layers


def _render_input_error(st, error: Exception) -> None:
    """Show one rejected input as location, cause, and concrete correction."""
    issue = explain_input_error(error)
    st.error(f"参数校验失败｜{issue.field_path}")
    with st.container(border=True):
        st.markdown(f"**位置：** {issue.location}")
        st.markdown(f"**问题：** {issue.problem}")
        st.markdown(f"**怎么改：** {issue.suggestion}")
        with st.expander("查看原始技术信息"):
            st.code(issue.technical_message)


def _render_required_failures(st, failures) -> None:
    """Render blocking preflight checks with their editable parameters."""
    if not failures:
        return
    st.error(f"{len(failures)} 个 required 检查失败；请按下面逐项修改。")
    for index, check in enumerate(failures, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {check.name}｜参数位置：`{check.parameter_path}`**")
            st.markdown(
                f"当前值：**{check.value}**　　必须满足：**{check.criterion}**"
            )
            st.markdown(f"怎么改：{check.guidance}")


def _load_controls(st, session: CaseSession) -> None:
    left, right = st.columns([1, 1])
    with left:
        template_name = st.selectbox(
            "从模板开始", list(TEMPLATES), key="template_name", help=HELP["template"]
        )
        if st.button(
            "载入模板", width="stretch", help="用所选模板重置当前尚未提交的界面参数。"
        ):
            st.session_state.case_mapping = load_template_mapping(TEMPLATES[template_name])
            st.session_state.case_revision += 1
            st.session_state.last_message = f"已载入：{template_name}"
            st.rerun()
    with right:
        upload = st.file_uploader(
            "或载入完整 TOML",
            type=["toml"],
            help="相对路径文件需在下面相应的 CSV 上传控件中重新选择。",
            key=f"toml_upload_{st.session_state.case_revision}",
        )
        if st.button(
            "读取 TOML",
            disabled=upload is None,
            width="stretch",
            help="解析上传的完整配置并填充所有界面控件；此操作不启动正演。",
        ):
            st.session_state.case_mapping = parse_toml_bytes(upload.getvalue())
            st.session_state.case_revision += 1
            st.session_state.last_message = f"已读取 TOML：{upload.name}"
            st.rerun()


def _case_form(st, session: CaseSession) -> None:
    mapping = deepcopy(st.session_state.case_mapping)
    run = mapping.setdefault("run", {})
    mode = str(run.get("mode", "passive"))
    revision = st.session_state.case_revision

    # A normal container (not st.form) is intentional: geometry/source/PSD
    # selectors must immediately reveal their dependent fields.  Expensive work
    # is still gated by the explicit validation and run buttons below.
    with st.container(border=True):
        st.subheader("基础参数")
        st.caption("选择文件输入时，标有“必填”的上传项必须提供；校验失败会显示参数位置和修改方法。")
        col1, col2, col3 = st.columns(3)
        name = col1.text_input(
            "算例名称",
            str(run.get("name", "wavedispx_case")),
            key=f"name_{revision}",
            help=HELP["case_name"],
        )
        modes = ["active", "passive"]
        selected_mode = col2.selectbox(
            "模式", modes, index=_index(modes, mode), help=HELP["mode"], key=f"mode_{revision}"
        )
        default_output = PROJECT_ROOT / "output_python" / f"ui_{name}"
        output_directory = col3.text_input(
            "输出目录", str(default_output), help=HELP["output"], key=f"output_{revision}"
        )

        st.markdown("#### 速度—密度模型")
        edited_layers = st.data_editor(
            _layer_rows(mapping),
            num_rows="dynamic",
            width="stretch",
            key=f"layers_{revision}",
            column_config={
                "thickness_m": st.column_config.TextColumn(
                    "厚度 m（末层填 inf）", help=HELP["thickness"]
                ),
                "vp_mps": st.column_config.NumberColumn(
                    "Vp m/s", min_value=1.0, help=HELP["vp"]
                ),
                "vs_mps": st.column_config.NumberColumn(
                    "Vs m/s", min_value=1.0, help=HELP["vs"]
                ),
                "density_kgm3": st.column_config.NumberColumn(
                    "密度 kg/m³", min_value=1.0, help=HELP["density"]
                ),
                "qp": st.column_config.NumberColumn("Qp", help=HELP["qp"]),
                "qs": st.column_config.NumberColumn("Qs", help=HELP["qs"]),
            },
        )

        receiver = mapping.setdefault("geometry", {}).setdefault("receivers", {})
        st.markdown("#### 观测系统 Geometry")
        st.caption("可在界面参数化设计线阵/面阵，也可上传任意二维接收器坐标 CSV。")
        layouts = ["line", "grid", "file"]
        layout_labels = {
            "line": "参数设计｜线阵",
            "grid": "参数设计｜二维面阵",
            "file": "文件输入｜CSV",
        }
        receiver_layout = st.radio(
            "Geometry 输入方式",
            layouts,
            index=_index(layouts, receiver.get("layout", "line")),
            horizontal=True,
            format_func=layout_labels.get,
            help=HELP["receiver_layout"],
            key=f"receiver_layout_{revision}",
        )
        receiver_values: dict[str, Any] = {"layout": receiver_layout}
        receiver_upload = None
        if receiver_layout == "line":
            line_defaults = receiver if receiver.get("layout", "line") == "line" else {}
            c1, c2, c3, c4 = st.columns(4)
            receiver_values.update(
                start_x_m=c1.number_input("起点 x (m)", value=float(line_defaults.get("start_x_m", -100.0)), help=HELP["line_start"], key=f"rx0_{revision}"),
                stop_x_m=c2.number_input("终点 x (m)", value=float(line_defaults.get("stop_x_m", 100.0)), help=HELP["line_stop"], key=f"rx1_{revision}"),
                spacing_m=c3.number_input("道间距 (m)", min_value=0.01, value=float(line_defaults.get("spacing_m", 5.0)), help=HELP["spacing"], key=f"rdx_{revision}"),
                y_m=c4.number_input("阵列 y (m)", value=float(line_defaults.get("y_m", 0.0)), help=HELP["line_y"], key=f"ry_{revision}"),
            )
        elif receiver_layout == "grid":
            grid_defaults = receiver if receiver.get("layout") in {"grid", "array"} else {}
            c1, c2, c3 = st.columns(3)
            receiver_values.update(
                start_x_m=c1.number_input("x 起点 (m)", value=float(grid_defaults.get("start_x_m", -25.0)), help=HELP["grid_x_start"], key=f"gx0_{revision}"),
                stop_x_m=c2.number_input("x 终点 (m)", value=float(grid_defaults.get("stop_x_m", 25.0)), help=HELP["grid_x_stop"], key=f"gx1_{revision}"),
                spacing_x_m=c3.number_input("x 间距 (m)", min_value=0.01, value=float(grid_defaults.get("spacing_x_m", 5.0)), help=HELP["grid_x_spacing"], key=f"gdx_{revision}"),
            )
            c1, c2, c3, c4 = st.columns(4)
            receiver_values.update(
                start_y_m=c1.number_input("y 起点 (m)", value=float(grid_defaults.get("start_y_m", -25.0)), help=HELP["grid_y_start"], key=f"gy0_{revision}"),
                stop_y_m=c2.number_input("y 终点 (m)", value=float(grid_defaults.get("stop_y_m", 25.0)), help=HELP["grid_y_stop"], key=f"gy1_{revision}"),
                spacing_y_m=c3.number_input("y 间距 (m)", min_value=0.01, value=float(grid_defaults.get("spacing_y_m", 5.0)), help=HELP["grid_y_spacing"], key=f"gdy_{revision}"),
                distance_bin_m=c4.number_input("径向距离箱 (m)", min_value=0.01, value=float(grid_defaults.get("distance_bin_m", 5.0) or 5.0), help=HELP["distance_bin"], key=f"gdb_{revision}"),
            )
        else:
            receiver_upload = st.file_uploader(
                "上传接收器 CSV（x_m,y_m；必填）",
                type=["csv"],
                help=HELP["receiver_file"],
                key=f"receiver_upload_{revision}",
            )
            st.download_button(
                "下载接收器 CSV 模板",
                b"x_m,y_m\n0.0,0.0\n5.0,0.0\n10.0,0.0\n",
                file_name="receivers_template.csv",
                mime="text/csv",
                help="下载后可编辑为任意线阵、L 形阵列、十字阵或二维不规则阵列。",
            )
            receiver_values["coordinates_file"] = receiver.get("coordinates_file")
            receiver_values["distance_bin_m"] = st.number_input(
                "径向距离箱 (m；被动源必填)", min_value=0.01,
                value=float(receiver.get("distance_bin_m", 5.0) or 5.0),
                help=HELP["distance_bin"], key=f"fdb_{revision}"
            )

        active_values = None
        ambient_values = None
        noise_values = None
        ambient_upload = None
        psd_upload = None
        shot_upload = None
        shot_parse_error = None
        shot_input_mode = "manual"
        if selected_mode == "active":
            active = mapping.get("active_source", {})
            st.markdown("#### 主动源与子波")
            shot_input_modes = ["manual", "file"]
            shot_input_labels = {
                "manual": "界面输入｜单炮",
                "file": "文件输入｜Shot CSV",
            }
            shot_input_mode = st.radio(
                "炮点输入方式",
                shot_input_modes,
                horizontal=True,
                format_func=shot_input_labels.get,
                help=HELP["shot_input"],
                key=f"shot_input_{revision}",
            )
            shot_x_m = float(active.get("x_m", 0.0))
            shot_y_m = float(active.get("y_m", 0.0))
            if shot_input_mode == "file":
                shot_upload = st.file_uploader(
                    "上传炮点 CSV（shot_id,x_m,y_m；必填）",
                    type=["csv"],
                    help=HELP["shot_file"],
                    key=f"shot_upload_{revision}",
                )
                st.download_button(
                    "下载炮点 CSV 模板",
                    b"shot_id,x_m,y_m\nS001,0.0,0.0\nS002,-120.0,0.0\nS003,120.0,0.0\n",
                    file_name="shots_template.csv",
                    mime="text/csv",
                    help="炮点表可包含多炮；当前单炮算例会从表中选择一行运行。",
                )
                if shot_upload is not None:
                    try:
                        shots = parse_shot_csv(shot_upload.getvalue())
                        selected_shot = st.selectbox(
                            "选择本次运行的炮号",
                            shots,
                            format_func=lambda shot: (
                                f"{shot.shot_id}  |  x={shot.x_m:g} m, y={shot.y_m:g} m"
                            ),
                            help=HELP["shot_select"],
                            key=f"shot_select_{revision}",
                        )
                        shot_x_m = selected_shot.x_m
                        shot_y_m = selected_shot.y_m
                        st.caption(
                            f"已选 {selected_shot.shot_id}：({shot_x_m:g}, {shot_y_m:g}) m；"
                            "保存时写入 active_source.x_m / y_m。"
                        )
                    except CaseInputError as exc:
                        shot_parse_error = str(exc)
                        _render_input_error(st, exc)
                else:
                    st.warning("必填：请选择 Shot CSV；选择文件和炮号后才能校验算例。")
            else:
                c1, c2 = st.columns(2)
                shot_x_m = c1.number_input(
                    "炮点 x (m)", value=shot_x_m, help=HELP["source_x"], key=f"sx_{revision}"
                )
                shot_y_m = c2.number_input(
                    "炮点 y (m)", value=shot_y_m, help=HELP["source_y"], key=f"sy_{revision}"
                )
            c1, c2, c3 = st.columns(3)
            components = ["vertical", "horizontal", "both"]
            active_values = {
                "x_m": shot_x_m,
                "y_m": shot_y_m,
                "component": c1.selectbox("分量", components, index=_index(components, active.get("component", "both"), 2), help=HELP["component"], key=f"sc_{revision}"),
            }
            c1, c2, c3, c4 = st.columns(4)
            active_values["wavelet"] = c1.selectbox(
                "子波",
                ["ricker"],
                index=0,
                disabled=True,
                help=HELP["wavelet"],
                key=f"sw_{revision}",
            )
            active_values.update(
                dominant_frequency_hz=c2.number_input("Ricker 主频 (Hz)", min_value=0.01, value=float(active.get("dominant_frequency_hz", 20.0)), help=HELP["dominant_frequency"], key=f"sfd_{revision}"),
                maximum_frequency_hz=c3.number_input("正演最高频率 (Hz)", min_value=0.01, value=float(active.get("maximum_frequency_hz", 100.0)), help=HELP["forward_max_frequency"], key=f"sfm_{revision}"),
                dt_s=c4.number_input("采样间隔 dt (s)", min_value=0.00001, value=float(active.get("dt_s", 0.001)), format="%.5f", help=HELP["dt"], key=f"sdt_{revision}"),
            )
        else:
            ambient = mapping.get("ambient_sources", {})
            st.markdown("#### 环境噪声源空间分布")
            source_layouts = ["annulus", "line", "file"]
            source_layout_labels = {
                "annulus": "参数设计｜源环",
                "line": "参数设计｜源线",
                "file": "文件输入｜CSV",
            }
            source_layout = st.radio(
                "噪声源类型", source_layouts,
                index=_index(source_layouts, ambient.get("layout", "annulus")),
                horizontal=True,
                format_func=source_layout_labels.get,
                help=HELP["ambient_layout"],
                key=f"source_layout_{revision}"
            )
            c1, c2, c3, c4 = st.columns(4)
            ambient_values = {
                "component": c4.selectbox(
                    "被动源分量",
                    ["vertical", "three_component"],
                    index=_index(
                        ["vertical", "three_component"],
                        ambient.get("component", "vertical"),
                    ),
                    help=HELP["ambient_component"],
                    key=f"scomp_{revision}",
                ),
                "layout": source_layout,
                "count": int(c1.number_input("噪声源数量", min_value=1, value=int(ambient.get("count", 192)), step=1, help=HELP["source_count"], key=f"sn_{revision}")),
                "seed": int(c2.number_input("空间随机种子", min_value=0, value=int(ambient.get("seed", 42)), step=1, help=HELP["source_seed"], key=f"sseed_{revision}")),
                "angular_sampling": c3.selectbox("方位采样", ["stratified", "uniform", "random"], index=_index(["stratified", "uniform", "random"], ambient.get("angular_sampling", "stratified")), help=HELP["angular_sampling"], key=f"sas_{revision}"),
            }
            if source_layout == "annulus":
                c1, c2 = st.columns(2)
                ambient_values.update(
                    inner_radius_m=c1.number_input("源环内半径 (m)", min_value=0.0, value=float(ambient.get("inner_radius_m", 250.0)), help=HELP["inner_radius"], key=f"sri_{revision}"),
                    outer_radius_m=c2.number_input("源环外半径 (m)", min_value=0.01, value=float(ambient.get("outer_radius_m", 350.0)), help=HELP["outer_radius"], key=f"sro_{revision}"),
                )
            elif source_layout == "line":
                c1, c2, c3 = st.columns(3)
                ambient_values.update(
                    x_min_m=c1.number_input("源线 x 起点 (m)", value=float(ambient.get("x_min_m", -500.0) or -500.0), help=HELP["source_line_start"], key=f"slx0_{revision}"),
                    x_max_m=c2.number_input("源线 x 终点 (m)", value=float(ambient.get("x_max_m", 500.0) or 500.0), help=HELP["source_line_stop"], key=f"slx1_{revision}"),
                    y_m=c3.number_input("源线 y (m)", value=float(ambient.get("y_m", 300.0) or 300.0), help=HELP["source_line_y"], key=f"sly_{revision}"),
                )
            else:
                ambient_upload = st.file_uploader(
                    "上传噪声源 CSV（x_m,y_m；必填，数量自动读取）",
                    type=["csv"],
                    help=HELP["source_file"],
                    key=f"source_upload_{revision}",
                )
                st.download_button(
                    "下载噪声源 CSV 模板",
                    b"x_m,y_m\n-300.0,300.0\n0.0,350.0\n300.0,300.0\n",
                    file_name="ambient_sources_template.csv",
                    mime="text/csv",
                    help="下载后可输入道路、工厂、环带或任意实测噪声源位置。",
                )
                ambient_values["coordinates_file"] = ambient.get("coordinates_file")

            noise = mapping.get("noise", {})
            st.markdown("#### 噪声时间模型与 PSD")
            c1, c2, c3, c4 = st.columns(4)
            psd_shapes = ["ricker", "flat_band", "file"]
            temporal_models = ["stationary_gaussian", "poisson_ricker"]
            noise_values = {
                "temporal_model": c1.selectbox("时间模型", temporal_models, index=_index(temporal_models, noise.get("temporal_model", "stationary_gaussian")), help=HELP["temporal_model"], key=f"ntm_{revision}"),
                "psd_shape": c2.selectbox("PSD 形状", psd_shapes, index=_index(psd_shapes, noise.get("psd_shape", "ricker")), help=HELP["psd_shape"], key=f"npsd_{revision}"),
                "duration_s": c3.number_input("总记录时长 (s)", min_value=0.1, value=float(noise.get("duration_s", 600.0)), help=HELP["duration"], key=f"ndur_{revision}"),
                "window_s": c4.number_input("独立窗长 (s)", min_value=0.1, value=float(noise.get("window_s", 20.0)), help=HELP["window"], key=f"nwin_{revision}"),
            }
            c1, c2, c3, c4 = st.columns(4)
            noise_values.update(
                minimum_frequency_hz=c1.number_input("PSD 最低频率 (Hz)", min_value=0.0, value=float(noise.get("minimum_frequency_hz", 3.0)), help=HELP["psd_min"], key=f"nf0_{revision}"),
                dominant_frequency_hz=c2.number_input("PSD 主频 (Hz)", min_value=0.01, value=float(noise.get("dominant_frequency_hz", 12.0)), help=HELP["psd_dominant"], key=f"nfd_{revision}"),
                maximum_frequency_hz=c3.number_input("PSD 最高频率 (Hz)", min_value=0.01, value=float(noise.get("maximum_frequency_hz", 18.8)), help=HELP["psd_max"], key=f"nf1_{revision}"),
                dt_s=c4.number_input("采样间隔 dt (s)", min_value=0.00001, value=float(noise.get("dt_s", 0.005)), format="%.5f", help=HELP["dt"], key=f"ndt_{revision}"),
            )
            with st.expander("噪声高级参数"):
                c1, c2, c3, c4 = st.columns(4)
                noise_values.update(
                    seed=int(c1.number_input("时间随机种子", min_value=0, value=int(noise.get("seed", 314159)), step=1, help=HELP["noise_seed"], key=f"nseed_{revision}")),
                    source_amplitude=c2.number_input("单源相对振幅", min_value=0.000001, value=float(noise.get("source_amplitude", 1.0)), help=HELP["source_amplitude"], key=f"namp_{revision}"),
                    independent_sources=c3.checkbox("各源时间函数独立", value=bool(noise.get("independent_sources", True)), help=HELP["independent_sources"], key=f"nind_{revision}"),
                    event_rate_hz=c4.number_input("Poisson 事件率 (1/s)", min_value=0.000001, value=float(noise.get("event_rate_hz", 0.4)), help=HELP["event_rate"], disabled=noise_values["temporal_model"] != "poisson_ricker", key=f"ner_{revision}"),
                    overlap=0.0,
                )
                st.caption("当前独立合成窗 overlap 固定为 0；各窗由流式互相关器直接叠加。")
            if noise_values["psd_shape"] == "file":
                psd_upload = st.file_uploader(
                    "上传 PSD CSV（frequency_hz,power；必填）",
                    type=["csv"],
                    help=HELP["psd_file"],
                    key=f"psd_upload_{revision}",
                )
                st.download_button(
                    "下载 PSD CSV 模板",
                    b"frequency_hz,power\n3.0,0.0\n8.0,1.0\n18.0,0.0\n",
                    file_name="noise_psd_template.csv",
                    mime="text/csv",
                    help="下载后填入频率递增、功率非负的实测或设计 PSD。",
                )
                noise_values["psd_file"] = noise.get("psd_file")

        processing = mapping.get("processing", {})
        st.markdown("#### 频散计算范围")
        c1, c2, c3, c4 = st.columns(4)
        method_options = ["phase_shift", "hankel", "bessel"]
        processing_values = dict(processing)
        processing_values.update(
            dispersion_method=c1.selectbox("频散方法", method_options, index=_index(method_options, processing.get("dispersion_method", "phase_shift")), help=HELP["method"], key=f"pm_{revision}"),
            frequency_min_hz=c2.number_input("处理最低频率 (Hz)", min_value=0.0, value=float(processing.get("frequency_min_hz", 3.0)), help=HELP["processing_fmin"], key=f"pf0_{revision}"),
            frequency_max_hz=c3.number_input("处理最高频率 (Hz)", min_value=0.01, value=float(processing.get("frequency_max_hz", 18.0)), help=HELP["processing_fmax"], key=f"pf1_{revision}"),
            velocity_min_mps=c4.number_input("最低相速度 (m/s)", min_value=1.0, value=float(processing.get("velocity_min_mps", 150.0)), help=HELP["velocity_min"], key=f"pv0_{revision}"),
        )
        c1, c2, c3 = st.columns(3)
        processing_values.update(
            velocity_max_mps=c1.number_input("最高相速度 (m/s)", min_value=1.0, value=float(processing.get("velocity_max_mps", 500.0)), help=HELP["velocity_max"], key=f"pv1_{revision}"),
            velocity_step_mps=c2.number_input("速度步长 (m/s)", min_value=0.1, value=float(processing.get("velocity_step_mps", 2.0)), help=HELP["velocity_step"], key=f"pvs_{revision}"),
            distance_taper_fraction=c3.slider("距离域 taper", 0.0, 1.0, float(processing.get("distance_taper_fraction", 0.5)), 0.05, help=HELP["distance_taper"], key=f"pt_{revision}"),
        )

        store_responses = bool(run.get("store_source_responses", False))
        with st.expander("高级计算参数"):
            c1, c2, c3, c4 = st.columns(4)
            workers = int(c1.number_input("CPU workers", min_value=0, value=int(run.get("workers", 0)), step=1, help=HELP["workers"], key=f"workers_{revision}"))
            frequency_block = int(c2.number_input("频率块大小", min_value=1, value=int(run.get("frequency_block_size", 16)), step=1, help=HELP["frequency_block"], key=f"fb_{revision}"))
            source_block = int(c3.number_input("噪声源块大小", min_value=1, value=int(run.get("source_block_size", 16)), step=1, help=HELP["source_block"], key=f"sb_{revision}"))
            radial_step = c4.number_input("径向 Green 插值 (m)", min_value=0.001, value=float(run.get("radial_interpolation_step_m", 0.5)), format="%.3f", help=HELP["radial_step"], key=f"rs_{revision}")
            store_responses = st.checkbox(
                "保存逐噪声源完整传递响应",
                value=store_responses,
                help=HELP["store_responses"],
                key=f"store_{revision}",
                disabled=selected_mode != "passive",
            )
            c1, c2, c3, c4 = st.columns(4)
            processing_values.update(
                correlation_normalization=c1.selectbox("互相关归一化", ["spectral_whitening", "one_bit", "none"], index=_index(["spectral_whitening", "one_bit", "none"], processing.get("correlation_normalization", "spectral_whitening")), help=HELP["correlation_normalization"], key=f"pcn_{revision}"),
                zero_padding=c2.number_input("FFT 补零倍数", min_value=1.0, value=float(processing.get("zero_padding", 8.0)), help=HELP["zero_padding"], key=f"pzp_{revision}"),
                picker_smoothness=c3.number_input("拾取平滑约束", min_value=0.0, value=float(processing.get("picker_smoothness", 0.35)), help=HELP["picker_smoothness"], key=f"pps_{revision}"),
                picker_maximum_jump_mps=c4.number_input("拾取最大跳变 (m/s)", min_value=0.1, value=float(processing.get("picker_maximum_jump_mps", 30.0)), help=HELP["picker_jump"], key=f"ppj_{revision}"),
            )
            c1, c2, c3, c4 = st.columns(4)
            processing_values.update(
                radial_kernel_lookup_step=c1.number_input("径向核查表步长", min_value=0.0, value=float(processing.get("radial_kernel_lookup_step", 0.002)), format="%.4f", help=HELP["kernel_step"], key=f"pks_{revision}"),
                picker_minimum_energy=c2.slider("拾取最低能量", 0.0, 1.0, float(processing.get("picker_minimum_energy", 0.05)), 0.01, help=HELP["picker_energy"], key=f"pme_{revision}"),
                rayleigh_modes=int(c3.number_input("Rayleigh 理论模态数", min_value=1, value=int(processing.get("rayleigh_modes", 4)), step=1, help=HELP["rayleigh_modes"], key=f"prm_{revision}")),
                love_modes=int(c4.number_input("Love 理论模态数", min_value=1, value=int(processing.get("love_modes", 4)), step=1, help=HELP["love_modes"], key=f"plm_{revision}")),
            )
            processing_values["compare_methods"] = st.checkbox(
                "同时输出 phase_shift / Bessel(FJ) / Hankel 方法对照",
                value=bool(processing.get("compare_methods", False)),
                help=HELP["compare_methods"],
                key=f"pcm_{revision}",
            )

        submitted = st.button(
            "校验并绘制输入",
            type="primary",
            width="stretch",
            help="解析全部参数、绘制模型/geometry/源谱/理论曲线并执行采样、孔径和混叠检查；不运行波场正演。",
        )

    if not submitted:
        return
    try:
        candidate = deepcopy(mapping)
        candidate["run"] = {
            "name": name.strip(), "mode": selected_mode,
            "output_directory": output_directory,
            "workers": workers, "frequency_block_size": frequency_block,
            "source_block_size": source_block,
            "radial_interpolation_step_m": radial_step,
            "store_source_responses": bool(store_responses),
        }
        candidate["model"] = {"layers": _normalize_layers(edited_layers)}
        if receiver_layout == "file" and receiver_upload is not None:
            coordinate_csv_count(receiver_upload.getvalue())
            staged = session.stage_upload(receiver_upload.name, receiver_upload.getvalue(), kind="coordinates")
            receiver_values["coordinates_file"] = str(staged)
        if receiver_layout == "file" and not receiver_values.get("coordinates_file"):
            raise CaseInputError("请选择接收器坐标 CSV")
        candidate["geometry"] = {"receivers": receiver_values}
        candidate["processing"] = processing_values
        if selected_mode == "active":
            if shot_input_mode == "file" and shot_upload is None:
                raise CaseInputError("请选择炮点 Shot CSV")
            if shot_parse_error is not None:
                raise CaseInputError(shot_parse_error)
            candidate["active_source"] = active_values
            candidate.pop("ambient_sources", None)
            candidate.pop("noise", None)
        else:
            if ambient_values["layout"] == "file" and ambient_upload is not None:
                ambient_values["count"] = coordinate_csv_count(ambient_upload.getvalue())
                staged = session.stage_upload(ambient_upload.name, ambient_upload.getvalue(), kind="coordinates")
                ambient_values["coordinates_file"] = str(staged)
            if ambient_values["layout"] == "file" and not ambient_values.get("coordinates_file"):
                raise CaseInputError("请选择噪声源坐标 CSV")
            if noise_values["psd_shape"] == "file" and psd_upload is not None:
                staged = session.stage_upload(psd_upload.name, psd_upload.getvalue(), kind="psd")
                noise_values["psd_file"] = str(staged)
            if noise_values["psd_shape"] == "file" and not noise_values.get("psd_file"):
                raise CaseInputError("请选择 PSD CSV")
            candidate["ambient_sources"] = ambient_values
            candidate["noise"] = noise_values
            candidate.pop("active_source", None)
        config, invalidated = session.apply_mapping(candidate)
        review = session.run_stage("review")
        st.session_state.case_mapping = candidate
        st.session_state.last_message = (
            f"算例 {config.name} 校验完成；失效阶段：{', '.join(invalidated) or '无'}。"
        )
        st.success(st.session_state.last_message)
        if not review.ready:
            _render_required_failures(
                st,
                [
                    check
                    for check in review.checks
                    if check.level == "required" and not check.passed
                ],
            )
    except (CaseInputError, ConfigError, OSError, RuntimeError, ValueError) as exc:
        _render_input_error(st, exc)


def _review_tab(st, session: CaseSession) -> None:
    if session.workflow is None:
        st.info("先在“1 建立算例”点击“校验并绘制输入”。")
        return
    review = session.workflow.review_inputs()
    required_failures = [c for c in review.checks if c.level == "required" and not c.passed]
    warnings = [c for c in review.checks if c.level != "required" and not c.passed]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("接收器", review.diagnostics["receiver_count"])
    c2.metric("源", review.diagnostics["source_count"])
    c3.metric("阵列孔径", f"{review.diagnostics['receiver_aperture_m']:.1f} m")
    c4.metric("运行状态", "READY" if review.ready else "BLOCKED")
    figure = plot_preflight_review(review)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    rows = [
        {
            "状态": "PASS" if check.passed else ("FAIL" if check.level == "required" else "WARN"),
            "检查": check.name, "当前值": check.value,
            "判据": check.criterion, "级别": check.level,
        }
        for check in review.checks
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    if required_failures:
        _render_required_failures(st, required_failures)
        st.caption("运行按钮保持锁定，直到所有 required 项通过。")
    elif warnings:
        st.warning(f"必需检查通过，但有 {len(warnings)} 个建议项需要人工判断。")
    else:
        st.success("输入与自动检查全部通过，可以开始正演。")


def _call(st, label: str, action) -> bool:
    try:
        with st.spinner(label):
            action()
        st.session_state.last_message = f"{label}完成"
        st.success(f"{label}完成")
        st.rerun()
        return True
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        _render_input_error(st, exc)
        return False


def _run_tab(st, session: CaseSession) -> None:
    workflow = session.workflow
    if workflow is None:
        st.info("请先建立并校验算例。")
        return
    review = workflow.review_inputs()
    blocked = not review.ready
    st.caption("阶段结果保存在当前会话内；只修改下游参数时不会重算安全的上游阶段。")
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button(
        "1 正演/传递函数",
        disabled=blocked,
        width="stretch",
        help="主动：计算完整炮集；被动：计算源—接收器频率域 Green 传递函数。",
    ):
        _call(st, "正演/传递函数计算", workflow.compute_forward)
    if c2.button(
        "2 观测记录/NCF",
        disabled=blocked,
        width="stretch",
        help="主动：复用炮集；被动：混合全部有限时长随机源，再仅用接收器波形计算 NCF。",
    ):
        _call(st, "观测记录与互相关计算", workflow.simulate_observations)
    if c3.button(
        "3 频散与拾取",
        disabled=blocked,
        width="stretch",
        help="按选择的 phase_shift/Hankel/Bessel 方法成像，并叠加理论曲线和自动拾取。",
    ):
        _call(st, "频散能量与拾取", workflow.compute_dispersion)
    if c4.button(
        "4 保存结果",
        disabled=blocked,
        width="stretch",
        help="保存 HDF5、CSV、TOML、manifest 和包含模型/geometry/频散的汇总图。",
    ):
        _call(st, "结果保存", workflow.save_results)
    if c5.button(
        "一键运行全部",
        disabled=blocked,
        type="primary",
        width="stretch",
        help="按 1→2→3→4 顺序完整运行；已有且依赖安全的上游结果会自动复用。",
    ):
        _call(st, "完整流程", lambda: workflow.run_all(save=True))
    st.dataframe(workflow.status(), hide_index=True, width="stretch")
    if workflow.shot is not None or workflow.transfer is not None:
        st.markdown("#### 正演结果")
        figure = plot_workflow_forward(workflow)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    if workflow.config.mode == "passive" and workflow.correlation is not None:
        st.markdown("#### 混合记录与接收器互相关")
        figure = plot_workflow_observations(workflow)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    if workflow.active_analysis is not None or workflow.passive_analysis is not None:
        st.markdown("#### 频散能量、理论曲线与自动拾取")
        figure = plot_workflow_dispersion(workflow)
        st.pyplot(figure, width="stretch")
        plt.close(figure)


def _results_tab(st, session: CaseSession) -> None:
    workflow = session.workflow
    if workflow is None:
        st.info("尚无算例结果。")
        return
    portable_config = None
    if workflow.artifacts is not None:
        saved_config = workflow.artifacts.output_directory / "input.toml"
        if saved_config.is_file():
            portable_config = saved_config.read_bytes()
    st.download_button(
        "下载当前完整 TOML",
        portable_config or session.config_toml.encode("utf-8"),
        file_name=f"{workflow.config.name}.toml",
        mime="application/toml",
        width="stretch",
    )
    if workflow.artifacts is None:
        st.info("尚未保存最终结果；可先查看当前阶段状态，或回到运行页点击“保存结果”。")
        return
    artifacts = workflow.artifacts
    st.success(f"结果目录：{artifacts.output_directory}")
    if artifacts.summary_png.is_file():
        st.image(str(artifacts.summary_png), caption="最终汇总：模型、密度、geometry、波形/NCF 与频散")
    for filename, caption in (
        ("active_components.png", "主动 RR/ZR/RZ/ZZ 分量敏感性与联合 Rayleigh"),
        ("passive_components.png", "被动 ZZ/RR 与联合 Rayleigh"),
        ("active_method_comparison.png", "主动 phase-shift / Bessel(FJ) / Hankel 对照"),
        ("passive_method_comparison.png", "被动 phase-shift / Bessel(FJ) / Hankel 对照"),
    ):
        path = artifacts.output_directory / filename
        if path.is_file():
            st.image(str(path), caption=caption)
    st.code(
        "\n".join(
            (
                f"HDF5: {artifacts.results_h5}",
                f"Summary: {artifacts.summary_png}",
                f"Manifest: {artifacts.manifest_json}",
            )
        )
    )
    for filename in (
        "picked_rayleigh.csv",
        "picked_rayleigh_combined.csv",
        "picked_love.csv",
        "convergence.csv",
        "manifest.json",
    ):
        path = artifacts.output_directory / filename
        if path.is_file():
            st.download_button(
                f"下载 {filename}", path.read_bytes(), file_name=filename,
                mime="text/csv" if path.suffix == ".csv" else "application/json",
                key=f"download_{filename}", width="stretch",
            )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="WaveDispX", page_icon="🌊", layout="wide")
    session = _ensure_state(st)
    st.title("WaveDispX 波场正演与频散分析")
    st.caption("主动/被动源 · 体波/面波 · 3C 波场 · 噪声互相关 · 多方法频散成像")
    st.info(st.session_state.last_message)
    build_tab, review_tab, run_tab, result_tab, method_tab = st.tabs(
        ["1 建立算例", "2 输入审核", "3 运行", "4 结果与导出", "方法说明"]
    )
    with build_tab:
        _load_controls(st, session)
        _case_form(st, session)
    with review_tab:
        _review_tab(st, session)
    with run_tab:
        _run_tab(st, session)
    with result_tab:
        _results_tab(st, session)
    with method_tab:
        st.markdown(
            r"""
            ### 计算关系

            主动源在频率域计算 $u_i(\omega)=G_{is}(\omega)S(\omega)$；被动源先形成
            $u_i(t)=\sum_s G_{is}*q_s(t)$ 的有限时长混合记录，再仅从接收器波形计算
            $C_{ij}(\tau)=\langle u_i(t)u_j(t+\tau)\rangle$。处理阶段没有源编号，也不做
            “逐源理想互相关”捷径。

            被动默认使用相移叠加
            $E(f,c)=|\sum_r \hat C(f,r)e^{i2\pi fr/c}|$；Bessel/Hankel 是可选敏感性
            对照。理论 Rayleigh/Love 曲线由相同分层 $V_P,V_S,\rho$ 独立计算并叠加显示。

            `three_component` 被动源会把全局 Fx/Fy/Fz 按每个源—接收器方位分解，通过
            RR/TT/ZR/RZ/ZZ Green 张量传播并旋回 X/Y/Z；台站互相关再按每一台站对方位
            形成 RR/TT/ZZ。这不是频散图后的能量拼接。

            **当前稳定边界：** 水平分层、各向同性、地表源/接收器；竖直快速路径或物理
            三分量随机力。二维阵列频散当前按台间距径向平均，不保留方位各向异性信息。
            """
        )


if __name__ == "__main__":
    main()
