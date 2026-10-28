"""Compact CPU experiment studio; existing workbench remains available."""
from copy import deepcopy
from pathlib import Path
import sys

from wavedispx.classroom.qt_runtime import configure_qt
configure_qt()

import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, QTimer, QSize, QUrl
from PySide6.QtGui import QShortcut, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QSplitter, QPushButton, QLabel, QLineEdit,
    QCheckBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QScrollArea, QMenu, QGroupBox, QProgressBar, QListWidget, QListWidgetItem, QDialogButtonBox, QTextBrowser, QFileDialog)

from .app import MainWindow, FigurePane, TEMPLATES, WorkflowWorker, PROJECT_ROOT
from wavedispx.app_service import load_template_mapping
from .help_text import parameter_detail_html, parameter_spec, parameter_choices
from wavedispx.plotting import _layer_steps
from wavedispx.workflow_plotting import plot_workflow_observations
from wavedispx.config import config_from_mapping
from .studio_style import STUDIO_STYLE
from wavedispx.experiment import ExperimentRequest, run_experiment, save_experiment, wave_unavailable, theory_context


class StudioFigure(FigurePane):
    """Small navigation controls that do not crowd the plots."""

    def set_figure(self, figure):
        super().set_figure(figure)
        self._toolbar.setIconSize(QSize(16, 16))
        self._toolbar.setStyleSheet("QToolButton { padding: 2px; border: none; }")
        self._toolbar.setMaximumHeight(30)

    def clear(self):
        # deleteLater alone leaves the old canvas visible during a long preview.
        for widget in (self._canvas, self._toolbar):
            if widget is not None:
                widget.hide()
        super().clear()


class StudioWindow(MainWindow):
    """One-screen shell; all numerical work stays in the shared workflow."""

    def __init__(self, template_path=None, parent=None):
        # Opening the UI never starts a long passive experiment by default.
        self._starting_stage = False
        self.experiment_result = None
        super().__init__(template_path or TEMPLATES["主动源｜快速检查"], parent)

    def _build_ui(self):
        super()._build_ui()
        self.legacy = self.takeCentralWidget()
        self.legacy.setParent(self)
        self.legacy.hide()
        self.dialogs = []
        self._page_dialogs = {}
        self.setWindowTitle("WaveDispX Studio · CPU · QED")
        self.setStyleSheet(STUDIO_STYLE)
        self.setMinimumSize(1180, 760)
        central = QWidget()
        central.setObjectName("studioRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)
        header = QWidget()
        header.setObjectName("studioHeader")
        header_row = QHBoxLayout(header)
        brand = QLabel("WaveDispX  Studio")
        brand.setObjectName("studioBrand")
        header_row.addWidget(brand)
        subtitle = QLabel("波场正演与频散实验  /  CPU · QED")
        subtitle.setObjectName("studioSubtitle")
        header_row.addWidget(subtitle)
        header_row.addStretch()
        full = QPushButton("全屏 / 退出")
        full.setToolTip("可选全屏；默认是适配屏幕的普通窗口。Esc 返回普通窗口，不自动最大化。")
        full.clicked.connect(lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())
        self.examples_button = QPushButton("示例与说明")
        examples_menu = QMenu(self.examples_button)
        examples_menu.addAction("操作说明 / 波型限制 / CSV格式", self.show_studio_guide)
        examples_menu.addAction("自定义源 / 台站文件说明", lambda: self.show_studio_guide(PROJECT_ROOT / "examples/custom_geometry/README_CN.md"))
        examples_menu.addAction("打开自定义输入示例文件夹", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(PROJECT_ROOT / "examples/custom_geometry"))))
        examples_menu.addSeparator()
        for title, filename in (("主动倾斜力 · Rayleigh＋Love", "active_vector_studio.toml"),
                                ("被动三分量 · R/L短记录功能例", "passive_vector_energy_studio.toml"),
                                ("主动炮点 / 接收器CSV例", "observation_system/active_selected_shot.toml"),
                                ("被动位置 / Poisson时间例", "observation_system/passive_poisson.toml"),
                                ("可选HVSR · 短记录功能例", "passive_hvsr_studio.toml"),
                                ("非均匀源 · 东侧源＋线阵", "custom_geometry/passive_east_line.toml"),
                                ("非均匀源 · 双源簇＋不规则面阵", "custom_geometry/passive_clusters_array.toml"),
                                ("非均匀源 · 双源簇活动强度切换", "custom_geometry/passive_switching_array.toml")):
            examples_menu.addAction(title, lambda checked=False, filename=filename: self.load_studio_example(filename))
        self.examples_button.setMenu(examples_menu)
        self.examples_button.setToolTip("载入示例会替换当前输入，不自动计算；请先保存当前参数。短记录例不代表收敛。")
        header_row.addWidget(self.examples_button)
        header_row.addWidget(full)
        root.addWidget(header)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("模板库 · 选择后载入"))
        self.template_combo.setMinimumWidth(240)
        self.load_template_button.setText("载入模板")
        self.load_template_button.setToolTip("将所选模板载入为当前输入，会替换整套参数；仅选择下拉项不会改变当前算例。请先保存需要保留的配置。")
        self.save_button.setText("保存参数…")
        for widget in (self.template_combo, self.load_template_button,
                       self.open_button, self.save_button):
            bar.addWidget(widget)
        bar.addWidget(QLabel("计算目标"))
        self.product_combo = QComboBox()
        for raw, label in (("legacy", "频散＋理论（兼容流程）"), ("energy_theory", "频散能量＋理论"), ("energy", "仅频散能量"), ("theory", "仅QED理论曲线")):
            self.product_combo.addItem(label, raw)
        self.product_combo.setToolTip("日常频散计算目标。HVSR默认关闭，需要时在‘附加分析’中显式启用。仅理论使用独立的理论df。")
        self.product_combo.currentIndexChanged.connect(self._request_changed)
        bar.addWidget(self.product_combo)
        self.waves_combo = QComboBox()
        for raw, label in (("both", "Rayleigh＋Love"), ("rayleigh", "Rayleigh"), ("love", "Love")):
            self.waves_combo.addItem(label, raw)
        self.waves_combo.setToolTip("选择输出波型，不会自动改变激发源。未选择、不支持或无响应时结果页仍保留原因。HVSR不按R/L分离。")
        self.waves_combo.currentIndexChanged.connect(self._request_changed)
        bar.addWidget(self.waves_combo)
        bar.addStretch()
        self.case_badge = QLabel("当前输入")
        self.case_badge.setObjectName("caseBadge")
        bar.addWidget(self.case_badge)
        root.addLayout(bar)
        flow = QHBoxLayout()
        guide = QLabel("编辑输入  →")
        guide.setObjectName("sectionTitle")
        flow.addWidget(guide)
        self.preview_button = QPushButton("1  检查与预览")
        self.preview_button.setProperty("role", "secondary")
        self.preview_button.setToolTip("应用当前输入并检查 required 项；更新模型和观测系统预览，不运行波场正演。")
        self.preview_button.clicked.connect(lambda: self.validate_case(navigate=False))
        flow.addWidget(self.preview_button)
        flow.addWidget(QLabel("→"))
        self.compute_button = QPushButton("2  运行计算 · CPU")
        self.compute_button.setMinimumWidth(210)
        self.compute_button.setProperty("role", "primary")
        self.compute_button.setToolTip("计算波形/相关/频散，结果先留内存；不会自动保存大型波形。理论曲线只调用QED，失败不切换后端。")
        self.compute_button.clicked.connect(lambda: self.start_stage("analysis"))
        flow.addWidget(self.compute_button)
        flow.addWidget(QLabel("→"))
        self.export_button = QPushButton("3  保存结果")
        self.export_button.setProperty("role", "save")
        self.export_button.setToolTip("仅保存当前输入对应的已计算结果到配置的输出目录；参数有改动时必须先重新计算。")
        self.export_button.clicked.connect(lambda: self.start_stage("save"))
        flow.addWidget(self.export_button)
        flow.addStretch()
        self.review_button = QPushButton("检查详情")
        self.review_button.clicked.connect(lambda: self.select_page(1))
        flow.addWidget(self.review_button)
        logs = QPushButton("运行日志")
        logs.clicked.connect(lambda: self.select_page(2))
        flow.addWidget(logs)
        root.addLayout(flow)
        escape = QShortcut(QKeySequence("Escape"), self)
        escape.activated.connect(lambda: self.showNormal() if self.isFullScreen() else None)
        input_card = QWidget()
        input_card.setObjectName("inputCard")
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(10, 7, 10, 8)
        actions = QHBoxLayout()
        title = QLabel("实验输入")
        title.setObjectName("sectionTitle")
        actions.addWidget(title)
        hint = QLabel("按顺序设置；参数悬停查看中文解释；修改后重新检查。")
        hint.setObjectName("hint")
        actions.addWidget(hint)
        actions.addStretch()
        self.import_button = QPushButton("导入 CSV…")
        self.import_button.setToolTip("导入接收器、DAS通道、炮点或噪声源/PSD文件；具体入口随主动/被动模式变化。")
        actions.addWidget(self.import_button)
        for label, action in (("自动建议", lambda: self._parameter_action("apply_auto_design")),
                              ("高级参数 / 文件", lambda: self.select_page(0))):
            button = QPushButton(label)
            if label == "自动建议":
                button.setText("自动建议 / 重置")
                menu = QMenu(button)
                menu.addAction("填入候选参数（仍需检查收敛）", action)
                menu.addAction("恢复所选模板的自动参数默认值", lambda: self._parameter_action("reset_auto_design"))
                button.setMenu(menu)
                self.auto_button = button
                button.setToolTip("更新 f/v 范围、df/dv 和记录/lag 候选值；保留模型、源位置和数量。填入后仍可手改，输入审核不等于科学收敛验收。")
            else:
                button.clicked.connect(action)
            actions.addWidget(button)
        self.extra_analysis_button = QPushButton("附加分析")
        extra_menu = QMenu(self.extra_analysis_button)
        self.extra_analysis_actions = {}
        for key,label in (("off","关闭附加分析（默认）"),("hvsr","仅计算节点HVSR"),("energy_hvsr","频散＋理论，并计算节点HVSR")):
            action = extra_menu.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, key=key: self._set_extra_analysis(key))
            self.extra_analysis_actions[key] = action
        self.extra_analysis_button.setMenu(extra_menu)
        self.extra_analysis_button.setToolTip("默认不计算HVSR，也不显示其参数和结果页。仅在需要原始XYZ谱比时启用；不自动修改源分量。")
        self._base_product_before_hvsr = "legacy"
        actions.addWidget(self.extra_analysis_button)
        self.parameters_toggle = QPushButton("收起参数 ▴")
        self.parameters_toggle.setCheckable(True)
        self.parameters_toggle.setToolTip("收起输入区以放大结果；不会丢失参数或重新计算。再次点击展开。")
        self.parameters_toggle.toggled.connect(self._toggle_parameters)
        actions.addWidget(self.parameters_toggle)
        input_layout.addLayout(actions)
        self.compact = QTabWidget()
        self.compact.setFixedHeight(180)
        input_layout.addWidget(self.compact)
        root.addWidget(input_card)
        state_row = QHBoxLayout()
        self.state_chip = QLabel("待检查")
        self.state_chip.setObjectName("stateChip")
        state_row.addWidget(self.state_chip)
        self.notice = QLabel("尚未计算。先检查输入与观测系统。")
        self.notice.setWordWrap(True)
        state_row.addWidget(self.notice, 1)
        root.addLayout(state_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        root.addWidget(self.progress)
        self.sampling_summary = QLabel("记录时长与计算网格：输入审核后显示。")
        self.sampling_summary.setObjectName("samplingSummary")
        self.sampling_summary.setWordWrap(True)
        self.sampling_summary.setToolTip("扫描 df / 零填充只改变频率采样，不增加有限记录的独立频率分辨率。被动总时长、独立窗长、NCF lag 是不同量。")
        root.addWidget(self.sampling_summary)
        result_controls = QHBoxLayout()
        title = QLabel("模型 / 观测系统与计算结果")
        title.setObjectName("sectionTitle")
        result_controls.addWidget(title)
        result_controls.addStretch()
        hint = QLabel("以下仅改变显示，不重算")
        hint.setObjectName("hint")
        result_controls.addWidget(hint)
        self.overlay = QCheckBox("叠加 QED 曲线")
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(self.render_results)
        result_controls.addWidget(self.overlay)
        self.wave_toggle = QCheckBox("显示波形 / NCF")
        self.wave_toggle.toggled.connect(self.show_waveforms)
        result_controls.addWidget(self.wave_toggle)
        self.hvsr_station = QComboBox()
        self.hvsr_station.setToolTip("逐台站HVSR，不能把所有接收器默认平均成一条曲线。")
        self.hvsr_station.currentIndexChanged.connect(self.render_results)
        self.hvsr_station.hide()
        root.addLayout(result_controls)
        self.hvsr_controls = QWidget()
        hvrow = QHBoxLayout(self.hvsr_controls)
        hvrow.setContentsMargins(0,0,0,0)
        hvrow.setSpacing(12)
        hvrow.addWidget(QLabel("HVSR视图"))
        self.hvsr_view = QComboBox()
        for label, value in (("单站曲线", "single"), ("多站曲线叠加", "overlay"), ("全台站热图", "heatmap"), ("分量谱 / 方向诊断", "diagnostic")):
            self.hvsr_view.addItem(label, value)
        self.hvsr_view.setToolTip("只读显示已有HVSR，不重新正演；热图共用真实H/V色标，不对各站归一化。")
        self.hvsr_view.currentIndexChanged.connect(self._hvsr_view_changed)
        hvrow.addWidget(self.hvsr_view)
        hvrow.addWidget(self.hvsr_station)
        self.hvsr_select = QPushButton("选择叠加台站…")
        self.hvsr_select.clicked.connect(self._choose_hvsr_stations)
        self.hvsr_select.setToolTip("勾选最多12站叠加；所有台站请看热图。台站编号按输入顺序从0开始。")
        hvrow.addWidget(self.hvsr_select)
        self.hvsr_scatter = QCheckBox("显示窗间波动")
        self.hvsr_scatter.setChecked(True)
        self.hvsr_scatter.setToolTip("±1个对数标准差；表示窗间离散度，不是置信区间。")
        self.hvsr_scatter.toggled.connect(self.render_results)
        hvrow.addWidget(self.hvsr_scatter)
        hvrow.addWidget(QLabel("点Geometry选单站；Ctrl/⌘＋点击增减叠加站"))
        hvrow.addStretch()
        self.hvsr_controls.hide()
        self._hvsr_selected = []
        root.addWidget(self.hvsr_controls)
        split = QSplitter(Qt.Orientation.Horizontal)
        context = QSplitter(Qt.Orientation.Vertical)
        self.model_panel = StudioFigure("模型预览")
        self.geometry_panel = StudioFigure("Geometry预览")
        context.addWidget(self.model_panel)
        context.addWidget(self.geometry_panel)
        split.addWidget(context)
        self.result_tabs = QTabWidget()
        self.rayleigh_panel = StudioFigure("检查并计算后显示实际频散结果")
        self.love_panel = StudioFigure("当前未计算")
        self.wave_panel = StudioFigure("当前未计算波形")
        self.hvsr_panel = StudioFigure("未选择节点HVSR。需要被动XYZ原始记录，不使用NCF或DAS轴向应变代替。")
        self.result_tabs.addTab(self.rayleigh_panel, "Rayleigh / DAS")
        self.result_tabs.addTab(self.love_panel, "Love")
        self.result_tabs.currentChanged.connect(self.render_results)
        split.addWidget(self.result_tabs)
        split.setSizes([370, 1050])
        root.addWidget(split, 1)
        self.setCentralWidget(central)
        self._set_ui_state("dirty")

    def _toggle_parameters(self, collapsed):
        self.compact.setVisible(not collapsed)
        self.parameters_toggle.setText("展开参数 ▾" if collapsed else "收起参数 ▴")

    def _set_ui_state(self, state):
        """Operational state only; green never certifies numerical convergence."""
        labels = {"dirty": "● 待检查", "ready": "● 可计算", "running": "● 处理中",
                  "result": "● 结果可查看", "error": "● 需要处理"}
        self.state_chip.setText(labels[state])
        self.state_chip.setProperty("tone", state)
        self.state_chip.style().unpolish(self.state_chip)
        self.state_chip.style().polish(self.state_chip)
        busy = state == "running"
        for widget in (self.preview_button, self.auto_button, self.compute_button,
                       self.compact, self.load_template_button, self.open_button,
                       self.template_combo, self.save_button, self.parameter_page, self.import_button):
            widget.setEnabled(not busy)
        self.export_button.setEnabled(state == "result")
        self.progress.setVisible(busy)
        self.compute_button.setText("CPU 处理中…" if busy else "2  检查并运行 · CPU" if state == "dirty" else "2  运行计算 · CPU")
        self.product_combo.setEnabled(not busy)
        self.waves_combo.setEnabled(not busy)
        self.extra_analysis_button.setEnabled(not busy)
        self.examples_button.setEnabled(not busy)

    def _set_extra_analysis(self, product):
        if self._running:
            return
        from dataclasses import asdict
        mapping = self.parameter_page.mapping()
        request = ExperimentRequest.from_mapping(mapping)
        if "hvsr" not in request.product:
            self._base_product_before_hvsr = request.product
        target = self._base_product_before_hvsr if product == "off" else product
        if target == request.product:
            self.rebuild_controls()
            return
        mapping["experiment"] = {**asdict(request), "product": target}
        self.parameter_page.set_mapping(mapping)
        self._mark_dirty()
        self.rebuild_controls()

    def _request_changed(self, *_):
        if not hasattr(self, "compact") or self._running:
            return
        mapping = self.parameter_page.mapping()
        from dataclasses import asdict
        request = asdict(ExperimentRequest.from_mapping(mapping))
        request.update(product=self.product_combo.currentData(), waves=self.waves_combo.currentData())
        mapping["experiment"] = request
        self.parameter_page.set_mapping(mapping)
        self._mark_dirty()
        self.rebuild_controls()

    def _load_mapping(self, mapping, base, message):
        try:
            mapping = deepcopy(mapping)
            mapping.setdefault("processing", {})["theory_solver"] = "qedisp"
            # Reject incomplete files before destroying the usable current case.
            request = ExperimentRequest.from_mapping(mapping)
            if request.product == "theory":
                theory_context(mapping, base)
                if not mapping.get("run", {}).get("mode") in {"active", "passive"} or "receivers" not in mapping.get("geometry", {}):
                    raise ValueError("Studio参数编辑器需要完整模板（含run.mode和geometry.receivers）。请载入任一模板后选择“仅理论曲线”；这些观测参数不会参与理论计算。最小理论输入可通过Python theory_context接口执行。")
            else:
                config_from_mapping(mapping, base_directory=base)
        except Exception as exc:
            self._show_error("输入文件无效；当前算例未替换", exc)
            return
        super()._load_mapping(mapping, base, message)
        self._base_product_before_hvsr = request.product if "hvsr" not in request.product else "legacy"
        self.rebuild_controls()
        self.clear_result_views()
        self._set_ui_state("dirty")
        self.notice.setText(message + " · 请检查当前输入")

    def clear_result_views(self):
        self.experiment_result = None
        if hasattr(self, "hvsr_station"):
            self.hvsr_station.hide()
            self.hvsr_controls.hide()
            self._hvsr_selected = []
        for panel in (self.rayleigh_panel, self.love_panel, self.wave_panel, self.hvsr_panel):
            panel.show_placeholder("参数已变化；重新计算后显示，避免旧结果与新输入混用。")

    def _mark_dirty(self):
        super()._mark_dirty()
        if hasattr(self, "notice"):
            self._set_ui_state("dirty")
            self.notice.setText("参数已修改 · 当前结果失效 · 点击更新预览或检查并计算")
            self.clear_result_views()
            self.model_panel.show_placeholder("输入已修改；检查后更新模型预览。")
            self.geometry_panel.show_placeholder("输入已修改；检查后更新观测系统预览。")
            self.sampling_summary.setText("参数已修改：时长和网格摘要待重新审核。")

    def set_value(self, path, text, original):
        try:
            if self._running:
                raise ValueError("计算中不能修改输入；请等待当前阶段完成。")
            if path == ("geometry", "receivers", "layout"):
                self._parameter_action({"line": "receiver_line", "grid": "receiver_grid", "file": "receiver_file"}[text])
                return
            if path == ("ambient_sources", "layout"):
                self._parameter_action({"annulus": "source_annulus", "line": "source_line", "file": "source_file"}[text])
                return
            if isinstance(original, bool) and text.lower() not in {"true", "false"}:
                raise ValueError("布尔参数只能为true或false")
            integer_fields = {("ambient_sources", "count"), ("ambient_sources", "seed"),
                              ("noise", "seed"), ("noise", "receiver_noise_seed"),
                              ("processing", "rayleigh_modes"), ("processing", "love_modes"),
                              ("run", "workers"), ("run", "frequency_block_size"), ("run", "source_block_size")}
            value = (text.lower() == "true" if isinstance(original, bool)
                     else int(text) if path in integer_fields
                     else float(text) if isinstance(original, (float, int)) else text)
            if path in {("run", "workers"), ("run", "frequency_block_size"), ("run", "source_block_size")}:
                minimum = 0 if path[-1] == "workers" else 1
                if value < minimum:
                    raise ValueError(f"{'.'.join(path)} 必须为整数且≥{minimum}；workers=0表示自动。")
            mapping = self.parameter_page.mapping()
            target = mapping
            for key in path[:-1]:
                target = target[key]
            if target.get(path[-1]) == value:
                return
            target[path[-1]] = value
            if path == ("active_source", "component") and value == "vector":
                for key, default in (("force_amplitude", 1.), ("force_azimuth_deg", 0.), ("force_inclination_deg", 90.)):
                    target.setdefault(key, default)
            if path == ("active_source", "component") and value != "vector":
                for key in ("force_amplitude", "force_azimuth_deg", "force_inclination_deg"):
                    target.pop(key, None)
            if path == ("ambient_sources", "component") and value == "three_component":
                target.setdefault("polarization", "legacy_xyz")
            if path == ("ambient_sources", "polarization"):
                if value == "legacy_xyz":
                    target.pop("horizontal_power_fraction", None)
                    target.pop("horizontal_azimuth_deg", None)
                else:
                    target.setdefault("horizontal_power_fraction", 2./3.)
                    if value == "directed_hv":
                        target.setdefault("horizontal_azimuth_deg", 0.)
                    else:
                        target.pop("horizontal_azimuth_deg", None)
            mapping["processing"]["theory_solver"] = "qedisp"
            self.parameter_page.set_mapping(mapping)
            self._mark_dirty()
            if path in {("noise", "temporal_model"), ("das", "enabled"), ("active_source", "component"),
                        ("ambient_sources", "component"), ("ambient_sources", "polarization")}:
                self.rebuild_controls()
        except Exception as exc:
            self._show_error("参数格式错误：" + ".".join(map(str, path)), exc)
            # Never leave invalid text on screen while using an older value.
            self.rebuild_controls()

    def _parameter_action(self, action):
        if self._running:
            return
        if action in {"activity_file", "clear_activity_file"}:
            mapping = self.parameter_page.mapping()
            if mapping["run"]["mode"] != "passive":
                return
            try:
                source = mapping.setdefault("ambient_sources", {})
                if action == "clear_activity_file":
                    source.pop("activity_file", None)
                else:
                    filename, _ = QFileDialog.getOpenFileName(self, "选择逐窗源活动CSV", str(PROJECT_ROOT / "examples/custom_geometry"), "CSV (*.csv)")
                    if not filename:
                        return
                    source["activity_file"] = str(Path(filename).resolve())
                    from wavedispx.source_activity import load_source_activity_matrix
                    config = config_from_mapping(mapping, base_directory=self.session.workspace_directory)
                    load_source_activity_matrix(config.ambient_sources, int(round(config.noise.duration_s/config.noise.window_s)))
                self.parameter_page.set_mapping(mapping)
                self._mark_dirty()
                self.rebuild_controls()
                self.notice.setText("源活动表已更新；source_index按坐标文件行顺序从0编号。请检查位置、窗数与振幅倍数；未列项为0。" if action == "activity_file" else "已移除源活动表；所有源恢复每窗振幅倍数1。请重新检查。")
            except Exception as exc:
                self._show_error("源活动表无效；原配置未替换", exc)
            return
        super()._parameter_action(action)
        self.rebuild_controls()
        if action in {"source_annulus", "source_line", "source_file"}:
            self.notice.setText("源布局操作：保留源极化、H/V功率、方向和活动文件。请检查新位置；有activity文件时须核对source_index与新坐标行顺序及窗数，再计算。")

    def show_windowed(self):
        """Use logical screen pixels, leaving room for macOS chrome and Dock."""
        area = self.screen().availableGeometry()
        width, height = min(1600, int(area.width()*.94)), min(960, int(area.height()*.92))
        self.setMinimumSize(min(1180, width), min(760, height))
        self.showNormal()
        self.resize(width, height)
        self.move(area.x()+(area.width()-width)//2, area.y()+(area.height()-height)//2)

    def load_studio_example(self, filename):
        if self._running:
            return
        path = PROJECT_ROOT / "examples" / filename
        self._load_mapping(load_template_mapping(path), path.parent, "已载入示例："+filename)
        self.validate_case(navigate=False)

    def show_studio_guide(self, path=None):
        dialog = QDialog(self)
        dialog.setWindowTitle("WaveDispX · 示例、输入方式与计算边界")
        dialog.resize(min(1000, self.width()), min(780, self.height()))
        browser = QTextBrowser()
        path = path if isinstance(path, Path) else PROJECT_ROOT / "docs/studio/README_zh.md"
        browser.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent)+"/"))
        browser.setMarkdown(path.read_text(encoding="utf-8"))
        browser.setOpenExternalLinks(True)
        layout = QVBoxLayout(dialog)
        layout.addWidget(browser)
        self.dialogs.append(dialog)
        dialog.show()

    def rebuild_controls(self):
        selected = self.compact.currentIndex()
        while self.compact.count():
            page = self.compact.widget(0)
            self.compact.removeTab(0)
            page.deleteLater()
        mapping = self.parameter_page.mapping()
        request = ExperimentRequest.from_mapping(mapping)
        self.product_combo.blockSignals(True)
        for i in range(self.product_combo.count()-1,-1,-1):
            if "hvsr" in str(self.product_combo.itemData(i)):
                self.product_combo.removeItem(i)
        if "hvsr" in request.product:
            self.product_combo.addItem("仅节点HVSR" if request.product == "hvsr" else "频散＋理论＋HVSR", request.product)
        self.product_combo.blockSignals(False)
        for key,action in self.extra_analysis_actions.items():
            action.setChecked(key == (request.product if "hvsr" in request.product else "off"))
        self.extra_analysis_button.setText("附加分析 · HVSR已启用" if "hvsr" in request.product else "附加分析")
        for combo, value in ((self.product_combo, request.product), (self.waves_combo, request.waves)):
            combo.blockSignals(True)
            index = combo.findData(value)
            if index < 0:
                combo.addItem(value, value)
                index = combo.findData(value)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        hv_index = self.result_tabs.indexOf(self.hvsr_panel)
        if "hvsr" in request.product and hv_index < 0:
            self.result_tabs.addTab(self.hvsr_panel, "节点 HVSR")
        elif "hvsr" not in request.product and hv_index >= 0:
            self.result_tabs.removeTab(hv_index)
        mode_label = "主动源" if mapping["run"]["mode"] == "active" else "被动源"
        layout_label = {"line": "线阵", "grid": "面阵", "file": "文件观测系统"}.get(mapping["geometry"]["receivers"]["layout"], "自定义")
        self.case_badge.setText(f"当前输入：{mode_label} · {layout_label} · {'DAS' if mapping.get('das', {}).get('enabled') else '节点'}")
        old_menu = self.import_button.menu()
        menu = QMenu(self.import_button)
        imports = [("接收器坐标 CSV", "receiver_file"), ("DAS 通道 / 端点 CSV", "das_channel_file")]
        imports += ([("主动炮点 CSV", "shot_file")] if mapping["run"]["mode"] == "active" else
                    [("噪声源坐标 CSV", "source_file"), ("噪声 PSD CSV", "psd_file"), ("逐窗源活动 CSV", "activity_file")])
        if mapping["run"]["mode"] == "passive" and mapping.get("ambient_sources", {}).get("activity_file"):
            imports.append(("移除源活动表（恢复全源活动）", "clear_activity_file"))
        for label, action in imports:
            menu.addAction(label, lambda checked=False, a=action: self._parameter_action(a))
        self.import_button.setMenu(menu)
        if old_menu is not None:
            old_menu.deleteLater()
        layers = mapping["model"]["layers"]
        table = QTableWidget(len(layers), 6)
        table.setAlternatingRowColors(True)
        keys = ("thickness_m", "vp_mps", "vs_mps", "density_kgm3", "qp", "qs")
        table.setHorizontalHeaderLabels(["层厚 m", "Vp m/s", "Vs m/s", "ρ kg/m³", "Qp", "Qs"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row, layer in enumerate(layers):
            for col, key in enumerate(keys):
                item = QTableWidgetItem(str(layer.get(key, float("inf"))))
                item.setToolTip(parameter_detail_html(f"model.layers[].{key}", layer.get(key), mapping, compact=True))
                table.setItem(row, col, item)
        table.cellChanged.connect(lambda r, c: self.set_value(
            ("model", "layers", r, keys[c]), table.item(r, c).text(), 0.0))
        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(table)
        layer_actions = QHBoxLayout()
        for label, action in (("＋ 有限层", "add_layer"), ("－ 最后有限层", "remove_layer")):
            button = QPushButton(label)
            button.clicked.connect(lambda _, a=action: self._parameter_action(a))
            layer_actions.addWidget(button)
        layer_actions.addWidget(QLabel("末层 inf 为半空间；修改后更新预览。"))
        layer_actions.addStretch()
        model_layout.addLayout(layer_actions)
        self.compact.addTab(model_page, "① 地层模型")
        groups = [("Geometry", ["geometry", "das"]),
                  ("源与记录", ["active_source"] if mapping["run"]["mode"] == "active" else ["ambient_sources", "noise"]),
                  ("成像", ["processing"])]
        groups.append(("计算设置", ["run", "experiment"]))
        hidden = {"theory_solver", "picker_smoothness", "picker_maximum_jump_mps", "picker_minimum_energy",
                  "radial_kernel_lookup_step", "independent_sources", "overlap", "source_amplitude", "product", "waves"}
        if "hvsr" not in request.product:
            hidden.update({"hvsr_smoothing_b", "hvsr_floor_ratio"})
        for title, sections in groups:
            page = QWidget()
            group_row = QHBoxLayout(page)
            group_row.setContentsMargins(8, 0, 8, 4)
            group_row.setSpacing(10)
            group_grids = {}
            counts = {}
            fields = []
            for section in sections:
                values = mapping.get(section, {})
                if section == "run":
                    values = {key: values.get(key, default) for key, default in
                              (("workers", 0), ("frequency_block_size", 16), ("source_block_size", 16))}
                prefix = (section,)
                if section == "geometry":
                    values = values["receivers"]
                    prefix += ("receivers",)
                for key, value in values.items():
                    if key == "event_rate_hz" and mapping.get("noise", {}).get("temporal_model") != "poisson_ricker":
                        continue
                    if key not in hidden and isinstance(value, (str, float, int, bool)):
                        fields.append((prefix + (key,), value))
            for index, (path, value) in enumerate(fields):
                if title == "Geometry":
                    group_name = "接收器位置与布局" if path[0] == "geometry" else "DAS 测量与标距"
                elif title == "源与记录" and path[0] == "active_source":
                    group_name = ("炮点与激发方向" if path[-1] in {"x_m", "y_m", "component"} or path[-1].startswith("force_") else
                                  "时间采样与记录" if path[-1] in {"dt_s", "record_duration_s"} else "子波与正演带宽")
                elif title == "源与记录":
                    group_name = ("噪声源空间分布" if path[0] == "ambient_sources" else
                                  "噪声频谱 / PSD" if path[-1] in {"minimum_frequency_hz", "maximum_frequency_hz", "dominant_frequency_hz", "psd_shape", "psd_file", "peak_frequency_hz"} else "时间过程与记录")
                elif title == "计算设置":
                    group_name = ("CPU并行（非GPU）" if path[-1] == "workers" else
                                  "分块与内存" if path[0] == "run" else "理论 / 可选分析")
                else:
                    group_name = ("频率扫描范围" if path[-1].startswith("frequency_") or path[-1] == "zero_padding" else
                                  "相速度扫描范围" if path[-1].startswith("velocity_") else "成像与处理方法")
                if group_name not in group_grids:
                    box = QGroupBox(group_name)
                    grid = QGridLayout(box)
                    grid.setSpacing(6)
                    group_row.addWidget(box, 1, Qt.AlignmentFlag.AlignTop)
                    group_grids[group_name] = grid
                    counts[group_name] = 0
                grid = group_grids[group_name]
                edit = QLineEdit(str(value))
                edit.setMinimumWidth(85)
                spec = parameter_spec(".".join(path))
                label = QLabel(spec.label + (f" ({spec.unit})" if spec.unit and spec.unit != "—" else ""))
                if path in {("active_source", "record_duration_s"), ("noise", "maximum_lag_s"),
                            ("processing", "frequency_step_hz")}:
                    label.setText(label.text() + " · 0=自动")
                label.setObjectName("fieldLabel")
                label.setWordWrap(True)
                tip = parameter_detail_html(".".join(path), value, mapping, compact=True)
                edit.setToolTip(tip)
                label.setToolTip(tip)
                choices = parameter_choices(".".join(path))
                if isinstance(value, bool) or choices:
                    edit = QComboBox()
                    options = (("true", "开启"), ("false", "关闭")) if isinstance(value, bool) else choices
                    for raw, text in options:
                        edit.addItem(text, raw)
                    selected_value = str(value).lower() if isinstance(value, bool) else value
                    if edit.findData(selected_value) < 0:
                        edit.addItem(f"{selected_value}（文件输入值）", selected_value)
                    edit.setCurrentIndex(edit.findData(selected_value))
                    edit.setToolTip(tip)
                    edit.currentIndexChanged.connect(lambda _, p=path, e=edit, v=value: self.set_value(p, str(e.currentData()), v))
                else:
                    edit.editingFinished.connect(lambda p=path, e=edit, v=value: self.set_value(p, e.text(), v))
                edit.setObjectName(".".join(path))
                if path[0] == "das" and path[-1] != "enabled" and not mapping.get("das", {}).get("enabled"):
                    edit.setEnabled(False)
                    edit.setToolTip("先开启 DAS，再设置标距和测量参数。<br>" + tip)
                label.setBuddy(edit)
                slot = QWidget()
                slot_layout = QVBoxLayout(slot)
                slot_layout.setContentsMargins(0, 0, 0, 0)
                slot_layout.setSpacing(3)
                slot_layout.addWidget(label)
                slot_layout.addWidget(edit)
                row, col = divmod(counts[group_name], 3 if title == "Geometry" or mapping["run"]["mode"] == "passive" else 2)
                grid.addWidget(slot, row, col)
                counts[group_name] += 1
            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            wrapper_layout.addWidget(scroll)
            self.compact.addTab(wrapper, {"Geometry": "② 观测系统 / DAS", "源与记录": "③ 源与记录", "成像": "④ 频散成像", "计算设置": "⑤ 计算 / 理论 / HVSR" if "hvsr" in request.product else "⑤ 计算 / 理论"}[title])
        self.compact.setCurrentIndex(max(0, selected))

    def select_page(self, page):
        index = {"parameters": 0, "review": 1, "run": 2, "results": 3, "guide": 4}.get(page, page)
        # Background compute must not open a modal page over the experiment.
        if index in (2, 3) and self._starting_stage:
            return
        if index in self._page_dialogs:
            dialog = self._page_dialogs[index]
            dialog.show()
            dialog.raise_()
            return
        widget = (self.parameter_page, self.review_page, self.run_page, self.results_page, self.guide_page)[int(index)]
        dialog = QDialog(self)
        dialog.setWindowTitle(self.PAGE_NAMES[int(index)])
        dialog.resize(1200, 780)
        layout = QVBoxLayout(dialog)
        layout.addWidget(widget)
        self.dialogs.append(dialog)
        self._page_dialogs[index] = dialog
        dialog.finished.connect(lambda _: self.rebuild_controls())
        widget.show()
        dialog.show()

    def validate_case(self, *args, **kwargs):
        if self._running:
            return False
        mapping = self.parameter_page.mapping()
        try:
            request = ExperimentRequest.from_mapping(mapping)
        except ValueError as exc:
            self._show_error("计算目标参数无效", exc)
            self._set_ui_state("error")
            return False
        if request.product == "theory":
            try:
                context = theory_context(mapping, self.session.workspace_directory)
                self._theory_context = context
                self._dirty = False
                self._set_ui_state("ready")
                self.notice.setText("仅理论：检查模型与频率；不要求激发源、观测分量或记录时长。QED求根将在运行按钮后执行。")
                self.geometry_panel.show_placeholder("本次仅求理论曲线；不使用 Geometry 或源方向。")
                fig, ax = plt.subplots(figsize=(4,3), layout="constrained")
                for key,label,color in (("vp_mps","Vp","#ca8518"),("vs_mps","Vs","#117b96")):
                    values,depth = _layer_steps(context.config, np.array([getattr(l,key) for l in context.config.model.layers]))
                    ax.plot(values,depth,label=label,color=color)
                density,depth = _layer_steps(context.config,np.array([l.density_kgm3 for l in context.config.model.layers]))
                rhoax=ax.twiny()
                rhoax.plot(density,depth,"--",color="#965cb1")
                rhoax.set_xlabel("Density (kg/m³)",color="#965cb1")
                ax.invert_yaxis()
                ax.set(xlabel="Vp / Vs (m/s)",ylabel="Depth (m)")
                ax.legend(loc="lower right")
                self.model_panel.set_figure(fig)
                self.sampling_summary.setText(f"理论频率 {context.config.processing.frequency_min_hz:g}–{context.config.processing.frequency_max_hz:g} Hz；实际求值步长 {request.theory_step_hz:g} Hz；无时间采样。")
                return True
            except Exception as exc:
                self._dirty = True
                self._set_ui_state("error")
                self._show_error("理论输入无效", exc)
                return False
        mapping["processing"]["theory_solver"] = "qedisp"
        self.parameter_page.set_mapping(mapping)
        kwargs["navigate"] = False
        ready = super().validate_case(*args, **kwargs)
        if self._dirty:
            self._set_ui_state("error")
            self.model_panel.show_placeholder("输入尚未通过检查，请修正参数后更新预览。")
            self.geometry_panel.show_placeholder("输入尚未通过检查，请修正参数后更新预览。")
            self.notice.setText("输入检查失败，旧模型和 Geometry 不代表当前输入。")
        elif self._review is not None:
            config, geometry = self._review.config, self._review.geometry
            fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
            for key, label, color in (("vp_mps", "Vp", "#ca8518"), ("vs_mps", "Vs", "#117b96")):
                values, depth = _layer_steps(config, np.array([getattr(l, key) for l in config.model.layers]))
                ax.plot(values, depth, label=label, color=color)
            rho, depth = _layer_steps(config, np.array([l.density_kgm3 for l in config.model.layers]))
            density_axis = ax.twiny()
            density_axis.plot(rho, depth, "--", color="#965cb1")
            density_axis.set_xlabel("Density (kg/m³)", color="#965cb1")
            ax.invert_yaxis()
            ax.set(xlabel="Vp / Vs (m/s)", ylabel="Depth (m)")
            ax.legend(loc="lower right")
            self.model_panel.set_figure(fig)
            fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
            ax.scatter(geometry.source_x_m, geometry.source_y_m, marker="*", color="#c55b35", label="Sources")
            ax.scatter(geometry.receiver_x_m, geometry.receiver_y_m, marker="v", s=12, label="Receivers")
            from wavedispx.plotting import _plot_das_geometry
            _plot_das_geometry(ax, self._review.das_geometry)
            source = config.active_source if config.mode == "active" else config.ambient_sources
            azimuth = (source.force_azimuth_deg if config.mode == "active" and source.component == "vector"
                       else source.horizontal_azimuth_deg if config.mode == "passive" and source.polarization == "directed_hv" else None)
            if azimuth is not None:
                horizontal = (np.cos(np.deg2rad(source.force_inclination_deg)) if config.mode == "active"
                              else np.sqrt(source.horizontal_power_fraction))
                span = max(np.ptp(np.r_[geometry.source_x_m, geometry.receiver_x_m]),
                           np.ptp(np.r_[geometry.source_y_m, geometry.receiver_y_m]), 1.)
                angle = np.deg2rad(azimuth)
                ax.quiver(geometry.source_x_m, geometry.source_y_m,
                          np.full(len(geometry.source_x_m), .08*span*horizontal*np.cos(angle)),
                          np.full(len(geometry.source_y_m), .08*span*horizontal*np.sin(angle)),
                          angles="xy", scale_units="xy", scale=1, color="#7b529e", label="Horizontal force direction")
            ax.set(xlabel="x (m)", ylabel="y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=8)
            self.geometry_panel.set_figure(fig)
            mode = "主动" if config.mode == "active" else "被动"
            self.notice.setText(f"{mode} · {config.processing.dispersion_method} · CPU正演 / QED理论曲线 · 输入审核通过（不等于收敛验收）" if ready else "输入审核有失败项，请打开审核详情")
            d = self._review.diagnostics
            timing = (f"记录周期≈{d['record_duration_s']:.3g} s；1/T≈{d['physical_frequency_step_hz']:.3g} Hz"
                      if config.mode == "active" else
                      f"总记录 {config.noise.duration_s:g} s / 独立窗 {config.noise.window_s:g} s；NCF lag ±{d['passive_maximum_lag_s']:.3g} s；1/T窗={1/config.noise.window_s:.3g} Hz")
            p = config.processing
            self.sampling_summary.setText(f"审核估计｜{timing}｜f={p.frequency_min_hz:g}–{p.frequency_max_hz:g} Hz；扫描 df≈{d['display_frequency_step_hz']:.3g} Hz；v={p.velocity_min_mps:g}–{p.velocity_max_mps:g} / dv={p.velocity_step_mps:g} m/s")
            has_result = self.workflow is not None and (self.workflow.active_analysis is not None or self.workflow.passive_analysis is not None)
            self._set_ui_state("result" if ready and has_result else "ready" if ready else "error")
            if not has_result and self.experiment_result is None:
                for wave,panel in (("rayleigh",self.rayleigh_panel),("love",self.love_panel)):
                    reason = wave_unavailable(config,wave)
                    panel.show_placeholder(reason or "输入已检查；点击运行计算后显示结果。")
        return ready

    def start_stage(self, stage):
        if self._running:
            return
        request = ExperimentRequest.from_mapping(self.parameter_page.mapping())
        if stage == "save" and self.experiment_result is not None and not self._dirty:
            result, context, mapping = self.experiment_result, self._experiment_workflow, deepcopy(self.parameter_page.mapping())
            self._start_experiment_worker("experiment_save", lambda: save_experiment(result, context, mapping))
            return
        if stage in {"analysis", "all"} and (request.product != "legacy" or request.waves != "both"):
            if not self.validate_case(navigate=False):
                return
            context = self._theory_context if request.product == "theory" else self.workflow
            self._experiment_workflow = context
            self._start_experiment_worker("experiment", lambda: run_experiment(context, request))
            return
        if stage == "save" and (self._dirty or self.workflow is None or
                (self.workflow.active_analysis is None and self.workflow.passive_analysis is None)):
            self.notice.setText("尚无当前参数对应的计算结果。请先点击“检查并计算”；保存不会替你启动正演。")
            return
        self._starting_stage = True
        try:
            super().start_stage(stage)
        finally:
            self._starting_stage = False
        self.parameter_page.setEnabled(not self._running)
        self.compute_button.setEnabled(not self._running)
        self.compact.setEnabled(not self._running)
        if self._running:
            self._set_ui_state("running")
            self.notice.setText("CPU计算中；请等待当前阶段结束。此版暂不支持强制中止。")
        for widget in (self.load_template_button, self.open_button):
            widget.setEnabled(not self._running)

    def _start_experiment_worker(self, stage, action):
        worker = WorkflowWorker(stage, action)
        worker.signals.succeeded.connect(self._stage_succeeded)
        worker.signals.failed.connect(self._stage_failed)
        self._running = True
        self._active_worker = worker
        self._set_ui_state("running")
        self.run_page.append_log("开始计算目标："+self.product_combo.currentText())
        self.notice.setText("按所选目标计算；未选择的波型会保留说明，不自动改变源。")
        self.thread_pool.start(worker)

    def _stage_succeeded(self, stage, result):
        if stage in {"experiment", "experiment_save"}:
            self._running = False
            self._active_worker = None
            if stage == "experiment":
                self.experiment_result = result
                if result.hvsr is not None:
                    self.hvsr_station.blockSignals(True)
                    self.hvsr_station.clear()
                    self.hvsr_station.addItems([f"台站 {i}" for i in range(len(result.hvsr.geometric_mean))])
                    self.hvsr_station.blockSignals(False)
                    self.hvsr_station.show()
                    count = len(result.hvsr.geometric_mean)
                    self._hvsr_selected = np.unique(np.linspace(0,count-1,min(5,count)).astype(int)).tolist()
                    self.hvsr_controls.show()
                    self.result_tabs.setCurrentWidget(self.hvsr_panel)
                self.notice.setText("计算目标完成；各波型页保留状态说明。结果未经通用收敛认证。")
            else:
                self.notice.setText("结果已保存到独立目录："+str(result))
            self._set_ui_state("result")
            self.statusBar().showMessage(self.notice.text())
            self.run_page.append_log(self.notice.text())
            self.render_results()
            # Complete the small context canvases after the worker/Qt resize
            # cycle, including the independent density axis on Retina.
            for panel in (self.model_panel, self.geometry_panel):
                if panel._canvas is not None:
                    panel._canvas.draw()
            return
        self._starting_stage = True
        try:
            super()._stage_succeeded(stage, result)
        finally:
            self._starting_stage = False
        # The old run page is retained for logs, not a second hidden canvas.
        self.run_page.figure.show_placeholder("结果显示在 Studio 主窗口中。")
        self.parameter_page.setEnabled(True)
        self.compute_button.setEnabled(True)
        self.compact.setEnabled(True)
        self.load_template_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.notice.setText("计算完成 · 当前参数对应的结果 · 修改显示不重算")
        self._set_ui_state("result" if self.workflow.active_analysis is not None or self.workflow.passive_analysis is not None else "ready")
        if stage == "save":
            self.notice.setText(f"结果已保存：{self.workflow.config.output_directory}")
        self.render_results()
        if self.wave_toggle.isChecked():
            self.show_waveforms(True)

    def _stage_failed(self, *args):
        super()._stage_failed(*args)
        self.parameter_page.setEnabled(True)
        self.compute_button.setEnabled(True)
        self.compact.setEnabled(True)
        self.load_template_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.notice.setText("计算失败；未切换后端。打开日志查看原因。")
        self._set_ui_state("error")
        if args and args[0] == "experiment":
            self.experiment_result = None
            for panel in (self.rayleigh_panel, self.love_panel, self.hvsr_panel):
                panel.show_placeholder("本次计算失败，未产生可用结果：\n"+str(args[1]))

    def render_results(self, *_):
        if self._dirty or self._running:
            return
        if self.experiment_result is not None:
            self.hvsr_controls.setVisible(self.experiment_result.hvsr is not None)
            if self.experiment_result.hvsr is not None:
                self._draw_hvsr_geometry()
            self._render_experiment()
            return
        if self.workflow is None:
            return
        analysis = self.workflow.active_analysis if self.workflow.config.mode == "active" else self.workflow.passive_analysis
        if analysis is None:
            return
        rayleigh = analysis.rayleigh_combined if self.workflow.config.mode == "active" else analysis.dispersion
        if self.workflow.config.mode == "active":
            shot = self.workflow.shot
            timing = f"记录周期 {len(shot.time_s)*self.workflow.config.active_source.dt_s:g} s，末样点 {shot.time_s[-1]:g} s" if shot is not None else ""
        else:
            correlation = self.workflow.correlation
            timing = (f"总记录 {self.workflow.config.noise.duration_s:g} s，{correlation.independent_window_count} 独立窗，实际 lag ±{correlation.ncf.lag_time_s[-1]:g} s"
                      if correlation is not None else "")
        f, v = rayleigh.frequency_hz, rayleigh.velocity_mps
        df = np.median(np.diff(f)) if len(f) > 1 else float("nan")
        dv = np.median(np.diff(v)) if len(v) > 1 else float("nan")
        self.sampling_summary.setText(f"当前结果实际坐标｜{timing}｜f={f[0]:g}–{f[-1]:g} Hz / df={df:.4g}（{len(f)}点）；v={v[0]:g}–{v[-1]:g} m/s / dv={dv:.4g}（{len(v)}点）")
        love = (analysis.love if self.workflow.config.mode == "active"
                else analysis.component_images.get("tt"))
        for panel, image, curves, title in (
            (self.rayleigh_panel, rayleigh, analysis.theory.rayleigh_mps, "Rayleigh / DAS"),
            (self.love_panel, love, analysis.theory.love_mps, "Love")):
            if self.result_tabs.currentWidget() is not panel:
                continue
            reason = wave_unavailable(self.workflow.config, "love" if panel is self.love_panel else "rayleigh")
            if reason:
                panel.show_placeholder(reason)
                continue
            if image is None:
                panel.show_placeholder("当前观测未生成 Love 能量图，不以理论曲线代替观测。")
                continue
            if not np.any(image.energy > 0):
                panel.show_placeholder("当前分量能量为零；请检查源方向与观测分量。不会把零图或理论线当成有效频散能量。")
                continue
            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
            mesh = ax.pcolormesh(image.frequency_hz, image.velocity_mps, image.energy,
                                 shading="nearest", cmap="turbo", vmin=0, vmax=1)
            if self.overlay.isChecked():
                for mode in range(curves.shape[1]):
                    ax.plot(image.frequency_hz, curves[:, mode], "--", lw=1.2, label=f"QED M{mode}")
                ax.legend(loc="upper right")
            ax.set(xlabel="Frequency (Hz)", ylabel="Phase velocity (m/s)",
                   title=f"{title} | {self.workflow.config.mode} | {self.workflow.config.processing.dispersion_method}")
            fig.colorbar(mesh, ax=ax, label="Normalized image")
            panel.set_figure(fig)

    def _hvsr_view_changed(self, *_):
        if hasattr(self,"hvsr_scatter"):
            self.hvsr_scatter.blockSignals(True)
            self.hvsr_scatter.setChecked(self.hvsr_view.currentData() == "single")
            self.hvsr_scatter.blockSignals(False)
        self.render_results()

    def _select_hvsr_station(self, station, multi=False):
        if self._dirty or self._running or self.experiment_result is None or self.experiment_result.hvsr is None:
            return
        if not 0 <= station < len(self.experiment_result.hvsr.geometric_mean):
            return
        if multi:
            selected = set(self._hvsr_selected)
            if station in selected:
                selected.remove(station)
            elif len(selected) < 12:
                selected.add(station)
            else:
                self.notice.setText("最多叠加12站；请取消部分台站，或查看全台站热图。")
                return
            self._hvsr_selected = sorted(selected)
        self.hvsr_station.blockSignals(True)
        self.hvsr_station.setCurrentIndex(station)
        self.hvsr_station.blockSignals(False)
        self.hvsr_view.blockSignals(True)
        self.hvsr_view.setCurrentIndex(self.hvsr_view.findData("overlay" if multi else "single"))
        self.hvsr_view.blockSignals(False)
        self.hvsr_scatter.blockSignals(True)
        self.hvsr_scatter.setChecked(not multi)
        self.hvsr_scatter.blockSignals(False)
        self.result_tabs.setCurrentWidget(self.hvsr_panel)
        self.render_results()

    def _choose_hvsr_stations(self):
        if self.experiment_result is None or self.experiment_result.hvsr is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("选择HVSR叠加台站（最多12站）")
        dialog.resize(430,520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("编号为输入行顺序，从0开始；不对台站做平均。"))
        choices = QListWidget()
        g = self._experiment_workflow.geometry
        for i,(x,y) in enumerate(zip(g.receiver_x_m,g.receiver_y_m)):
            item = QListWidgetItem(f"台站 {i}   x={x:g} m, y={y:g} m")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if i in self._hvsr_selected else Qt.CheckState.Unchecked)
            choices.addItem(item)
        layout.addWidget(choices)
        warning = QLabel("")
        layout.addWidget(warning)
        clear = QPushButton("取消全部勾选")
        clear.clicked.connect(lambda: [choices.item(i).setCheckState(Qt.CheckState.Unchecked) for i in range(choices.count())])
        layout.addWidget(clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(dialog.reject)
        def accept():
            selected = [i for i in range(choices.count()) if choices.item(i).checkState() == Qt.CheckState.Checked]
            if not 1 <= len(selected) <= 12:
                warning.setText("请选择1–12个台站；更多台站请使用热图。")
                return
            self._hvsr_selected = selected
            dialog.accept()
            self.render_results()
        buttons.accepted.connect(accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _draw_hvsr_geometry(self):
        g = self._experiment_workflow.geometry
        if g is None:
            return
        station = max(0, self.hvsr_station.currentIndex())
        fig, ax = plt.subplots(figsize=(4,3), layout="constrained")
        ax.scatter(g.receiver_x_m,g.receiver_y_m,marker="v",s=15,color="#1768ac",label="Receivers")
        selected = self._hvsr_selected if self.hvsr_view.currentData() == "overlay" else [station]
        if selected:
            ax.scatter(g.receiver_x_m[selected],g.receiver_y_m[selected],s=55,facecolors="none",edgecolors="#b64f70",label="Selected",zorder=4)
        ax.annotate(f"S{station}\n({g.receiver_x_m[station]:g}, {g.receiver_y_m[station]:g}) m",
                    (g.receiver_x_m[station],g.receiver_y_m[station]),xytext=(6,12),textcoords="offset points",fontsize=8)
        span=max(np.ptp(g.receiver_x_m),np.ptp(g.receiver_y_m),1.)
        cx=(g.receiver_x_m.min()+g.receiver_x_m.max())/2
        cy=(g.receiver_y_m.min()+g.receiver_y_m.max())/2
        ax.set(xlabel="Receiver x (m)",ylabel="Receiver y (m)",
               xlim=(cx-.6*span,cx+.6*span),ylim=(cy-.6*span,cy+.6*span))
        ax.set_aspect("equal",adjustable="box")
        ax.legend(fontsize=7,loc="lower left")
        overview=ax.inset_axes([.67,.67,.31,.29])
        overview.scatter(g.source_x_m,g.source_y_m,marker="*",s=8,color="#c08016")
        overview.scatter(g.receiver_x_m,g.receiver_y_m,marker="v",s=3,color="#1768ac")
        overview.set_title("Survey: sources + receivers",fontsize=6)
        overview.set_xticks([])
        overview.set_yticks([])
        overview.set_aspect("equal",adjustable="datalim")
        self.geometry_panel.set_figure(fig)
        def pick(event):
            if event.inaxes is not ax or self.geometry_panel._toolbar.mode:
                return
            points = ax.transData.transform(np.column_stack((g.receiver_x_m,g.receiver_y_m)))
            cursor = ax.transData.transform((event.xdata,event.ydata))
            distance = np.linalg.norm(points-cursor,axis=1)
            nearest = int(distance.argmin())
            if distance[nearest] <= 12*fig.dpi/72:
                key = str(event.key or "").lower()
                self._select_hvsr_station(nearest, multi=any(k in key for k in ("control","ctrl","cmd","super")))
        self.geometry_panel._canvas.mpl_connect("button_press_event",pick)

    def _render_experiment(self):
        result = self.experiment_result
        panel = self.result_tabs.currentWidget()
        if panel is self.wave_panel:
            self.show_waveforms(True)
            return
        if panel is self.hvsr_panel:
            h = result.hvsr
            if h is None:
                panel.show_placeholder("本次未计算HVSR。需要被动XYZ原始记录，不使用NCF或DAS应变替代。")
                return
            station = max(0,self.hvsr_station.currentIndex())
            from .hvsr_view import curve_figure, heatmap_figure
            view = self.hvsr_view.currentData()
            self.hvsr_select.setEnabled(view == "overlay")
            self.hvsr_scatter.setEnabled(view in {"single", "overlay"})
            self.sampling_summary.setText(f"原始XYZ的HVSR｜{len(h.geometric_mean)}台站 / {h.processed_windows}窗；台站{station}有效窗数 {h.valid_windows[station].min()}–{h.valid_windows[station].max()}。只切换显示，不重新正演。")
            if view in {"single", "overlay"}:
                ids = [station] if view == "single" else self._hvsr_selected
                panel.set_figure(curve_figure(h, ids, scatter=self.hvsr_scatter.isChecked()))
                return
            if view == "heatmap":
                g = self._experiment_workflow.geometry
                fig, axis, order = heatmap_figure(h, g.receiver_x_m, g.receiver_y_m)
                panel.set_figure(fig)
                def pick_row(event):
                    if event.inaxes is fig.axes[0] and event.ydata is not None and not panel._toolbar.mode:
                        self._select_hvsr_station(int(order[np.argmin(np.abs(axis-event.ydata))]))
                panel._canvas.mpl_connect("button_press_event", pick_row)
                return
            f, mean, spread = h.frequency_hz, h.geometric_mean[station], h.log_std[station]
            if not np.isfinite(mean).any():
                panel.show_placeholder(f"台站 {station} 的HVSR无有效频点。请检查垂直谱和窗长：至少10周期/频点、3个有效窗，垂直谱须高于相对下限。不会以小常数补出峰值。")
                return
            fig, axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
            axes[0].plot(f,mean,label="Geometric mean H/V")
            axes[0].fill_between(f,mean*np.exp(-spread),mean*np.exp(spread),alpha=.2,label="Window log scatter ±1σ")
            axes[0].set(xlabel="Frequency (Hz)",ylabel="H/V",title=f"Station {station} | {h.processed_windows} windows")
            axes[0].legend(fontsize=7)
            for i,label in enumerate("XYZ"):
                axes[1].semilogy(f,h.mean_amplitude_xyz[station,:,i],label=label)
            axes[1].set(xlabel="Frequency (Hz)",ylabel="Fourier amplitude (native units)")
            axes[1].legend()
            mesh=axes[2].pcolormesh(f,h.directions_deg,h.directional_geometric_mean[station].T,shading="nearest")
            axes[2].set(xlabel="Frequency (Hz)",ylabel="Azimuth from +X (deg)")
            fig.colorbar(mesh,ax=axes[2],label="Directional H/V")
            fig.suptitle("Raw XYZ HVSR | not Rayleigh ellipticity | invalid bins remain NaN")
            self.sampling_summary.setText(f"HVSR：全记录 {h.processed_windows} 窗；台站{station}有效窗数 {h.valid_windows[station].min()}–{h.valid_windows[station].max()}；KO b={h.smoothing_b:g}；垂直谱相对下限={h.vertical_floor_ratio:g}。")
            panel.set_figure(fig)
            return
        if panel not in (self.rayleigh_panel,self.love_panel):
            return
        wave = "rayleigh" if panel is self.rayleigh_panel else "love"
        status = result.statuses[wave]
        image = result.images.get(wave)
        curves = getattr(result.theory,wave+"_mps") if result.theory is not None else None
        if image is None and (result.request.product != "theory" or curves is None or not np.isfinite(curves).any()):
            panel.show_placeholder(status)
            self.sampling_summary.setText(status)
            return
        fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
        if image is not None:
            mesh=ax.pcolormesh(image.frequency_hz,image.velocity_mps,image.energy,shading="nearest",cmap="turbo",vmin=0,vmax=1)
            fig.colorbar(mesh,ax=ax,label="Normalized image")
        if curves is not None and (self.overlay.isChecked() or result.request.product == "theory"):
            for i in range(curves.shape[1]):
                ax.plot(result.theory.frequency_hz,curves[:,i],"--",label=f"QED M{i}")
            if curves.shape[1]:
                ax.legend()
        ax.set(xlabel="Frequency (Hz)",ylabel="Phase velocity (m/s)",title=wave.title()+" | "+result.request.product)
        if image is not None:
            f,v=image.frequency_hz,image.velocity_mps
            df=np.median(np.diff(f)) if len(f)>1 else float("nan")
            dv=np.median(np.diff(v)) if len(v)>1 else float("nan")
            c=self._experiment_workflow.config
            timing=(f"主动记录周期 {len(self._experiment_workflow.shot.time_s)*c.active_source.dt_s:g} s"
                    if c.mode=="active" else f"被动总记录 {c.noise.duration_s:g} s / 独立窗 {c.noise.window_s:g} s")
            self.sampling_summary.setText(f"{timing}｜实际 f={f[0]:g}–{f[-1]:g} Hz / df={df:.4g}；v={v[0]:g}–{v[-1]:g} m/s / dv={dv:.4g}｜{status}")
        else:
            self.sampling_summary.setText(status)
        panel.set_figure(fig)

    def show_waveforms(self, enabled):
        index = self.result_tabs.indexOf(self.wave_panel)
        if not enabled:
            if index >= 0:
                self.result_tabs.removeTab(index)
            return
        if index < 0:
            self.result_tabs.addTab(self.wave_panel, "波形 / 双边NCF")
        self.result_tabs.setCurrentWidget(self.wave_panel)
        if self.experiment_result is not None and self.experiment_result.request.product == "theory":
            self.wave_panel.show_placeholder("本次仅计算理论曲线，没有执行波场正演，不能显示旧算例的波形。")
            return
        if self.experiment_result is not None and self.experiment_result.request.product == "hvsr":
            context = self._experiment_workflow
            if hasattr(context, "hvsr_preview"):
                record=context.hvsr_preview
                station=max(0,self.hvsr_station.currentIndex())
                fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
                for i,label in enumerate("XYZ"):
                    ax.plot(np.arange(len(record))*context.config.noise.dt_s,record[:,station,i],label=label)
                ax.set(xlabel="Time (s)",ylabel="Native displacement",title=f"Station {station} | first window only; HVSR uses all windows")
                ax.legend()
                self.wave_panel.set_figure(fig)
            else:
                self.wave_panel.show_placeholder("当前载入的是HVSR统计结果，未包含首窗原始XYZ波形；不会为显示自动重新正演。")
            return
        if not self._dirty and not self._running and self.workflow is not None:
            if self.workflow.shot is not None or self.workflow.correlation is not None:
                self.wave_panel.set_figure(plot_workflow_observations(self.workflow))

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted():
            for panel in self.findChildren(FigurePane):
                panel.clear()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = StudioWindow(args.template)
    window.show_windowed()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            app.quit()
        QTimer.singleShot(1500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
