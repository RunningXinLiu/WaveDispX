"""Native WaveDispX desktop workbench built with PySide6.

The widgets in this module do not implement any modelling mathematics.  They
delegate validation and every computational stage to :class:`CaseSession`, the
same application boundary used by the Streamlit UI and tutorial notebooks.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import traceback
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import numpy as np

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from wavedispx.app_service import (
    apply_automatic_parameters,
    CaseInputError,
    CaseSession,
    coordinate_csv_count,
    explain_input_error,
    load_template_mapping,
    parse_shot_csv,
    parse_toml_bytes,
    reset_automatic_parameters,
    toml_dumps,
)
from wavedispx.das import load_das_geometry_csv
from wavedispx.preflight import (
    PreflightReview,
    plot_preflight_review,
    save_preflight_review,
)
from wavedispx.workflow_plotting import (
    plot_workflow_dispersion,
    plot_workflow_forward,
    plot_workflow_observations,
)

from .config_editor import ConfigTreeEditor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = {
    "主动源｜快速检查": PROJECT_ROOT / "examples/active_quick.toml",
    "DAS｜主动线阵快速检查": PROJECT_ROOT / "examples/active_das_line_quick.toml",
    "DAS｜主动面阵快速检查": PROJECT_ROOT / "examples/active_das_grid_quick.toml",
    "DAS｜逐道光缆文件": PROJECT_ROOT / "examples/active_das_channel_file_quick.toml",
    "主动源｜正式两层模型": PROJECT_ROOT / "examples/active_two_layer.toml",
    "主动源｜高阶模态 3C": PROJECT_ROOT / "examples/active_high_modes_3c.toml",
    "被动源｜快速检查": PROJECT_ROOT / "examples/passive_quick.toml",
    "DAS｜被动线阵快速检查": PROJECT_ROOT / "examples/passive_das_line_quick.toml",
    "DAS｜被动面阵快速检查": PROJECT_ROOT / "examples/passive_das_grid_quick.toml",
    "DAS｜被动逐道光缆": PROJECT_ROOT / "examples/passive_das_channel_file_quick.toml",
    "被动源｜正式收敛": PROJECT_ROOT / "examples/passive_two_layer.toml",
    "被动源｜高阶模态物理 3C": PROJECT_ROOT / "examples/passive_high_modes_3c.toml",
    "被动源｜二维面阵": PROJECT_ROOT / "examples/passive_grid_quick.toml",
}


APP_STYLE = """
QMainWindow, QWidget { background: #f5f7fa; color: #1c2733; font-size: 13px; }
QFrame#sidebar { background: #172638; border: 0; }
QFrame#sidebar QLabel { background: transparent; }
QLabel#brand { color: white; font-size: 25px; font-weight: 700; }
QLabel#tagline { color: #a9bdd2; font-size: 11px; }
QListWidget#navigation { background: transparent; border: 0; color: #cbd8e5; outline: 0; }
QListWidget#navigation::item { padding: 13px 14px; margin: 3px 8px; border-radius: 7px; }
QListWidget#navigation::item:selected { background: #2f80ed; color: white; }
QListWidget#navigation::item:hover { background: #243b54; }
QFrame#topbar, QFrame#card { background: white; border: 1px solid #dce3ea; border-radius: 9px; }
QLabel#pageTitle { font-size: 21px; font-weight: 700; color: #142231; }
QLabel#pageSubtitle { color: #647586; }
QLabel#editorHint { color: #647586; }
QPushButton, QToolButton { background: white; border: 1px solid #bfcbd7; border-radius: 6px; padding: 7px 11px; }
QPushButton:hover, QToolButton:hover { border-color: #2f80ed; color: #1f6dcc; }
QPushButton:disabled { color: #9aa8b5; background: #eef1f4; }
QPushButton#primary { background: #2f80ed; color: white; border-color: #2f80ed; font-weight: 600; }
QPushButton#success { background: #168f5b; color: white; border-color: #168f5b; font-weight: 600; }
QPushButton#danger { background: #bd3b42; color: white; border-color: #bd3b42; }
QComboBox, QLineEdit, QPlainTextEdit, QTreeWidget, QTableWidget, QTextBrowser, QListWidget#gallery {
  background: white; border: 1px solid #cfd8e2; border-radius: 5px; selection-background-color: #2f80ed;
}
QComboBox { padding: 6px 9px; min-height: 20px; }
QHeaderView::section { background: #edf2f7; color: #33485d; padding: 7px; border: 0; border-right: 1px solid #d8e0e8; }
QTabWidget::pane { border: 1px solid #d5dee7; background: white; }
QTabBar::tab { background: #e9eef4; padding: 9px 18px; }
QTabBar::tab:selected { background: white; color: #1f6dcc; font-weight: 600; }
QProgressBar { border: 1px solid #cfd8e2; border-radius: 5px; text-align: center; background: white; }
QProgressBar::chunk { background: #2f80ed; }
QStatusBar { background: white; border-top: 1px solid #dce3ea; }
QToolTip { background: #fffef8; color: #172433; border: 1px solid #8fa9c1; border-radius: 6px; padding: 8px; }
QTextBrowser#parameterHelpCard { background: #f0f6ff; border: 1px solid #b9d3f2; border-left: 5px solid #2f80ed; border-radius: 7px; padding: 7px; }
"""


class WorkerSignals(QObject):
    succeeded = Signal(str, object)
    failed = Signal(str, str, str)


class WorkflowWorker(QRunnable):
    """Run one core workflow action without blocking the Qt event loop."""

    def __init__(self, key: str, action: Callable[[], object]):
        super().__init__()
        self.key = key
        self.action = action
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.action()
        except Exception as exc:  # pragma: no cover - delivered to the GUI
            self.signals.failed.emit(self.key, str(exc), traceback.format_exc())
        else:
            self.signals.succeeded.emit(self.key, result)


class PageHeading(QWidget):
    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class FigurePane(QWidget):
    """Replaceable Matplotlib canvas with standard pan/zoom tools."""

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self._figure = None
        self._canvas = None
        self._toolbar = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel(placeholder)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet(
            "background:white;border:1px dashed #b9c6d2;border-radius:8px;color:#718193;padding:30px;"
        )
        self._layout.addWidget(self.placeholder, 1)

    def set_figure(self, figure) -> None:
        self.clear()
        self._figure = figure
        self._canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def clear(self) -> None:
        if self.placeholder is not None and self.placeholder.parent() is self:
            self._layout.removeWidget(self.placeholder)
            self.placeholder.hide()
        if self._toolbar is not None:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.deleteLater()
            self._toolbar = None
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None

    def show_placeholder(self, message: str) -> None:
        self.clear()
        self.placeholder.setText(message)
        self.placeholder.show()
        self._layout.addWidget(self.placeholder, 1)


class ParameterPage(QWidget):
    mapping_changed = Signal()
    error = Signal(str, object)
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.addWidget(
            PageHeading(
                "1  建立算例",
                "先载入最接近的模板。双击当前值编辑；悬停或单击任意参数，会显示单位、硬约束、推荐值、调大/调小影响和当前设置判断。",
            )
        )

        tools = QFrame()
        tools.setObjectName("card")
        tools_layout = QGridLayout(tools)
        tools_layout.setContentsMargins(10, 8, 10, 8)
        for index, (label, key, tooltip) in enumerate((
            ("自动填入 f/v/df/dv", "apply_auto_design", "按当前模型、源谱、Geometry 与 DAS 标距计算频散范围/网格，并填入主动记录时长或被动 NCF lag。填入后仍可逐项编辑。"),
            ("恢复模板默认", "reset_auto_design", "只恢复所选模板中的 f/v/df/dv 与记录窗/lag；不覆盖模型、Geometry 或源位置。"),
            ("＋ 模型层", "add_layer", "在末层半空间之前增加一个有限层，并复制邻近层的 Vp/Vs/ρ/Q 作为可编辑起点。修改层厚和速度后必须重新审核理论频散。"),
            ("－ 模型层", "remove_layer", "删除最下面一个有限层；末层 inf 半空间始终保留。该操作会使全部正演、NCF 和频散缓存失效。"),
            ("线阵", "receiver_line", "用参数设计线阵：x 起点、x 终点、道间距和固定 y。会替换当前接收器表，但不会改模型或源。"),
            ("二维面阵", "receiver_grid", "用 x/y 起终点和间距设计规则二维阵列。被动处理还会增加径向距离箱 distance_bin_m。"),
            ("接收器 CSV", "receiver_file", "导入任意二维接收器坐标。文件必须为 UTF-8 CSV，至少两行，包含 x_m,y_m，且不能有重复位置。"),
            ("DAS 光缆 CSV", "das_channel_file", "导入逐道 DAS 几何。每行除 x_m,y_m 外填写两个端点，或填写该道标距与方位；同时切换为接收器文件输入。"),
            ("源环", "source_annulus", "被动源改为环形方位照明；默认使用 stratified 方位抽样和至少 192 个源，适合检验近似弥散场恢复。"),
            ("源线", "source_line", "被动源改为道路/线性噪声带；单侧照明可能造成 NCF 不对称，解释结果时应保留该限制。"),
            ("噪声源 CSV", "source_file", "导入已知噪声源位置 x_m,y_m；源数会按有效数据行自动更新，空间随机种子不再改变位置。"),
            ("炮点 CSV", "shot_file", "读取 shot_id,x_m,y_m 多炮表并明确选择本次的一炮；选中坐标会固化到 active_source。"),
            ("PSD CSV", "psd_file", "导入 frequency_hz,power 外部功率谱；频率必须严格递增、功率非负，输入审核会画出实际谱形。"),
        )):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda checked=False, name=key: self.action_requested.emit(name))
            tools_layout.addWidget(button, index // 7, index % 7)
        layout.addWidget(tools)

        self.tabs = QTabWidget()
        self.editor = ConfigTreeEditor()
        self.tabs.addTab(self.editor, "可视参数树")

        raw_page = QWidget()
        raw_layout = QVBoxLayout(raw_page)
        raw_buttons = QHBoxLayout()
        to_raw = QPushButton("参数树 → 刷新 TOML")
        from_raw = QPushButton("TOML → 应用到参数树")
        from_raw.setObjectName("primary")
        raw_buttons.addWidget(to_raw)
        raw_buttons.addWidget(from_raw)
        raw_buttons.addStretch(1)
        raw_layout.addLayout(raw_buttons)
        self.raw_toml = QPlainTextEdit()
        fixed = QFont("Menlo")
        fixed.setStyleHint(QFont.StyleHint.Monospace)
        self.raw_toml.setFont(fixed)
        self.raw_toml.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        raw_layout.addWidget(self.raw_toml, 1)
        self.raw_hint = QLabel("完整 TOML 可直接粘贴；点击“应用到参数树”后才会进入审核。")
        self.raw_hint.setObjectName("editorHint")
        raw_layout.addWidget(self.raw_hint)
        self.tabs.addTab(raw_page, "完整 TOML")
        layout.addWidget(self.tabs, 1)

        to_raw.clicked.connect(self.sync_tree_to_toml)
        from_raw.clicked.connect(self.apply_raw_toml)
        self.raw_toml.textChanged.connect(self._raw_changed)
        self.editor.mapping_changed.connect(self._tree_changed)

    def set_mapping(self, mapping: dict[str, Any]) -> None:
        self.editor.set_mapping(mapping)
        self.raw_toml.blockSignals(True)
        self.raw_toml.setPlainText(toml_dumps(mapping))
        self.raw_toml.blockSignals(False)
        self.raw_hint.setText("参数树与完整 TOML 已同步。")

    def mapping(self) -> dict[str, Any]:
        return self.editor.mapping()

    def sync_tree_to_toml(self) -> None:
        try:
            text = toml_dumps(self.editor.mapping())
        except Exception as exc:
            self.error.emit("无法生成 TOML", exc)
            return
        self.raw_toml.blockSignals(True)
        self.raw_toml.setPlainText(text)
        self.raw_toml.blockSignals(False)
        self.raw_hint.setText("已用可视参数树刷新完整 TOML。")

    def apply_raw_toml(self) -> None:
        try:
            mapping = parse_toml_bytes(self.raw_toml.toPlainText().encode("utf-8"))
        except Exception as exc:
            self.error.emit("TOML 解析失败", exc)
            return
        self.set_mapping(mapping)
        self.mapping_changed.emit()
        self.tabs.setCurrentIndex(0)

    def _tree_changed(self) -> None:
        self.raw_hint.setText("可视参数已修改；完整 TOML 尚未刷新。")
        self.mapping_changed.emit()

    def _raw_changed(self) -> None:
        self.raw_hint.setText("完整 TOML 已编辑；点击“应用到参数树”后生效。")


class ReviewPage(QWidget):
    review_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        heading_row = QHBoxLayout()
        heading_row.addWidget(
            PageHeading(
                "2  输入审核",
                "正演前一次看清 Vp、Vs、密度、源—接收器 Geometry、子波/PSD 和理论频散；required 失败会锁定计算。",
            ),
            1,
        )
        self.review_button = QPushButton("重新审核当前参数")
        self.review_button.setObjectName("primary")
        self.export_button = QPushButton("导出审核图与坐标")
        heading_row.addWidget(self.review_button)
        heading_row.addWidget(self.export_button)
        layout.addLayout(heading_row)

        self.status = QLabel("尚未审核")
        self.status.setStyleSheet(
            "background:#edf2f7;border-radius:7px;padding:9px 12px;font-weight:600;"
        )
        layout.addWidget(self.status)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure = FigurePane("载入或修改参数后，点击“重新审核当前参数”。")
        splitter.addWidget(self.figure)

        lower = QSplitter(Qt.Orientation.Horizontal)
        self.checks = QTableWidget(0, 6)
        self.checks.setHorizontalHeaderLabels(("状态", "检查", "当前值", "要求", "参数位置", "怎么改"))
        self.checks.setWordWrap(True)
        self.checks.verticalHeader().setVisible(False)
        self.checks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.checks.horizontalHeader().setStretchLastSection(True)
        lower.addWidget(self.checks)
        self.diagnostics = QTableWidget(0, 2)
        self.diagnostics.setHorizontalHeaderLabels(("诊断量", "值"))
        self.diagnostics.verticalHeader().setVisible(False)
        self.diagnostics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diagnostics.horizontalHeader().setStretchLastSection(True)
        lower.addWidget(self.diagnostics)
        lower.setSizes((1050, 380))
        splitter.addWidget(lower)
        splitter.setSizes((620, 270))
        layout.addWidget(splitter, 1)

        self.review_button.clicked.connect(self.review_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.export_button.setEnabled(False)

    def set_review(self, review: PreflightReview) -> None:
        failures = sum(not check.passed and check.level == "required" for check in review.checks)
        warnings = sum(not check.passed and check.level != "required" for check in review.checks)
        if review.ready:
            self.status.setText(f"✓ READY｜required 全部通过｜建议项警告 {warnings} 个")
            self.status.setStyleSheet(
                "background:#daf4e7;color:#096b41;border-radius:7px;padding:9px 12px;font-weight:700;"
            )
        else:
            self.status.setText(f"✕ BLOCKED｜required 失败 {failures} 个｜修正后才能开始计算")
            self.status.setStyleSheet(
                "background:#ffe1e2;color:#9f2228;border-radius:7px;padding:9px 12px;font-weight:700;"
            )
        self.figure.set_figure(plot_preflight_review(review))
        self.checks.setRowCount(len(review.checks))
        for row, check in enumerate(review.checks):
            if check.passed:
                status = "✓ PASS"
            elif check.level == "required":
                status = "✕ FAIL"
            else:
                status = "⚠ WARN"
            values = (status, check.name, check.value, check.criterion, check.parameter_path, check.guidance)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if status == "✕ FAIL":
                    item.setBackground(Qt.GlobalColor.red if column == 0 else Qt.GlobalColor.transparent)
                self.checks.setItem(row, column, item)
        self.checks.resizeColumnsToContents()
        self.checks.setColumnWidth(5, max(330, self.checks.columnWidth(5)))
        self.checks.resizeRowsToContents()

        self.diagnostics.setRowCount(len(review.diagnostics))
        for row, (key, value) in enumerate(review.diagnostics.items()):
            self.diagnostics.setItem(row, 0, QTableWidgetItem(str(key)))
            self.diagnostics.setItem(row, 1, QTableWidgetItem(str(value)))
        self.diagnostics.resizeColumnsToContents()
        self.export_button.setEnabled(True)


class RunPage(QWidget):
    stage_requested = Signal(str)

    LABELS = {
        "review": "输入审核",
        "forward": "正演 / Green 传递",
        "observations": "波形 / 接收端 NCF",
        "analysis": "频散成像与拾取",
        "all": "一键运行并保存全部",
        "save": "保存当前结果",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.addWidget(
            PageHeading(
                "3  分阶段计算",
                "计算在后台线程运行。各阶段共享内存缓存；仅受修改影响的下游结果会失效，无需重复已完成的正演。",
            )
        )
        button_row = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for key in ("forward", "observations", "analysis", "all", "save"):
            button = QPushButton(self.LABELS[key])
            if key == "all":
                button.setObjectName("success")
            button.clicked.connect(lambda checked=False, stage=key: self.stage_requested.emit(stage))
            button_row.addWidget(button)
            self.buttons[key] = button
        button_row.addStretch(1)
        layout.addLayout(button_row)

        progress_row = QHBoxLayout()
        self.current = QLabel("等待已通过的输入审核")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        progress_row.addWidget(self.current, 1)
        progress_row.addWidget(self.progress, 1)
        layout.addLayout(progress_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure = FigurePane("运行某个阶段后，这里显示对应的波场、混合记录/NCF 或频散能量。")
        splitter.addWidget(self.figure)
        lower = QSplitter(Qt.Orientation.Horizontal)
        self.status_table = QTableWidget(5, 3)
        self.status_table.setHorizontalHeaderLabels(("阶段", "状态", "耗时 s"))
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.status_table.horizontalHeader().setStretchLastSection(True)
        lower.addWidget(self.status_table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("运行日志和错误详情会出现在这里。")
        lower.addWidget(self.log)
        lower.setSizes((460, 980))
        splitter.addWidget(lower)
        splitter.setSizes((650, 220))
        layout.addWidget(splitter, 1)

    def set_running(self, running: bool, stage: str = "") -> None:
        for button in self.buttons.values():
            button.setEnabled(not running)
        if running:
            self.current.setText(f"正在运行：{self.LABELS.get(stage, stage)}")
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{timestamp}] {message}")

    def update_status(self, rows: list[dict[str, object]]) -> None:
        self.status_table.setRowCount(len(rows))
        names = {
            "inputs": "输入审核",
            "forward": "正演 / Green",
            "observations": "波形 / NCF",
            "analysis": "频散分析",
            "saved": "结果保存",
        }
        for row, status in enumerate(rows):
            seconds = status.get("seconds")
            values = (
                names.get(str(status["stage"]), str(status["stage"])),
                str(status["status"]),
                "—" if seconds is None else f"{float(seconds):.3f}",
            )
            for column, value in enumerate(values):
                self.status_table.setItem(row, column, QTableWidgetItem(value))
        self.status_table.resizeColumnsToContents()


class ResultsPage(QWidget):
    open_requested = Signal()
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._directory: Path | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        heading_row = QHBoxLayout()
        heading_row.addWidget(
            PageHeading(
                "4  结果浏览与导出",
                "浏览完整 summary、分量/方法对照图，并核对 HDF5、CSV、manifest 与可复现 input.toml。",
            ),
            1,
        )
        refresh = QPushButton("刷新结果")
        open_folder = QPushButton("打开输出文件夹")
        open_folder.setObjectName("primary")
        heading_row.addWidget(refresh)
        heading_row.addWidget(open_folder)
        layout.addLayout(heading_row)
        self.directory_label = QLabel("尚无已保存结果")
        self.directory_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.directory_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.images = QListWidget()
        self.images.setObjectName("gallery")
        self.images.setMinimumWidth(245)
        splitter.addWidget(self.images)

        right = QSplitter(Qt.Orientation.Vertical)
        self.image_label = QLabel("保存结果后选择 PNG 查看")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#20262e;color:white;")
        self.image_label.setMinimumSize(500, 380)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)
        right.addWidget(self.scroll)
        self.artifacts = QTableWidget(0, 3)
        self.artifacts.setHorizontalHeaderLabels(("文件", "大小", "绝对路径"))
        self.artifacts.verticalHeader().setVisible(False)
        self.artifacts.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifacts.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.artifacts)
        right.setSizes((650, 250))
        splitter.addWidget(right)
        splitter.setSizes((260, 1180))
        layout.addWidget(splitter, 1)

        refresh.clicked.connect(self.refresh_requested)
        open_folder.clicked.connect(self.open_requested)
        self.images.currentItemChanged.connect(self._show_image)

    def load_directory(self, directory: Path | None) -> None:
        self._directory = directory
        self.images.clear()
        self.artifacts.setRowCount(0)
        self.image_label.setText("未找到已保存的 PNG")
        self.image_label.setPixmap(QPixmap())
        if directory is None or not directory.is_dir():
            self.directory_label.setText("尚无已保存结果")
            return
        self.directory_label.setText(str(directory))
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        images = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        for path in images:
            item = QListWidgetItem(path.relative_to(directory).as_posix())
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.images.addItem(item)
        self.artifacts.setRowCount(len(files))
        for row, path in enumerate(files):
            size = path.stat().st_size
            display_size = f"{size / 2**20:.2f} MiB" if size >= 2**20 else f"{size / 1024:.1f} KiB"
            for column, value in enumerate((path.name, display_size, str(path))):
                self.artifacts.setItem(row, column, QTableWidgetItem(value))
        self.artifacts.resizeColumnsToContents()
        if images:
            summary_row = next((i for i, path in enumerate(images) if path.name == "summary.png"), 0)
            self.images.setCurrentRow(summary_row)

    def _show_image(self, current: QListWidgetItem | None, previous=None) -> None:
        if current is None:
            return
        path = Path(str(current.data(Qt.ItemDataRole.UserRole)))
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image_label.setText(f"无法读取图像：{path}")
            return
        viewport = self.scroll.viewport().size()
        target = QSize(max(900, viewport.width() - 20), max(600, viewport.height() - 20))
        scaled = pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())


class GuidePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.addWidget(PageHeading("5  流程与方法", "软件界面不改变计算定义；这里给出当前实现的关键边界。"))
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(
            """
            <h2>WaveDispX 计算链</h2>
            <ol>
              <li><b>输入审核：</b>从层状 Vp/Vs/ρ/Q、源和接收器构建 geometry，画子波/PSD，并独立计算理论 Rayleigh/Love 曲线。</li>
              <li><b>正演：</b>用分层介质离散波数 Green 张量计算频率域源—接收器传递；主动源使用 Ricker 或理想单位脉冲时间函数，被动源保留带限传递函数。</li>
              <li><b>真实被动记录：</b>每个源在每个独立时间窗生成随机源时间函数，全部源在接收器端线性混合，程序只对混合后的接收器记录做互相关/互谱累积。没有把逐源理想互相关当作有限记录结果。</li>
              <li><b>频散成像：</b>主动默认 Hankel；被动默认 phase_shift。选择 compare_methods 可在同一网格对照 phase_shift、Bessel(FJ) 和 Hankel，识别方法相关旁瓣。</li>
              <li><b>拾取与理论：</b>动态规划在归一化能量上拾取连续脊线；白色理论线由同一层状模型独立计算，不参与正演频散能量的生成。</li>
            </ol>
            <h3>三种成像核</h3>
            <p><b>相移法：</b>按候选相速度 c 施加 exp(+iωr/c) 并沿距离叠加。直观、适合径向 NCF 道集。</p>
            <p><b>FJ / Bessel：</b>用 J₀(ωr/c) 核做频率—Bessel 变换，通常更符合二维轴对称波场，但有限孔径仍会产生核旁瓣。</p>
            <p><b>Hankel：</b>使用出射波 Hankel 核，主动炮集默认；对近场、体波和不规则检距的敏感性与相移法不同。</p>
            <h3>被动收敛</h3>
            <p>源数主要改善方位照明；记录时长和独立窗数主要降低有限样本互相关方差。二者不能互相完全替代。阵列孔径决定低频分辨率，道距/径向距离箱决定高频空间混叠。</p>
            <h3>文件格式</h3>
            <ul>
              <li>接收器/噪声源：<code>x_m,y_m</code></li>
              <li>主动炮点：<code>shot_id,x_m,y_m</code>（shot_id 可选）</li>
              <li>外部 PSD：<code>frequency_hz,power</code></li>
              <li>DAS 逐道光缆：<code>channel_id,x_m,y_m,endpoint_minus_x_m,endpoint_minus_y_m,endpoint_plus_x_m,endpoint_plus_y_m</code>；也可用每道 <code>gauge_length_m,fiber_azimuth_deg</code></li>
            </ul>
            """
        )
        layout.addWidget(browser, 1)


class MainWindow(QMainWindow):
    """WaveDispX desktop shell around the shared staged workflow."""

    PAGE_NAMES = ("建立算例", "输入审核", "分阶段计算", "结果浏览", "流程与方法")

    def __init__(self, template_path: str | Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WaveDispX Desktop Workbench")
        self.setMinimumSize(1220, 780)
        self.resize(1580, 980)
        self.setStyleSheet(APP_STYLE)
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self._running = False
        self._dirty = True
        self._review: PreflightReview | None = None
        self._current_config = None
        self._active_worker: WorkflowWorker | None = None

        initial = Path(template_path).expanduser().resolve() if template_path else TEMPLATES["被动源｜快速检查"]
        self.session = CaseSession(initial.parent)
        self._build_ui()
        for template_name, template_file in TEMPLATES.items():
            if template_file.resolve() == initial:
                self.template_combo.setCurrentText(template_name)
                break
        self._connect_ui()
        self._load_mapping(load_template_mapping(initial), initial.parent, f"已载入 {initial.name}")
        self.validate_case(navigate=False, show_ready_dialog=False)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(205)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 22, 12, 16)
        brand = QLabel("WaveDispX")
        brand.setObjectName("brand")
        tagline = QLabel("主动 / 被动波场与频散工作台")
        tagline.setObjectName("tagline")
        tagline.setWordWrap(True)
        side_layout.addWidget(brand)
        side_layout.addWidget(tagline)
        side_layout.addSpacing(24)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        for number, name in enumerate(self.PAGE_NAMES, start=1):
            self.navigation.addItem(QListWidgetItem(f"{number}   {name}"))
        self.navigation.setCurrentRow(0)
        side_layout.addWidget(self.navigation, 1)
        version = QLabel("v0.2 · shared core")
        version.setObjectName("tagline")
        side_layout.addWidget(version)
        root.addWidget(sidebar)

        work = QWidget()
        work_layout = QVBoxLayout(work)
        work_layout.setContentsMargins(12, 10, 12, 8)
        topbar = QFrame()
        topbar.setObjectName("topbar")
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(10, 7, 10, 7)
        top_layout.addWidget(QLabel("起始模板"))
        self.template_combo = QComboBox()
        self.template_combo.addItems(TEMPLATES.keys())
        self.template_combo.setCurrentText("被动源｜快速检查")
        self.template_combo.setMinimumWidth(250)
        self.template_combo.setToolTip(
            "选择一套完整、已通过输入审核的起始参数。quick 用于几分钟内验证流程；正式/高阶模板计算量更大。"
        )
        top_layout.addWidget(self.template_combo)
        self.load_template_button = QPushButton("载入模板")
        self.open_button = QPushButton("打开 TOML…")
        self.save_button = QPushButton("保存 TOML…")
        self.load_template_button.setToolTip("用当前模板完整替换参数树；不会立即开始正演。")
        self.open_button.setToolTip("读取本地完整 TOML；其中相对 CSV 路径按 TOML 所在目录解析。")
        self.save_button.setToolTip("把当前可视参数树保存为可复现 TOML；保存前不会自动运行正演。")
        top_layout.addWidget(self.load_template_button)
        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.save_button)
        top_layout.addStretch(1)
        self.review_now_button = QPushButton("审核模型与 Geometry")
        self.review_now_button.setObjectName("primary")
        self.review_now_button.setToolTip(
            "不运行波场正演。先检查 Vp/Vs/ρ、源—接收器位置、子波/PSD、理论频散、Nyquist、空间混叠和孔径。"
        )
        top_layout.addWidget(self.review_now_button)
        work_layout.addWidget(topbar)

        self.stack = QStackedWidget()
        self.parameter_page = ParameterPage()
        self.review_page = ReviewPage()
        self.run_page = RunPage()
        self.results_page = ResultsPage()
        self.guide_page = GuidePage()
        for page in (
            self.parameter_page,
            self.review_page,
            self.run_page,
            self.results_page,
            self.guide_page,
        ):
            self.stack.addWidget(page)
        work_layout.addWidget(self.stack, 1)
        root.addWidget(work, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("WaveDispX 桌面工作台已就绪")

    def _connect_ui(self) -> None:
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.load_template_button.clicked.connect(self.load_selected_template)
        self.open_button.clicked.connect(self.open_toml)
        self.save_button.clicked.connect(self.save_toml)
        self.review_now_button.clicked.connect(self.validate_case)
        self.parameter_page.mapping_changed.connect(self._mark_dirty)
        self.parameter_page.error.connect(self._show_error)
        self.parameter_page.action_requested.connect(self._parameter_action)
        self.review_page.review_requested.connect(self.validate_case)
        self.review_page.export_requested.connect(self.export_review)
        self.run_page.stage_requested.connect(self.start_stage)
        self.results_page.refresh_requested.connect(self.refresh_results)
        self.results_page.open_requested.connect(self.open_output_directory)

    @property
    def workflow(self):
        return self.session.workflow

    def select_page(self, page: str | int) -> None:
        if isinstance(page, int):
            index = page
        else:
            aliases = {
                "parameters": 0,
                "review": 1,
                "run": 2,
                "results": 3,
                "guide": 4,
            }
            index = aliases.get(page, self.PAGE_NAMES.index(page) if page in self.PAGE_NAMES else 0)
        self.navigation.setCurrentRow(index)

    def _load_mapping(self, mapping: dict[str, Any], base: Path, message: str) -> None:
        mapping = deepcopy(mapping)
        mapping.setdefault("processing", {}).setdefault("frequency_step_hz", 0.0)
        if mapping.get("run", {}).get("mode") == "passive":
            mapping.setdefault("noise", {}).setdefault("maximum_lag_s", 0.0)
        else:
            mapping.setdefault("active_source", {}).setdefault("record_duration_s", 0.0)
        self.session = CaseSession(base)
        self.parameter_page.set_mapping(mapping)
        self._dirty = True
        self._review = None
        self._current_config = None
        self.review_page.figure.show_placeholder("参数已载入；正在等待输入审核。")
        self.run_page.figure.show_placeholder("完成输入审核后可分阶段计算。")
        self.results_page.load_directory(None)
        self.run_page.append_log(message)
        self.statusBar().showMessage(message, 8000)

    def load_selected_template(self) -> None:
        name = self.template_combo.currentText()
        path = TEMPLATES[name]
        self._load_mapping(load_template_mapping(path), path.parent, f"已载入模板：{name}")
        self.validate_case(navigate=False, show_ready_dialog=False)

    def open_toml(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "打开 WaveDispX TOML", str(PROJECT_ROOT), "TOML (*.toml)")
        if not filename:
            return
        path = Path(filename).resolve()
        try:
            mapping = parse_toml_bytes(path.read_bytes())
        except Exception as exc:
            self._show_error("无法打开 TOML", exc)
            return
        self._load_mapping(mapping, path.parent, f"已打开：{path}")
        self.validate_case(navigate=False, show_ready_dialog=False)

    def save_toml(self) -> None:
        try:
            mapping = self.parameter_page.mapping()
            content = toml_dumps(mapping)
        except Exception as exc:
            self._show_error("无法保存 TOML", exc)
            return
        suggested = str(self.session.workspace_directory / f"{mapping.get('run', {}).get('name', 'wavedispx_case')}.toml")
        filename, _ = QFileDialog.getSaveFileName(self, "保存 WaveDispX TOML", suggested, "TOML (*.toml)")
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".toml":
            path = path.with_suffix(".toml")
        path.write_text(content, encoding="utf-8")
        self.statusBar().showMessage(f"已保存：{path}", 10000)
        self.run_page.append_log(f"配置已保存：{path}")

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.review_now_button.setText("参数已修改｜重新审核")
        self.statusBar().showMessage("参数已修改；已锁定计算，重新审核后生效。")

    def validate_case(self, checked=False, *, navigate: bool = True, show_ready_dialog: bool = False) -> bool:
        if self._running:
            return False
        try:
            mapping = self.parameter_page.mapping()
            config, invalidated = self.session.apply_mapping(mapping)
            review = self.session.run_stage("review")
        except Exception as exc:
            self._show_error("输入审核失败", exc)
            if navigate:
                self.select_page("parameters")
            return False
        self._current_config = config
        self._review = review
        self._dirty = False
        self.review_now_button.setText("审核模型与 Geometry")
        self.review_page.set_review(review)
        self.run_page.update_status(self.workflow.status())
        if invalidated:
            self.run_page.append_log("参数已应用；失效阶段：" + ", ".join(invalidated))
        diagnostics = review.diagnostics
        message = (
            f"输入审核 {'通过' if review.ready else '未通过'}｜"
            f"{diagnostics.get('receiver_count')} receivers｜{diagnostics.get('source_count')} sources"
        )
        self.run_page.append_log(message)
        self.statusBar().showMessage(message, 10000)
        if navigate:
            self.select_page("review")
        if show_ready_dialog and review.ready:
            QMessageBox.information(self, "输入审核通过", "所有 required 检查通过，可以进入正演。")
        return bool(review.ready)

    def export_review(self) -> None:
        if self._review is None:
            self.validate_case()
        if self._review is None:
            return
        default = self._current_config.output_directory / "preflight"
        directory = QFileDialog.getExistingDirectory(self, "选择输入审核导出目录", str(default.parent))
        if not directory:
            return
        try:
            outputs = save_preflight_review(self._review, Path(directory))
        except Exception as exc:
            self._show_error("导出输入审核失败", exc)
            return
        self.run_page.append_log("输入审核已导出：" + str(outputs["figure"]))
        QMessageBox.information(self, "导出完成", f"审核图、检查报告、源和接收器坐标已保存到：\n{directory}")

    def _parameter_action(self, action: str) -> None:
        try:
            mapping = self.parameter_page.mapping()
            if action == "apply_auto_design":
                mapping, diagnostics = apply_automatic_parameters(
                    mapping, base_directory=self.session.workspace_directory
                )
                self.run_page.append_log(
                    "已套用自动推荐："
                    f"f={float(diagnostics['recommended_frequency_min_hz']):.2f}–"
                    f"{float(diagnostics['recommended_frequency_max_hz']):.2f} Hz，"
                    f"df={float(diagnostics['recommended_display_frequency_step_hz']):.4g} Hz"
                )
            elif action == "reset_auto_design":
                template = load_template_mapping(
                    TEMPLATES[self.template_combo.currentText()]
                )
                mode = mapping.get("run", {}).get("mode")
                if template.get("run", {}).get("mode") != mode:
                    template = load_template_mapping(
                        TEMPLATES[
                            "主动源｜快速检查"
                            if mode == "active"
                            else "被动源｜快速检查"
                        ]
                    )
                mapping = reset_automatic_parameters(mapping, template)
                self.run_page.append_log("已恢复当前模式模板的自动参数默认值。")
            elif action == "add_layer":
                layers = mapping.setdefault("model", {}).setdefault("layers", [])
                if not layers:
                    layers.append({"thickness_m": float("inf"), "vp_mps": 760.0, "vs_mps": 440.0, "density_kgm3": 2000.0, "qp": float("inf"), "qs": float("inf")})
                reference = deepcopy(layers[-2] if len(layers) > 1 else layers[-1])
                reference["thickness_m"] = 10.0
                layers.insert(max(0, len(layers) - 1), reference)
            elif action == "remove_layer":
                layers = mapping["model"]["layers"]
                if len(layers) <= 1:
                    raise CaseInputError("模型至少需要保留一个半空间层")
                layers.pop(-2)
            elif action == "receiver_line":
                mode = mapping.get("run", {}).get("mode", "passive")
                mapping.setdefault("geometry", {})["receivers"] = {
                    "layout": "line",
                    "start_x_m": 5.0 if mode == "active" else 0.0,
                    "stop_x_m": 80.0 if mode == "active" else 200.0,
                    "spacing_m": 5.0,
                    "y_m": 0.0,
                }
            elif action == "receiver_grid":
                mapping.setdefault("geometry", {})["receivers"] = {
                    "layout": "grid",
                    "start_x_m": -50.0,
                    "stop_x_m": 50.0,
                    "spacing_x_m": 5.0,
                    "start_y_m": -50.0,
                    "stop_y_m": 50.0,
                    "spacing_y_m": 5.0,
                    "distance_bin_m": 5.0,
                }
            elif action == "receiver_file":
                self._choose_receiver_file(mapping)
            elif action == "das_channel_file":
                self._choose_das_channel_file(mapping)
            elif action == "source_annulus":
                self._require_passive(mapping)
                current = mapping.setdefault("ambient_sources", {})
                mapping["ambient_sources"] = {
                    "component": current.get("component", "vertical"),
                    "layout": "annulus",
                    "count": max(192, int(current.get("count", 192))),
                    "seed": int(current.get("seed", 42)),
                    "inner_radius_m": 250.0,
                    "outer_radius_m": 450.0,
                    "angular_sampling": "stratified",
                }
            elif action == "source_line":
                self._require_passive(mapping)
                current = mapping.setdefault("ambient_sources", {})
                mapping["ambient_sources"] = {
                    "component": current.get("component", "vertical"),
                    "layout": "line",
                    "count": int(current.get("count", 192)),
                    "seed": int(current.get("seed", 42)),
                    "x_min_m": -500.0,
                    "x_max_m": 500.0,
                    "y_m": 300.0,
                }
            elif action == "source_file":
                self._choose_source_file(mapping)
            elif action == "shot_file":
                self._choose_shot_file(mapping)
            elif action == "psd_file":
                self._choose_psd_file(mapping)
            else:
                return
        except Exception as exc:
            self._show_error("参数操作失败", exc)
            return
        self.parameter_page.set_mapping(mapping)
        self._mark_dirty()

    def _require_passive(self, mapping: dict[str, Any]) -> None:
        if mapping.get("run", {}).get("mode") != "passive":
            raise CaseInputError("该操作仅用于 passive 模式；请先载入被动源模板")

    def _choose_receiver_file(self, mapping: dict[str, Any]) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择接收器坐标 CSV", str(PROJECT_ROOT / "examples"), "CSV (*.csv)")
        if not filename:
            raise CaseInputError("未选择接收器 CSV")
        path = Path(filename).resolve()
        count = coordinate_csv_count(path.read_bytes())
        if count < 2:
            raise CaseInputError("接收器 CSV 至少需要两个坐标")
        receivers: dict[str, Any] = {"layout": "file", "coordinates_file": str(path)}
        if mapping.get("run", {}).get("mode") == "passive":
            receivers["distance_bin_m"] = 5.0
        mapping.setdefault("geometry", {})["receivers"] = receivers

    def _choose_das_channel_file(self, mapping: dict[str, Any]) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择逐道 DAS 光缆 CSV",
            str(PROJECT_ROOT / "examples"),
            "CSV (*.csv)",
        )
        if not filename:
            raise CaseInputError("未选择 DAS 光缆 CSV")
        path = Path(filename).resolve()
        das_geometry = load_das_geometry_csv(path)
        receivers: dict[str, Any] = {
            "layout": "file",
            "coordinates_file": str(path),
        }
        if mapping.get("run", {}).get("mode") == "passive":
            receivers["distance_bin_m"] = 5.0
        mapping.setdefault("geometry", {})["receivers"] = receivers
        das = mapping.setdefault("das", {})
        das.update(
            enabled=True,
            geometry_mode="channel_file",
            measurement=das.get("measurement", "strain_rate"),
            gauge_length_m=float(
                np.median(das_geometry.gauge_length_by_channel_m)
            ),
            fiber_azimuth_deg=0.0,
        )

    def _choose_source_file(self, mapping: dict[str, Any]) -> None:
        self._require_passive(mapping)
        filename, _ = QFileDialog.getOpenFileName(self, "选择噪声源坐标 CSV", str(PROJECT_ROOT / "examples"), "CSV (*.csv)")
        if not filename:
            raise CaseInputError("未选择噪声源 CSV")
        path = Path(filename).resolve()
        count = coordinate_csv_count(path.read_bytes())
        current = mapping.setdefault("ambient_sources", {})
        mapping["ambient_sources"] = {
            "component": current.get("component", "vertical"),
            "layout": "file",
            "count": count,
            "seed": int(current.get("seed", 42)),
            "coordinates_file": str(path),
        }

    def _choose_shot_file(self, mapping: dict[str, Any]) -> None:
        if mapping.get("run", {}).get("mode") != "active":
            raise CaseInputError("炮点 CSV 仅用于 active 模式；请先载入主动源模板")
        filename, _ = QFileDialog.getOpenFileName(self, "选择炮点 CSV", str(PROJECT_ROOT / "examples"), "CSV (*.csv)")
        if not filename:
            raise CaseInputError("未选择炮点 CSV")
        shots = parse_shot_csv(Path(filename).read_bytes())
        labels = [f"{shot.shot_id}｜x={shot.x_m:g}, y={shot.y_m:g}" for shot in shots]
        selected, accepted = QInputDialog.getItem(self, "选择本次主动炮点", "炮点", labels, 0, False)
        if not accepted:
            raise CaseInputError("尚未从炮点表选择本次运行的炮点")
        shot = shots[labels.index(selected)]
        active = mapping.setdefault("active_source", {})
        active["x_m"] = shot.x_m
        active["y_m"] = shot.y_m

    def _choose_psd_file(self, mapping: dict[str, Any]) -> None:
        self._require_passive(mapping)
        filename, _ = QFileDialog.getOpenFileName(self, "选择噪声 PSD CSV", str(PROJECT_ROOT / "examples"), "CSV (*.csv)")
        if not filename:
            raise CaseInputError("未选择 PSD CSV")
        noise = mapping.setdefault("noise", {})
        noise["psd_shape"] = "file"
        noise["psd_file"] = str(Path(filename).resolve())

    def start_stage(self, stage: str) -> None:
        if self._running:
            return
        if self._dirty or self.workflow is None or self._review is None:
            if not self.validate_case(navigate=False, show_ready_dialog=False):
                self.select_page("review")
                QMessageBox.warning(self, "计算已锁定", "输入审核未通过。请根据 required 失败项修改参数。")
                return
        if not self._review.ready:
            self.select_page("review")
            QMessageBox.warning(self, "计算已锁定", "输入审核存在 required 失败项。")
            return
        if stage == "save":
            action = self.workflow.save_results
        else:
            action = lambda: self.session.run_stage(stage)
        worker = WorkflowWorker(stage, action)
        worker.signals.succeeded.connect(self._stage_succeeded)
        worker.signals.failed.connect(self._stage_failed)
        self._running = True
        self._active_worker = worker
        self.run_page.set_running(True, stage)
        self.run_page.append_log(f"开始：{RunPage.LABELS.get(stage, stage)}")
        self.select_page("run")
        self.thread_pool.start(worker)

    @Slot(str, object)
    def _stage_succeeded(self, stage: str, result: object) -> None:
        self._running = False
        self._active_worker = None
        self.run_page.set_running(False, stage)
        self.run_page.current.setText(f"完成：{RunPage.LABELS.get(stage, stage)}")
        self.run_page.update_status(self.workflow.status())
        self.run_page.append_log(f"完成：{RunPage.LABELS.get(stage, stage)}")
        try:
            if stage == "forward":
                figure = plot_workflow_forward(self.workflow)
            elif stage == "observations":
                figure = plot_workflow_observations(self.workflow)
            else:
                figure = plot_workflow_dispersion(self.workflow)
            self.run_page.figure.set_figure(figure)
        except Exception as exc:
            self.run_page.append_log(f"阶段完成，但界面绘图失败：{exc}")
        if stage in {"all", "save"}:
            self.refresh_results()
            self.select_page("results")
        self.statusBar().showMessage(f"已完成：{RunPage.LABELS.get(stage, stage)}", 10000)

    @Slot(str, str, str)
    def _stage_failed(self, stage: str, message: str, details: str) -> None:
        self._running = False
        self._active_worker = None
        self.run_page.set_running(False, stage)
        self.run_page.current.setText(f"失败：{RunPage.LABELS.get(stage, stage)}")
        self.run_page.append_log(f"失败：{message}\n{details}")
        self._show_error(f"{RunPage.LABELS.get(stage, stage)}失败", RuntimeError(message), details)

    def refresh_results(self) -> None:
        directory = None
        if self.workflow is not None:
            directory = self.workflow.config.output_directory
        self.results_page.load_directory(directory)

    def open_output_directory(self) -> None:
        if self.workflow is None:
            self._show_error("没有输出目录", CaseInputError("请先建立并校验算例"))
            return
        directory = self.workflow.config.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _show_error(self, title: str, error: object, details: str = "") -> None:
        exc = error if isinstance(error, Exception) else RuntimeError(str(error))
        issue = explain_input_error(exc)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(f"参数位置：{issue.field_path}\n\n问题：{issue.problem}\n\n怎么改：{issue.suggestion}")
        box.setInformativeText(issue.technical_message)
        if details:
            box.setDetailedText(details)
        box.exec()
        self.statusBar().showMessage(f"{title}：{issue.technical_message}", 15000)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._running:
            QMessageBox.warning(self, "计算仍在运行", "请等待当前阶段结束后再关闭窗口。")
            event.ignore()
            return
        event.accept()
