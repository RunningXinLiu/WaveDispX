"""Studio-only visual tokens; no scientific display transforms."""

STUDIO_STYLE = """
QMainWindow, QWidget#studioRoot { background: #edf2f7; color: #182c43; }
QWidget { font-size: 13px; color: #182c43; }
QWidget#studioHeader { background: #142b46; border-radius: 9px; }
QLabel#studioBrand { color: white; font-size: 22px; font-weight: 700; }
QLabel#studioSubtitle { color: #bfd0e5; font-size: 12px; }
QLabel#caseBadge { color: #153b63; background: #dceafb; border-radius: 6px;
    padding: 6px 10px; font-weight: 600; }
QLabel#sectionTitle { font-size: 14px; font-weight: 700; color: #193e67; }
QLabel#fieldLabel { color: #405673; font-size: 12px; }
QLabel#hint { color: #536981; font-size: 12px; }
QWidget#inputCard { background: white; border-radius: 8px; }
QGroupBox { background: #f8fafc; border: 1px solid #cbd6e3; border-radius: 7px;
    margin-top: 10px; padding: 12px 8px 6px; font-weight: 600; color: #244b75; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background: white; border: 1px solid #aebfd1; border-radius: 6px;
    padding: 7px 12px; min-height: 22px; font-weight: 600; }
QPushButton:hover { background: #e9f2ff; border-color: #4681c4; }
QPushButton:pressed { background: #d2e5ff; }
QPushButton:focus { border: 2px solid #2563eb; }
QPushButton[role="primary"] { background: #1d5ed2; color: white; border: 1px solid #1d5ed2; }
QPushButton[role="primary"]:hover { background: #164bb0; }
QPushButton[role="secondary"] { background: #e4efff; color: #174c93; border-color: #91b4df; }
QPushButton[role="save"] { background: #e1f2ed; color: #14624e; border-color: #83b6a5; }
QPushButton:disabled { background: #e7ecf2; color: #8493a6; border-color: #d0dae6; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white; color: #112e50;
    border: 1px solid #a5b9d0; border-radius: 5px; padding: 3px 7px;
    min-height: 20px; selection-background-color: #c3ddff; font-weight: 500; }
QLineEdit:focus, QComboBox:focus { border: 2px solid #2563eb; background: #f4f9ff; }
QLineEdit:disabled, QComboBox:disabled { color: #8291a2; background: #ecf0f5; }
QComboBox::drop-down { border: none; width: 22px; }
QTabWidget::pane { border: 1px solid #c4d2e2; background: white; border-radius: 6px; }
QTabBar::tab { background: #e3eaf3; color: #3e5876; padding: 8px 17px;
    margin-right: 3px; border: 1px solid #c4d2e2; border-top-left-radius: 6px;
    border-top-right-radius: 6px; font-weight: 600; }
QTabBar::tab:selected { background: #1d5ed2; color: white; border-color: #1d5ed2; }
QTabBar::tab:hover:!selected { background: #d2e3f8; }
QTabBar::tab:disabled { color: #91a0b2; background: #edf1f5; }
QTableWidget { background: white; alternate-background-color: #f2f6fc;
    gridline-color: #dce5ef; border: none; selection-background-color: #d4e5ff; selection-color: #102c4f; }
QHeaderView::section { background: #e6eef9; color: #234b77; padding: 6px;
    border: none; border-right: 1px solid #c9d8eb; font-weight: 600; }
QScrollArea { border: none; background: white; }
QScrollBar:vertical { background: #edf2f7; width: 9px; }
QScrollBar::handle:vertical { background: #9bafc7; border-radius: 4px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox { spacing: 6px; color: #2c486b; }
QCheckBox::indicator { width: 17px; height: 17px; }
QLabel#stateChip { padding: 6px 12px; border-radius: 6px; font-weight: 700; }
QLabel#stateChip[tone="dirty"] { background: #fff0ca; color: #774b00; }
QLabel#stateChip[tone="ready"] { background: #dceafb; color: #174d8f; }
QLabel#stateChip[tone="running"] { background: #1d5ed2; color: white; }
QLabel#stateChip[tone="result"] { background: #d9f0e5; color: #165a40; }
QLabel#stateChip[tone="error"] { background: #fce0e0; color: #962b2b; }
QLabel#samplingSummary { color: #455e7c; font-size: 12px; padding: 3px 0; }
QProgressBar { border: none; background: #d8e4f4; max-height: 3px; min-height: 3px; }
QProgressBar::chunk { background: #1d5ed2; }
QToolTip { background: #fff9e9; color: #173b61; border: 1px solid #c99b36; padding: 9px; }
"""
