"""Compact PySide6 classroom with one backend and a lazy ideal overlay."""
from pathlib import Path
import sys
from .qt_runtime import configure_qt
configure_qt()

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QTableWidget, QTableWidgetItem, QComboBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QMessageBox, QTabWidget, QLineEdit, QGroupBox, QHeaderView, QSizePolicy)

from .engine import PRESETS, COLUMNS, model_from_rows, model_csv, read_model, calculate, add_ideal
from .plotting import draw_model, draw_wave, export_result
from .help_text import TIPS


class Signals(QObject):
    progress = Signal(str)
    done = Signal(object)
    error = Signal(str)


class Job(QRunnable):
    def __init__(self, function, kwargs):
        super().__init__()
        self.function, self.kwargs = function, kwargs
        self.signals = Signals()

    def run(self):
        try:
            self.signals.done.emit(self.function(**self.kwargs, progress=self.signals.progress.emit))
        except Exception as exc:
            self.signals.error.emit(f'{type(exc).__name__}: {exc}')


def spin(low, high, value, decimals=None):
    obj = QSpinBox() if decimals is None else QDoubleSpinBox()
    obj.setRange(low, high)
    if decimals is not None:
        obj.setDecimals(decimals)
    obj.setValue(value)
    return obj


class ClassroomWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WaveDispX Classroom · 弹性层状介质频散实验室')
        self.resize(1440, 900)
        self.setMinimumSize(1200, 780)
        self.result = None
        self.busy = False
        self.pool = QThreadPool(self)
        self.setStyleSheet('''QWidget {font-size:13px; color:#203448;}
            QMainWindow {background:#f3f6fa;} QGroupBox {font-weight:600; margin-top:9px; padding-top:12px;}
            QPushButton {padding:5px 10px;} QTableWidget {background:white;}
            QTabWidget::pane {background:white; border:1px solid #d5dfe9;}
            QPushButton#run {background:#137e91; color:white; font-weight:600; border:0; border-radius:5px; padding:8px 20px;}
            QLabel#title {font-size:21px; font-weight:700; color:#0d5868;}''')
        main = QWidget()
        outer = QVBoxLayout(main)
        self.setCentralWidget(main)
        header = QHBoxLayout()
        outer.addLayout(header)
        title = QLabel('弹性层状介质 · 频散实验室')
        title.setObjectName('title')
        header.addWidget(title)
        header.addStretch()
        self.run_button = QPushButton('计算频散曲线')
        self.run_button.setObjectName('run')
        self.run_button.setToolTip('按当前模型、频带及模态数运行所选后端，同时计算 R/L 曲线。若已勾选 ideal，曲线完成后再计算能量叠加。')
        self.export_button = QPushButton('导出结果 ZIP')
        self.export_button.setEnabled(False)
        self.export_button.setToolTip('导出所选后端的 R/L 曲线 CSV、模型、配置、图片、原始数组和来源记录。分别计算并导出两套 backend，即可在软件外对比。')
        header.addWidget(self.run_button)
        header.addWidget(self.export_button)

        self.editor = QWidget()
        self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        top = QVBoxLayout(self.editor)
        top.setContentsMargins(0,0,0,0)
        outer.addWidget(self.editor)
        panels = QHBoxLayout()
        top.addLayout(panels)
        model_box = QGroupBox('层模型 · h / Vp / Vs / ρ 独立输入')
        model_layout = QVBoxLayout(model_box)
        panels.addWidget(model_box, 6)
        row = QHBoxLayout()
        model_layout.addLayout(row)
        self.presets = QComboBox()
        self.presets.addItems(PRESETS)
        row.addWidget(self.presets,1)
        self.load_button = QPushButton('导入 CSV')
        self.save_button = QPushButton('导出模型')
        self.add_button = QPushButton('＋层')
        self.remove_button = QPushButton('－层')
        for button,tip in [
            (self.load_button,'导入标准六列 CSV，表头带单位；不支持的模型会给出错误并保留当前模型。'),
            (self.save_button,'导出当前模型为标准 CSV，可在其他 backend 或下一次课堂中复用。'),
            (self.add_button,'在半空间上方插入一个有限层；新增示例参数可直接编辑。'),
            (self.remove_button,'删除当前选中的有限层。底部半空间必须保留。')]:
            button.setToolTip(tip)
            row.addWidget(button)
        self.table = QTableWidget(0,6)
        self.table.setHorizontalHeaderLabels(['h (m)','Vp (m/s)','Vs (m/s)','ρ (kg/m³)','Qp','Qs'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setFixedHeight(150)
        for j,key in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(j).setToolTip(TIPS[key])
        model_layout.addWidget(self.table)
        model_layout.addWidget(QLabel('末层 h = inf（半空间） · 弹性介质 Qp/Qs = inf · 悬停参数查看中文说明'))
        settings = QGroupBox('计算与显示')
        grid = QGridLayout(settings)
        panels.addWidget(settings,5)
        self.backend = QComboBox()
        self.backend.addItem('QEDispInv','qedisp')
        self.backend.addItem('CPS','cps')
        self.modes = spin(1,12,3)
        self.fmin, self.fmax = spin(.01,10000,4,3), spin(.02,10000,40,3)
        self.nf, self.show_modes = spin(2,501,73), spin(1,12,3)
        self.cmin, self.cmax = spin(1,30000,100,1), spin(2,30000,900,1)
        self.nc, self.gamma = spin(20,1001,321), spin(.00001,100,.2,5)
        fields = [
            ('曲线后端','backend'),('计算模态数','modes'),
            ('起始频率 (Hz)','fmin'),('终止频率 (Hz)','fmax'),
            ('频率点数','nf'),('显示模态数','show_modes'),
            ('速度下限 (m/s)','cmin'),('速度上限 (m/s)','cmax'),
            ('ideal 速度点数','nc'),('ideal γ (s⁻¹)','gamma')]
        for index,(label,key) in enumerate(fields):
            text = QLabel(label)
            text.setToolTip(TIPS[key])
            row,col = index//2, (index%2)*2
            grid.addWidget(text,row,col)
            grid.addWidget(getattr(self,key),row,col+1)
        paths = QHBoxLayout()
        top.addLayout(paths)
        self.cps_path, self.qed_path = QLineEdit(), QLineEdit()
        self.cps_path.setPlaceholderText('留空使用内置 CPS / 已有安装')
        self.qed_path.setPlaceholderText('留空使用内置 QEDispInv / 已有安装')
        for text,key in [('QED 文件','qed_path'),('CPS 目录','cps_path')]:
            label = QLabel(text)
            label.setToolTip(TIPS[key])
            paths.addWidget(label)
            paths.addWidget(getattr(self,key),1)
        for key in ['presets','backend','modes','fmin','fmax','nf','show_modes','cmin','cmax','nc','gamma','cps_path','qed_path']:
            getattr(self,key).setToolTip(TIPS[key])

        view_controls = QHBoxLayout()
        outer.addLayout(view_controls)
        view_controls.addWidget(QLabel('一维模型与频散曲线'))
        view_controls.addStretch()
        self.energy = QCheckBox('叠加 ideal 能量')
        self.energy.setToolTip(TIPS['energy'])
        view_controls.addWidget(self.energy)
        self.energy_note = QLabel('默认仅显示频散曲线')
        self.energy_note.setToolTip('ideal 使用 WaveDispX 全局刚度核；只在勾选时计算，与曲线后端和显示模态数独立。')
        view_controls.addWidget(self.energy_note)
        plots = QHBoxLayout()
        outer.addLayout(plots,1)
        self.figures, self.canvases = {}, {}
        model_panel = QWidget()
        model_view = QVBoxLayout(model_panel)
        model_view.setContentsMargins(0,0,0,0)
        self._add_canvas('model',model_view,(3.7,5.2))
        plots.addWidget(model_panel,3)
        self.tabs = QTabWidget()
        plots.addWidget(self.tabs,7)
        for key,label in [('rayleigh','Rayleigh（R 波）'),('love','Love（L 波）')]:
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0,0,0,0)
            self._add_canvas(key,box,(8,5.2))
            self.tabs.addTab(panel,label)
        self.stats = QLabel('修改模型后点击“计算频散曲线”。模态数含基阶 0。')
        self.stats.setWordWrap(True)
        outer.addWidget(self.stats)
        self.statusBar().showMessage('就绪 · 水平分层 / 各向同性 / 固体弹性介质')

        self.presets.currentTextChanged.connect(self.load_preset)
        self.table.itemChanged.connect(self._table_changed)
        self.backend.currentIndexChanged.connect(self.backend_changed)
        for obj in [self.fmin,self.fmax,self.nf,self.modes]:
            obj.valueChanged.connect(self.invalidate)
        for obj in [self.cmin,self.cmax,self.nc,self.gamma]:
            obj.editingFinished.connect(self.energy_settings_changed)
        self.cps_path.textChanged.connect(self.runtime_changed)
        self.qed_path.textChanged.connect(self.runtime_changed)
        self.load_button.clicked.connect(self.import_model)
        self.save_button.clicked.connect(self.save_model)
        self.add_button.clicked.connect(self.add_layer)
        self.remove_button.clicked.connect(self.remove_layer)
        self.run_button.clicked.connect(self.start)
        self.export_button.clicked.connect(self.export)
        self.energy.toggled.connect(self.toggle_energy)
        self.show_modes.valueChanged.connect(self.redraw)
        self.tabs.currentChanged.connect(self.update_stats)
        self.backend_changed()
        self.load_preset(self.presets.currentText())

    def _add_canvas(self,key,layout,size):
        fig = Figure(figsize=size)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(230,180)
        toolbar = NavigationToolbar2QT(canvas,self)
        for action in toolbar.actions():
            tips = {'Home':'恢复原始坐标范围','Back':'返回上一个视图','Forward':'前进到下一个视图',
                    'Pan':'拖动平移；右键拖动缩放坐标轴','Zoom':'框选放大图形','Subplots':'调整子图边距',
                    'Customize':'调整图形外观','Save':'保存当前图形为图片'}
            if action.text() in tips: action.setToolTip(tips[action.text()])
        layout.addWidget(toolbar)
        layout.addWidget(canvas,1)
        self.figures[key], self.canvases[key] = fig,canvas

    def backend_changed(self,*_):
        self.qed_path.setEnabled(self.backend.currentData()=='qedisp')
        self.cps_path.setEnabled(self.backend.currentData()=='cps')
        self.invalidate()

    def runtime_changed(self,*_):
        if self.sender().isEnabled(): self.invalidate()

    def set_model(self,model):
        self.table.blockSignals(True)
        self.table.setRowCount(len(model.layers))
        for i,layer in enumerate(model.layers):
            for j,key in enumerate(COLUMNS):
                item = QTableWidgetItem(f'{getattr(layer,key):g}')
                item.setToolTip(TIPS[key])
                self.table.setItem(i,j,item)
        self.table.blockSignals(False)
        self.invalidate()

    def _table_changed(self,item):
        # Updating tooltips does not invalidate or recurse into itemChanged.
        self.invalidate()

    def get_model(self):
        return model_from_rows([[self.table.item(i,j).text() if self.table.item(i,j) else ''
                                 for j in range(6)] for i in range(self.table.rowCount())])

    def load_preset(self,name):
        self.set_model(model_from_rows([(*row,np.inf,np.inf) for row in PRESETS[name]]))

    def invalidate(self,*_):
        self.result = None
        self.export_button.setEnabled(False)
        self.stats.setText('输入已更改，请计算频散曲线。模态数含基阶 0。')
        self.energy_note.setText('勾选时计算 ideal' if self.energy.isChecked() else '默认仅显示频散曲线')
        for wave in ('rayleigh','love'):
            self.figures[wave].clear()
            self.canvases[wave].draw_idle()
        try:
            draw_model(self.figures['model'],self.get_model())
            self.statusBar().showMessage('模型有效 · 等待计算')
        except Exception as exc:
            self.figures['model'].clear()
            self.statusBar().showMessage(str(exc))
        self.canvases['model'].draw_idle()

    def add_layer(self):
        i = max(0,self.table.rowCount()-1)
        self.table.blockSignals(True)
        self.table.insertRow(i)
        for j,value in enumerate(['10','750','320','1800','inf','inf']):
            item = QTableWidgetItem(value)
            item.setToolTip(TIPS[COLUMNS[j]])
            self.table.setItem(i,j,item)
        self.table.blockSignals(False)
        self.invalidate()

    def remove_layer(self):
        i = self.table.currentRow()
        if 0 <= i < self.table.rowCount()-1:
            self.table.removeRow(i)
            self.invalidate()
        else: self.statusBar().showMessage('请选择有限层；底部半空间必须保留。')

    def import_model(self):
        path,_ = QFileDialog.getOpenFileName(self,'导入标准模型 CSV','','CSV (*.csv)')
        if path:
            try: self.set_model(read_model(Path(path).read_text(encoding='utf-8-sig')))
            except Exception as exc: QMessageBox.warning(self,'模型未导入',str(exc))

    def save_model(self):
        try: text = model_csv(self.get_model())
        except Exception as exc:
            QMessageBox.warning(self,'模型无效',str(exc)); return
        path,_ = QFileDialog.getSaveFileName(self,'导出模型','model.csv','CSV (*.csv)')
        if path:
            try: Path(path).write_text(text,encoding='utf-8')
            except OSError as exc: QMessageBox.warning(self,'保存失败',str(exc))

    def velocity_axis(self):
        if self.cmax.value() <= self.cmin.value(): raise ValueError('速度上限必须大于下限。')
        return np.linspace(self.cmin.value(),self.cmax.value(),self.nc.value())

    def start(self):
        if self.busy: return
        try:
            if self.fmax.value() <= self.fmin.value(): raise ValueError('频率上限必须大于下限。')
            kwargs = dict(model=self.get_model(),frequency=np.linspace(self.fmin.value(),self.fmax.value(),self.nf.value()),
                          modes=self.modes.value(),velocity=self.velocity_axis(),gamma=self.gamma.value(),
                          backend=self.backend.currentData(),cps_bin=self.cps_path.text().strip() or None,
                          qedisp=self.qed_path.text().strip() or None)
        except Exception as exc:
            QMessageBox.warning(self,'输入无效',str(exc)); return
        self.invalidate()
        self.launch(calculate,kwargs,self.complete,'curves')

    def launch(self,function,kwargs,callback,kind):
        self.busy = True
        self.job_kind = kind
        self.editor.setEnabled(False)
        self.run_button.setEnabled(False)
        self.energy.setEnabled(False)
        self.export_button.setEnabled(False)
        self.job = Job(function,kwargs)
        self.job.signals.progress.connect(self.statusBar().showMessage)
        self.job.signals.done.connect(callback)
        self.job.signals.error.connect(self.failed)
        self.pool.start(self.job)

    def unlock(self):
        self.busy = False
        self.editor.setEnabled(True)
        self.run_button.setEnabled(True)
        self.energy.setEnabled(True)
        self.export_button.setEnabled(self.result is not None)

    def complete(self,result):
        self.result = result
        self.unlock()
        self.show_modes.blockSignals(True)
        self.show_modes.setMaximum(result['modes'])
        self.show_modes.setValue(result['modes'])
        self.show_modes.blockSignals(False)
        self.redraw()
        self.statusBar().showMessage(f"{result['backend']} 曲线完成 · {result['timings']['curves_s']:.2f} s")
        if self.energy.isChecked(): self.start_ideal()

    def toggle_energy(self,checked):
        if self.busy: return
        if checked:
            if self.result is None: self.start()
            elif self.result['response'] is None: self.start_ideal()
            else: self.redraw()
        else: self.redraw()

    def energy_settings_changed(self):
        if self.busy or self.result is None: return
        try: velocity = self.velocity_axis()
        except ValueError as exc:
            self.statusBar().showMessage(str(exc)); return
        if np.array_equal(velocity,self.result['velocity']) and self.gamma.value()==self.result['gamma']: return
        self.result = {**self.result,'velocity':velocity,'gamma':self.gamma.value(),'response':None,
                       'timings':{**self.result['timings'],'ideal_s':0.0}}
        self.redraw()
        if self.energy.isChecked(): self.start_ideal()

    def start_ideal(self):
        if self.busy or self.result is None: return
        try: velocity = self.velocity_axis()
        except ValueError as exc:
            self.failed(str(exc)); return
        self.energy_note.setText('正在计算 ideal，已有曲线保留')
        self.launch(add_ideal,dict(result=self.result,velocity=velocity,gamma=self.gamma.value()),self.ideal_complete,'ideal')

    def ideal_complete(self,result):
        self.result = result
        self.unlock()
        self.redraw()
        self.statusBar().showMessage(f"ideal 完成 · {result['timings']['ideal_s']:.2f} s · 频散曲线未重算")

    def failed(self,message):
        self.unlock()
        if self.result is not None:
            self.energy.blockSignals(True)
            self.energy.setChecked(False)
            self.energy.blockSignals(False)
            self.redraw()
        else: self.stats.setText('曲线计算失败；未替换后端或填补结果。')
        self.statusBar().showMessage(message)
        QMessageBox.warning(self,'计算失败（已计算曲线保留）',message)

    def redraw(self,*_):
        if self.result is None: return
        self.wave_stats = {}
        for wave in ('rayleigh','love'):
            self.wave_stats[wave] = draw_wave(self.figures[wave],self.result,wave,self.energy.isChecked(),self.show_modes.value())
            self.canvases[wave].draw_idle()
        self.energy_note.setText(f"逐频归一化 · γ={self.result['gamma']:g} s⁻¹（非材料 Q）"
                                if self.energy.isChecked() and self.result['response'] is not None else '仅显示频散曲线')
        self.update_stats()

    def update_stats(self,*_):
        if self.result is not None:
            wave = 'love' if self.tabs.currentIndex()==1 else 'rayleigh'
            self.stats.setText(self.wave_stats.get(wave,'')+'  ·  缺失根保留为空')

    def export(self):
        if self.result is None: return
        path,_ = QFileDialog.getSaveFileName(self,'导出结果',f"dispersion_{self.result['backend']}.zip",'ZIP (*.zip)')
        if path:
            try:
                export_result(path,self.result)
                self.statusBar().showMessage(f'已导出：{path}')
            except Exception as exc: QMessageBox.warning(self,'导出失败',str(exc))

    def closeEvent(self,event):
        if self.busy:
            self.statusBar().showMessage('正在计算，请完成后关闭。')
            event.ignore()
        else: event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = ClassroomWindow()
    window.show()
    return app.exec()


if __name__=='__main__':
    raise SystemExit(main())
