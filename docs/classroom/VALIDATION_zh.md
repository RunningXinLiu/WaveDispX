# 教学版验收记录 · 2026-10-27

> 历史阶段记录：后续跨平台成品检查见 [打包验证记录](PACKAGING_VALIDATION_20261027_zh.md)，不扩大本页的科学验证范围。

状态：**指定案例功能与解析检查通过；存在已展示的根覆盖分歧，未作通用数值认证。**

## 影响范围

新增 `python_src/wavedispx/classroom/`、独立启动/界面检查脚本、专用测试与示例。
没有修改既有 WaveDispX 正演核、QEDispInv/vendor、CPS 或原工作台入口。
新 operator 标识 `wavedispx-classroom/1`；定义见 README 和结果包 config.json。
材料直接输入 h/Vp/Vs/ρ，仅相速度正演，无 Step4 总导数或材料经验关系适配。

## 执行和证据

环境：本机 macOS，既有 Conda seisloc；精确二进制路径/散列见 `qc/validation.json` 和结果包 manifest。
本地测试只涉及小型曲线与 f-c 响应，共约数 MB 检查产物；未生成波形或大型实验数据。

```sh
PYTHONPATH=python_src conda run --no-capture-output -n seisloc pytest -q \
  python_tests/test_classroom.py python_tests/test_qedisp.py --tb=short
PYTHONPATH=python_src conda run --no-capture-output -n seisloc python scripts/check_classroom_ui.py
```

测试：**11 passed in 7.56s**，完整日志 `qc/tests.txt`。教学版 7 项加原 QEDisp 4 项相关回归。

| 检查 | 标准/目的 | 实际结果与边界 |
|---|---|---|
| CSV 往返/非法输入 | 数值往返相等；有限 Q、流体、负体积模量、非标准表头、非半空间末层拒绝 | 通过 |
| SH 半空间 compliance | 与 `1/(ρ Vs² sqrt(k²−Ω²/Vs²))` 的模比较，相对误差 ≤1e−11 | 通过；仅解析半空间 |
| Rayleigh 半空间根 | 两后端与解析 Rayleigh 多项式根差 ≤0.002 m/s | 通过；3 频率；Love 输出无离散根 |
| ideal Rayleigh 半空间峰 | f=10 Hz、γ=.01、围绕解析根 ±3 m/s、dc=.01 m/s；峰位差 ≤.02 m/s | 通过；仅此峰位检查 |
| 两层 Love 特征方程 | 原生 CPS 基阶代入精确两层 SH 方程；无量纲残差 ≤2e−5 | 通过；4–40 Hz 的37频率 |
| 两层/四层原生后端 | 三阶、37频率，同有限根位置差 ≤.02 m/s；两后端基阶完整 | 通过；不保证高阶完备/跨分支身份 |
| ZIP 可复算导出 | 模型往返；保存原始功率与复响应模平方严格相等；逐文件 SHA256 验证 | 通过 |
| UI 状态 | 改输入清除结果并禁止导出；显示开关保留结果 | Qt 控件测试通过 |
| UI 完整运行 | 实例化真实 Cocoa 窗口，以按钮信号启动后台任务，切换 R/L 页并导出 | 通过；截图 `qc/desktop_{rayleigh,love}.png` 已查看；不是人工点击全部对话框验收 |

## 展示案例

`low_velocity_layer.csv`：10/10/20 m 三有限层+半空间；参数可直接查 CSV。
f=4–40 Hz、df=.5 Hz（73点）；c=100–900 m/s、dc=2.5 m/s（321点）；γ=.2 s⁻¹；曲线0–2阶。

- Rayleigh 有效根：QEDispInv/CPS 均为 73、73、69；共同根最大差 0.0000608 m/s。
- Love 有效根：QEDispInv 73、73、61；CPS 73、72、61；共同根最大差 0.0000598 m/s。
- **保留的分歧：4 Hz、Love 第1阶，QEDispInv 返回 749.996 m/s，CPS 无根。** 接近半空间 Vs=750 m/s，尚未独立判定是截止附近遗漏、伪根或其他寻根差异；不自动填补、不计入共同根最大差，也不把两者称为完整一致。
- 时长见 `qc/timing.json`，是本机小例的程序链耗时，不是公平跨求解器速度基准。
- `qc/low_velocity_layer.zip` 保存完整配置、原始复响应、曲线、无能量图/带能量图、日志与 manifest。截图不替代数组证据。

## 未完成或不适用

γ 和 f/c 网格的全带收敛、任意层数/反差/高阶完整性、模态身份跟踪、有限 Q、群速度、有限阵列/波形成像、独立 Windows/Linux 打包尚未验收。
ideal 是单位载荷表面 compliance 模平方的逐频归一化展示，不能将 γ=.2 或本例网格当通用数值设置。
半空间解析与指定根位置检查不验证全域响应幅值；未进行三求解器波场对照，也不继承历史 SASEM/SEM 证书。

## 环境故障与修复证据

最初 QApplication 在窗口创建时退出，日志显示同时加载 wheel 的 Qt.framework 与 Conda libQt6 dylib。
`QT_DEBUG_PLUGINS` 确认使用了 Conda `lib/qt6/plugins/platforms`。独立 QApplication 也复现，排除数值程序/新窗口逻辑。
仅设置 QCoreApplication libraryPaths 未消除冲突；绑定 PySide6 wheel 的 `QT_PLUGIN_PATH` 与 `QT_QPA_PLATFORM_PLUGIN_PATH` 后，独立启动和完整真实窗口均通过。
修复封装在 `qt_runtime.py`，只针对 macOS 存在 wheel Cocoa 插件的场景、只改变当前进程环境；未删除、重装或修改第三方环境文件。
