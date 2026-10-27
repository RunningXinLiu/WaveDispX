# WaveDispX Classroom：本科教学版

2026-10-27，单后端简洁界面版。PySide6 原生桌面；现有 WaveDispX 主动/被动工作台不变。

## 启动

在项目根目录执行 `./scripts/launch_classroom.sh`，或：

```sh
PYTHONPATH=python_src conda run --no-capture-output -n seisloc python -m wavedispx.classroom.app
```

复用已有 seisloc、PySide6、Matplotlib、WaveDispX 编译核心、QEDispInv forward 和原生 CPS。
本轮没有安装依赖。新机器须分别准备这些运行依赖和对应平台的原生二进制；这不是已打包的 Windows/macOS 安装包。
运行位置独立于物理参数：CPS 使用界面目录、`CPS_BIN` 或 `PATH`；QEDispInv 使用界面文件、
`QEDISPINV_FORWARD` 或项目 `.wavedispx-tools/QEDispInv/bin/forward`。
保留第三方源码、许可证和版本记录；未在本软件中复制第三方源码。

本机混装 Conda Qt 与 PySide6 wheel 时，应用为当前进程绑定 PySide6 自带插件目录，避免 QApplication 混载两套 Qt；不修改环境安装或系统 Qt。

## 标准模型 CSV v1

UTF-8，逗号分隔，固定表头和顺序，单位直接写在字段名中：

```csv
thickness_m,vp_mps,vs_mps,density_kgm3,qp,qs
10,750,320,1800,inf,inf
inf,1700,750,2150,inf,inf
```

- 从地表向下排列。有限层厚度为正；最后且仅最后一层厚度 `inf`，代表半空间。
- Vp、Vs 单位 m/s，密度 kg/m³。参数独立输入，不由 Brocher 等经验关系生成。
- 水平分层、各向同性、自由表面、固体/固体弹性介质。Vs > 0，密度 > 0，正体积模量要求 `Vp² > 4 Vs²/3`。
- `qp,qs` 预留为衰减参数；**本版必须为 inf**。有限 Q、流体层、非有限数值或缺失列均拒绝。
- 示例：`two_layer.csv`（软覆盖层）、`low_velocity_layer.csv`（低速夹层）、`halfspace.csv`（极限检查）。这些是教学案例，不是通用地层参数。

## 使用

1. 选择示例或导入 CSV，直接编辑表格；只有层模型表格可以滚动，参数区完整显示。所有参数、表头和单元格均有中文悬停说明。
2. 选择曲线 backend：QEDispInv 或 CPS。一次只运行选中的后端；未选中的后端无需安装。
3. 设置频带、实际频率点数和计算模态数。模态数包含基阶：3 表示 0、1、2 阶。点击右上角“计算频散曲线”。
4. 左侧始终显示同一张深度剖面中的 Vp、Vs、ρ：底部横轴为速度 m/s，顶部紫色横轴为密度 kg/m³，密度用紫色虚线。右侧通过 R/L 标签切换频散曲线，没有双后端叠线或差值面板。
5. ideal 默认关闭、不计算。勾选后才计算并叠加；曲线保持可见，求根不重跑，数组与坐标范围不改变。取消勾选仅隐藏；再次勾选可复用未改变的结果。若还没有曲线，先计算所选后端的曲线，再计算 ideal。
6. 修改 ideal 的速度范围/点数/γ 只使能量缓存失效，不清除曲线；勾选时重新计算 ideal。速度上下限同时控制曲线视窗，不限制后端求根。ideal 失败时保留已有曲线、允许继续导出。
7. 导出 ZIP 仅包含所选 backend 的 R/L 曲线、模型、配置、图件、数组、日志和来源信息。需要外部对比时，分别选两种 backend 计算、导出。CSV 字段为 `frequency_hz,mode,phase_velocity_mps,backend`；文件名建议 `dispersion_qedisp.zip` / `dispersion_cps.zip`。不在软件中生成两后端差值。

修改物理模型、频率轴、计算模态数或曲线 backend 会清除旧结果；显示模态数只控制 0…N−1 阶，不重算。缺失根保留 NaN，不跨截止补线。导出记录所用 backend 与二进制散列；外部比较时应注意根序号不代表已完成模态跟踪。

结果包配置 schema 升级为 `wavedispx-classroom/2`；ideal 算子仍为原 `wavedispx-classroom/1`，没有改变物理公式或显示归一化。

## 三个计算产品

1. **QEDispInv 曲线**：原项目 subprocess forward 接口，显式传入每层 h/Vp/Vs/ρ。只请求相速度，不调用 Step4、材料链式导数或反演。
2. **原生 CPS 曲线**：`sprep96 -FARR ... -NMOD ... -R -L → sdisp96 → sdpsrf96 -R/-L -ASC`。
   SI 输入显式转换为 km、km/s、g/cm³；CPS model96 的 Q=0 表示该弹性计算不使用衰减。与 disba 的 CPS 算法移植版区别标识。
   原生程序链参考：[CPS 官方手册，Modal Summation](https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/CPS/CPS330/cps330o.pdf)。
3. **ideal 频域响应图**：独立使用现有 WaveDispX 全局刚度核，在实际每个 f-c 点求
   `k=2πf/c, Ω=2πf−iγ, K(k,Ω)u=e`。
   Rayleigh 图为单位地表竖直载荷的地表 VV 响应模平方；Love 图为单位地表横向载荷的 TT 响应模平方。
   此处是单位谱域表面载荷的 compliance，不含点源逆空间变换的 `1/(2π)` 因子；该因子只影响公共幅度。
   保存复响应与未归一化 `P=|u|²`。显示为 `10log10(P/max_c P)`，逐频归一化，色标 −40…0 dB。

ideal 图不是曲线展宽/模态经验权重累加，不依赖显示的曲线阶数；含全频域响应。
γ > 0 用于取复频率响应，**不是材料 Q，也不是纯显示设置**；本版不声称严格 γ=0 极点图。
未采用有限阵列、波形窗、源子波、源半径或频率 taper；不应将此图解释为实测 FK、绝对物理能流或各模态激发强度。
R/L 各自逐频归一化，不能直接用两图亮度比较绝对强弱。峰值宽度/高度受 γ 与网格影响；加密速度轴和减小 γ 需要联合检查。
无根区域仍画出真实响应，不按理论根掩膜。均匀半空间没有离散 Love 导波，横向受迫响应仍可能非零。

## 教学练习

- 先看均匀半空间：Rayleigh 基阶近似水平，Love 无离散导波。
- 加软覆盖层：比较低频与高频相速度，增加模态数观察高阶截止。
- 加低速夹层：比较频散曲线和受迫响应亮脊，观察“曲线存在”与“响应很强”并不等价。
- 只改变层厚或 Vs，保留其他字段，比较结构对曲线的影响。

## 验证范围

本版见 `VALIDATION_V2_zh.md` 与 `qc_v2/`；首版历史记录保留在 `VALIDATION_zh.md` 和 `qc/`。已做实际二进制、解析极限、模型格式与 UI 流程检查；不宣称任意模型、多模态完备性或 ideal 全域数值收敛通过。
暂不包含群速度、反演、敏感核、有限 Q、模型拖拽、独立安装包和跨平台打包验收。
