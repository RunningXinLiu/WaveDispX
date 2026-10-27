# WaveDispX：面波正演与交互式频散教学软件

**修改层状速度模型，计算 Rayleigh / Love 频散曲线，观察地下结构如何改变面波传播。**

WaveDispX 包含面向本科课堂的 **Classroom 桌面版**，以及面向进一步研究的主动源、被动源波场与频散分析工作台。Classroom 使用 PySide6 图形界面，可选择 **QEDispInv 或原生 CPS** 计算相速度频散曲线，按需叠加 WaveDispX 的 ideal 频域响应图。

学生使用打包版即可，**无需安装 Python、Conda、编译器、QED 或 CPS**。计算在本机完成，下载后可离线使用。

**[下载桌面软件](https://github.com/RunningXinLiu/WaveDispX/releases/tag/classroom-2026.10.27-rc1)** · **[中文安装与使用说明](docs/classroom/INSTALL_AND_USE_zh.md)** · **[教学版详细说明](docs/classroom/README_zh.md)** · **[反馈问题](https://github.com/RunningXinLiu/WaveDispX/issues)**

![Windows 教学版实测界面：左侧为 Vp、Vs、密度剖面，右侧为频散曲线](docs/classroom/packaging_qc/windows-x64/smoke-output/compact_curves.png)

## 下载与安装

在 [Release 下载页](https://github.com/RunningXinLiu/WaveDispX/releases/tag/classroom-2026.10.27-rc1) 的 **Assets** 中选择对应软件包。公开下载不需要 GitHub 账号；`Source code` 是源码，不是可直接启动的软件。

| 电脑类型 | 下载文件 | 启动方式 |
|---|---|---|
| Windows，Intel / AMD 64 位 | `WaveDispX-Classroom-windows-x64.zip` | 全部解压，双击 `WaveDispX-Classroom.exe` |
| Mac，Apple M 系列芯片 | `WaveDispX-Classroom-macos-arm64.tar.gz` | 解压，打开 `WaveDispX-Classroom.app` |
| Mac，Intel 处理器 | `WaveDispX-Classroom-macos-x64.tar.gz` | 解压，打开 `WaveDispX-Classroom.app` |
| Linux，Intel / AMD 64 位 | `WaveDispX-Classroom-linux-x64.tar.gz` | 解压，运行 `WaveDispX-Classroom` |

请保留完整解压目录，尤其是 Windows / Linux 的 `_internal` 文件夹。界面中的“QED 文件”“CPS 目录”留空即可使用内置后端。另外下载 `WaveDispX-Classroom-中文使用说明与模型示例.zip`，即可获得离线 HTML 说明和三个模型示例。

当前版本是 **未签名的教学试用版（Pre-release）**，尚未做 Apple 公证。已在 Windows Server 2022、Ubuntu 22.04、macOS 14 Apple 芯片和 macOS 15 Intel 构建环境运行成品自检；不等于所有学生电脑、Win10/Win11 和旧系统均已验收。Windows ARM、32 位系统和 Linux ARM 暂无对应包。系统提示及具体安装步骤见[安装说明](docs/classroom/INSTALL_AND_USE_zh.md)。

## 五分钟开始第一次计算

1. 打开软件，选择“两层：软覆盖层”示例，或导入自己的 CSV 模型。
2. 选择一个曲线后端：**QEDispInv** 或 **CPS**。
3. 设置频率范围和模态数。例如 4–40 Hz、73 个频率点、3 个模态，表示计算 M0、M1、M2。
4. 点击“计算频散曲线”，通过 **Rayleigh / Love** 标签切换波型。
5. 如需观察背景响应，勾选“叠加 ideal 能量”。默认关闭，勾选后才计算，不改变已有曲线结果。
6. 点击“导出结果 ZIP”，保存模型、曲线 CSV、图件、配置和计算记录。

参数和表头提供中文悬停说明。层模型可直接编辑、增删有限层、导入和导出；固定参数区无需滚动。左图在同一深度剖面中显示 Vp、Vs、密度，速度与密度分别使用底部和顶部横轴；右图显示当前波型的频散曲线。

一次只计算所选后端。需要比较 QED 与 CPS 时，分别计算并导出，在外部对比结果。

## 模型格式

使用 UTF-8 CSV，按地表到地下排列，表头固定为六列，单位写在列名中：

```csv
thickness_m,vp_mps,vs_mps,density_kgm3,qp,qs
10,750,320,1800,inf,inf
inf,1700,750,2150,inf,inf
```

| 字段 | 含义 | 规则 |
|---|---|---|
| `thickness_m` | 层厚，m | 有限层 > 0；最后一层为 `inf`，表示半空间 |
| `vp_mps` | 纵波速度，m/s | 正数，且 Vp² > 4Vs²/3 |
| `vs_mps` | 横波速度，m/s | > 0，本版仅支持固体层 |
| `density_kgm3` | 密度，kg/m³ | > 0，例如 1800，而不是 1.8 |
| `qp`, `qs` | 纵波、横波品质因子 | 当前弹性教学版必须为 `inf` |

Vp、Vs、密度独立输入，不自动套用经验关系。示例模型：[软覆盖层](docs/classroom/two_layer.csv)、[低速夹层](docs/classroom/low_velocity_layer.csv)、[均匀半空间](docs/classroom/halfspace.csv)。

## ideal 图表示什么

ideal 图由 WaveDispX 层状介质刚度核独立计算：Rayleigh 使用单位地表竖直载荷的 VV 响应，Love 使用单位地表横向载荷的 TT 响应。显示响应模平方的逐频归一化 dB 图，并保存原始复响应及未归一化数组。

它是**频域受迫响应图**，不是将理论曲线展宽得到的示意图，也不是有限阵列实测波形的 FK 图。复频率参数 γ 会影响响应峰宽，不能当作材料 Q。显示模态数不控制 ideal 图的累加阶数；两种波型分别归一化，不能直接比较亮度来判断绝对能量大小。公式、归一化和适用边界见[教学版详细说明](docs/classroom/README_zh.md#三个计算产品)。

## 导出与课堂练习

结果 ZIP 包含所选后端的 R/L 曲线 CSV、模型、图件、配置、日志及来源记录；计算过 ideal 时还包含相应响应数组和图件。曲线 CSV 字段为 `frequency_hz,mode,phase_velocity_mps,backend`。缺失根保留缺失值，不跨截止频率补线。

可以从以下练习开始：改变覆盖层厚度观察频散变化；保持 Vp 和密度不变，只修改 Vs；增加计算模态数观察高阶截止；导入低速夹层，比较理论曲线与响应亮脊；用均匀半空间检查 Rayleigh 与 Love 的区别。

## 文档与开发

| 内容 | 入口 |
|---|---|
| 学生安装、操作、模型导入、导出和故障排查 | [安装与使用说明](docs/classroom/INSTALL_AND_USE_zh.md) |
| 教学版物理定义与界面行为 | [Classroom 说明](docs/classroom/README_zh.md) |
| 四平台成品验证、截图和测试边界 | [打包验证记录](docs/classroom/PACKAGING_VALIDATION_20261027_zh.md) |
| 原生后端编译与桌面打包 | [构建说明](packaging/classroom/README_zh.md) |
| 主动 / 被动波场、DAS 与研究工作台 | [Python/C++ 说明](README_PYTHON_CN.md) |
| 原始 MATLAB 程序与原作者说明 | [原始 Readme](Readme.txt) |
| 科学工作流与证据登记 | [正演工作流](docs/forward_workflow/WORKFLOW_zh.md) |

开发者可克隆本仓库，在独立 Python 环境中按构建说明准备依赖和编译核心，再启动 `python -m wavedispx.classroom.app`。打包依赖见 `packaging/classroom/requirements.txt`，原生构建配方见 `packaging/classroom/native/`，跨平台自动构建见 `.github/workflows/classroom-build.yml`。源码环境需要另外准备运行依赖；学生下载打包版即可。

Classroom 当前支持水平分层、各向同性、固体/固体、无耗散弹性介质的**相速度**计算。暂不提供群速度、反演、敏感核、有限 Q、流体层或各向异性计算。工作台的功能范围与课堂版不同，不能混用两者的支持范围或验证结论。

## 来源与许可说明

WaveDispX 延续本仓库原始 PassiveFW / MATLAB 程序的工作，原作者署名和参考文献保留于 [Readme.txt](Readme.txt)。教学曲线后端使用 [QEDispInv](https://github.com/pan3rock/QEDispInv) 和 Robert B. Herrmann 的 [Computer Programs in Seismology（CPS）](https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/index.html)。研究使用时请同时遵循所用组件的引用要求。

仓库公开不代表全部代码采用同一种许可证。第三方原始声明、固定源码和 SHA256 清单位于 `packaging/classroom/vendor/`；本仓库尚未设置统一顶层许可证。QED 根 LICENSE 与部分源码头存在不同表述，旧 CPS 快照的版本授权说明仍待补齐，不能直接套用当前上游网页的许可证。具体来源、现有声明与待核对项见[第三方组件说明](packaging/classroom/THIRD_PARTY_zh.md)。
