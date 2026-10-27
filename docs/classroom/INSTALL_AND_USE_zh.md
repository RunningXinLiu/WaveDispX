# WaveDispX Classroom 安装与使用说明

**适用版本：2026-10-27 跨平台候选版。** 这是桌面软件，解压后运行；计算在本机进行，启动与计算不需要联网。包内已包含 Python 运行环境、QEDispInv、CPS、WaveDispX 计算核和中文字体。

## 1. 选对软件包

| 你的电脑 | 软件包 | 启动文件 |
|---|---|---|
| Windows，Intel/AMD 64 位 | `WaveDispX-Classroom-windows-x64.zip` | `WaveDispX-Classroom.exe` |
| Mac，Apple M 系列芯片 | `WaveDispX-Classroom-macos-arm64.tar.gz` | `WaveDispX-Classroom.app` |
| Mac，Intel 处理器 | `WaveDispX-Classroom-macos-x64.tar.gz` | `WaveDispX-Classroom.app` |
| Linux，Intel/AMD 64 位 | `WaveDispX-Classroom-linux-x64.tar.gz` | `WaveDispX-Classroom` |

Mac 可在苹果菜单的“关于本机”查看芯片/处理器。Windows ARM、32 位 Windows 和 Linux ARM 不在当前包的覆盖范围内。

已验证环境为 Windows Server 2022 x64、Ubuntu 22.04 x64、macOS 14 Apple 芯片及 macOS 15 Intel；这不等于所有旧系统或学生的 Win10/Win11 已做实机验收。建议可用屏幕区域至少 1200×780。

从 [GitHub Release 公开下载页](https://github.com/RunningXinLiu/WaveDispX/releases/tag/classroom-2026.10.27-rc1) 的 **Assets** 选择上表中的软件包，不需要 GitHub 账号。不要把 `Source code` 当成桌面程序。教师直接转发的软件包也可使用；下载后按下面步骤解压。

Actions 链接用于构建记录，下载可能需要 GitHub 登录且有保存期限；学生优先使用 Release。若拿到的是 Actions 构建资料，进入 `release/`，再解压其中的软件包。

## 2. 安装和启动

### Windows

1. 对 ZIP 选择“全部解压”，放在自己可写的目录，例如“文档”中的 `WaveDispX` 文件夹。不要在压缩包预览窗口内直接运行。
2. 进入解压后的 `WaveDispX-Classroom` 文件夹，双击 `WaveDispX-Classroom.exe`。
3. 保留旁边的 `_internal` 文件夹；创建桌面快捷方式可以，但不能只复制一个 exe。
4. 不需要安装 Python、Conda、QED、CPS、Fortran、MSYS2 或 WSL，也不需要填写后端路径。

当前候选版未签名，Windows 可能显示来源提示。先核对文件来自教师且校验值一致；若学校电脑策略禁止运行，请由教师或管理员处理。不要为运行软件关闭系统防护。

### Mac

1. 选对 Apple 芯片或 Intel 包，双击 `.tar.gz` 解压。
2. 将 `WaveDispX-Classroom.app` 放到“应用程序”或自己的应用目录，再双击启动；保留整个 `.app`。
3. 当前候选版未做 Apple 公证。如果只提示“无法验证开发者”，确认来自教师后，按系统提供的“系统设置 → 隐私与安全性 → 仍要打开”处理。若提示已损坏或包含恶意软件，先停止并联系教师，不按普通来源提示处理。参见 [Apple 官方说明](https://support.apple.com/en-gb/102445)。

无需在 Mac 上安装 Homebrew、Python 或 gfortran，也无需输入终端解除系统保护的命令。

### Linux

在有桌面的 Linux 系统中解压并运行。终端示例（在软件包所在目录执行）：

```sh
tar -xzf WaveDispX-Classroom-linux-x64.tar.gz
cd WaveDispX-Classroom
./WaveDispX-Classroom
```

如果提示“没有执行权限”，可只给主程序增加自己的执行权限：

```sh
chmod u+x WaveDispX-Classroom
./WaveDispX-Classroom
```

保留整个文件夹。本包在 Ubuntu 22.04 验证，仍依赖操作系统正常的桌面/图形基础库；无桌面的 SSH 终端、旧 glibc 系统或其他发行版可能不能直接运行。出现 `GLIBC_... not found` 时应使用匹配系统的包，不要替换系统 libc。

## 3. 第一次计算：五分钟流程

1. 在左上角模型下拉框选择 **“两层：软覆盖层”**。第一次可以保持示例参数。
2. 在“曲线后端”选择 **QEDispInv** 或 **CPS**。下面的“QED 文件”和“CPS 目录”均留空。
3. 使用起始频率 **4 Hz**、终止频率 **40 Hz**、频率点数 **73**；“计算模态数”填 **3**，表示 **M0、M1、M2**，包含基阶。
4. 点击右上角 **“计算频散曲线”**。左图显示 Vp、Vs 和密度随深度的变化；右图用 **Rayleigh（R 波）/ Love（L 波）** 标签切换。
5. 默认只计算和显示曲线。需要时勾选 **“叠加 ideal 能量”**，等待计算后再观察背景能量图。频散曲线保持不变。
6. 点击 **“导出结果 ZIP”**，保存到自己的作业目录，例如 `two_layer_qedisp.zip`。

所有输入参数、表头和相关按钮都带中文悬停说明。表格内可以直接修改层参数；修改模型、频带、计算模态数或 backend 后需要重新计算。

## 4. 模型怎么输入、导入和保存

模型按地表到地下的顺序排列；“＋层”在底部半空间上方插入有限层，“－层”删除选中的有限层。末层半空间必须保留。点击“导出模型”可以保存当前模型；点击“导入 CSV”可读取外部模型。

标准文件采用 **UTF-8、英文逗号、固定六列表头**：

```csv
thickness_m,vp_mps,vs_mps,density_kgm3,qp,qs
10,750,320,1800,inf,inf
inf,1700,750,2150,inf,inf
```

| 列名 | 含义及单位 | 填写规则 |
|---|---|---|
| `thickness_m` | 层厚 h，m | 有限层 > 0；最后一层为 `inf` |
| `vp_mps` | 纵波速度 Vp，m/s | 正数，且 Vp² > 4Vs²/3 |
| `vs_mps` | 横波速度 Vs，m/s | 正数；本版不支持流体层 |
| `density_kgm3` | 密度 ρ，kg/m³ | 正数，例如 1800，不是 1.8 |
| `qp` | 纵波品质因子，无量纲 | 本版填写 `inf` |
| `qs` | 横波品质因子，无量纲 | 本版填写 `inf` |

`inf` 表示无穷大。本版限定固体、各向同性、无耗散弹性层状介质，所以 Qp/Qs 只能为 `inf`。Vp、Vs、密度独立输入，改一个不会自动修改其他参数。

随说明提供的 `examples/` 有三个模型：

- `two_layer.csv`：软覆盖层，用于第一次计算。
- `low_velocity_layer.csv`：低速夹层，用于观察高阶模态和频率变化。
- `halfspace.csv`：均匀半空间；没有离散 Love 导波是预期行为。

用 Excel 编辑时保存为“CSV UTF-8”，保留列名、列顺序及 `inf`；不要把中文逗号、空单元格或单位文字写进数值列。

## 5. 图和参数怎么看

| 参数/图件 | 作用 |
|---|---|
| 频率点数 | 实际求解的等间距频率点数；73 点、4–40 Hz 对应步长 0.5 Hz |
| 计算模态数 | 请求前 N 个模态，包含 M0；高阶在某些频率下可能无根 |
| 显示模态数 | 从已经计算的模态中显示前 N 阶，不重新求根 |
| 速度上下限 | 曲线显示范围，也是 ideal 的扫描范围；不会限定后端求根 |
| ideal 速度点数 | 能量图的速度采样精度；不改变曲线。默认 321，单次 f–c 点总数最多 200000 |
| ideal γ | 复频率偏移，单位 s⁻¹，默认 0.2；必须 > 0。它影响响应峰宽和幅值，不是材料 Q |
| 一维模型图 | Vp/Vs 读下方速度轴；紫色密度虚线读上方 kg/m³ 轴；共用深度轴 |

ideal 使用 WaveDispX 的频域单位表面载荷响应，不是把若干条曲线简单涂宽，也不是实测波形的 FK 能量。图上按每个频率归一化显示 −40 至 0 dB；R/L 两图亮度不能直接用于比较绝对能量。减少“显示模态数”不会截断 ideal 响应。

取消 ideal 勾选只隐藏背景；模型和能量参数没变时，再勾选可复用计算结果。调小 γ 往往需要更细速度采样；先使用默认值完成课堂实验。

## 6. 保存结果和比较两个 backend

每次仅计算所选 backend。若要比较，保持模型、频带、模态数一致，分别计算并导出 `case_qedisp.zip` 与 `case_cps.zip`，再在软件外比较 CSV。

结果 ZIP 中包括：

| 文件 | 用途 |
|---|---|
| `model.csv` | 本次计算的标准模型，可再次导入 |
| `rayleigh_curves.csv` / `love_curves.csv` | 频率、模态、相速度和 backend；单位 Hz、m/s |
| `*_curves.png`、`model.png` | 曲线及模型图片，可用于作业 |
| `*_ideal.png` | 已计算 ideal 时的叠加图片 |
| `arrays.npz` | 数值数组；已计算 ideal 时含复响应和原始功率 |
| `config.json`、`manifest.json`、`solver_logs.json` | 参数、来源、版本/散列及可用日志，用于复核 |

某些频率的高阶无根时保留 NaN，不应在外部作图时跨缺根区连线。两个后端的高阶截止附近结果可能不同；同名模态序号也不等于已完成跨后端分支跟踪。

**导出是否包含 ideal 取决于当前结果中是否已有能量数组。** 取消勾选仅隐藏背景，不会删除已计算的 ideal 数据；若只要纯曲线结果，取消勾选后点击“计算频散曲线”，再导出。

## 7. 常见问题

| 现象 | 处理方法 |
|---|---|
| 提示找不到 QED/CPS | 确认完整解压、没有单独移动 exe、两个路径框为空。曾配置过 `CPS_BIN` / `QEDISPINV_FORWARD` 的用户应检查这些覆盖设置 |
| 没有曲线/曲线不完整 | 查看状态栏和有效根数量；增大显示速度范围；高阶无根不等于程序失败；可先用两层示例和 M0 检查 |
| 均匀半空间 Love 图为空 | 无离散 Love 导波是预期结果；勾选 ideal 仍可能看到横向受迫响应 |
| CSV 导入失败 | 核对六列表头、英文逗号、单位、最后一层 `inf`、Qp/Qs=`inf` 和正数要求 |
| ideal 比曲线慢 | 频率数×速度点数决定扫描规模；先减小 ideal 速度点数，保持模型不变 |
| 参数修改后图消失 | 模型、频带、计算模态数、backend 改变会使旧曲线失效，点击计算即可 |
| 中文变方框或应用打不开 | 使用教师发的最终包，重新完整解压；不要删除内置字体/运行库。将系统版本和错误信息交给教师 |
| 界面显示不全 | 尝试最大化窗口，检查显示缩放后的可用区域；记录屏幕分辨率和缩放比例给教师 |

卸载时关闭程序、删除解压目录或 `.app` 即可。自己导出的模型和作业文件另行保存。更新时解压到新目录，先确认能运行，再处理旧目录；不要混合两个版本的运行库。

## 8. 给教师的交付说明

每位学生需要 **一个匹配平台的软件包 + 本使用说明 + examples/**。说明资料 ZIP 很小，里面不包含大型软件包；软件包需另发。四种平台包和 SHA256 校验清单都可从 [Release 下载页](https://github.com/RunningXinLiu/WaveDispX/releases/tag/classroom-2026.10.27-rc1) 获取，也可由教师离线转发。

当前为未签名公开试用版。第三方许可/源码说明仍有待核对项，详见仓库的 `packaging/classroom/THIRD_PARTY_zh.md`；测试边界见 `docs/classroom/PACKAGING_VALIDATION_20261027_zh.md`。建议先在少量学生机器上试用，再扩展到全班；不要把云端构建通过解释为所有历史操作系统均兼容。

问题反馈请附：系统版本与芯片类型、软件包文件名、所选 backend、报错原文/截图、可复现的模型 CSV 和计算参数。不要发送密码或账户凭据。
