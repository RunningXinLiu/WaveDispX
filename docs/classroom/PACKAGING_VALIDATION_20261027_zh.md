# 教学版跨平台候选包：2026-10-27

> 发布状态更新（2026-10-27）：本仓库公开，四平台原候选二进制按公开试用版提供于 [Release](https://github.com/RunningXinLiu/WaveDispX/releases/tag/classroom-2026.10.27-rc1)，未重编译。下文“私有”指构建与验收当时的状态；Actions 链接作为历史记录保留。第三方许可待核对项见 [组件说明](../../packaging/classroom/THIRD_PARTY_zh.md)。

四个平台均通过发行包实际运行检查与截图复核。此项是移植/包装验收，不扩大频散科学验证的适用范围。

| 平台 | 实测环境 | 对各自 v2 基准的最大曲线差（m/s） | 私有下载入口 |
|---|---|---:|---|
| windows-x64 | Windows-2022Server-10.0.20348-SP0 | 0 | [下载](https://github.com/RunningXinLiu/WaveDispX/actions/runs/40496166313/artifacts/12730339626) |
| linux-x64 | Linux-6.8.0-1064-azure-x86_64-with-glibc2.35 | 0 | [下载](https://github.com/RunningXinLiu/WaveDispX/actions/runs/40495293225/artifacts/12729748663) |
| macos-arm64 | macOS-14.8.9-arm64-arm-64bit | 2.16e-05 | [下载](https://github.com/RunningXinLiu/WaveDispX/actions/runs/40495293225/artifacts/12729518785) |
| macos-x64 | macOS-15.7.9-x86_64-i386-64bit | 0 | [下载](https://github.com/RunningXinLiu/WaveDispX/actions/runs/40495293225/artifacts/12729654913) |

## 给教师与学生的启动说明

教师登录 GitHub 下载对应 artifact；展开后，`release/` 内才是需要发给学生的应用压缩包。候选包按系统和 CPU 架构分发。Actions 文件保留 14 天，服务器保存验证后的长期副本。

- Windows x64：解压内层 ZIP，双击 `WaveDispX-Classroom.exe`。保留旁边的 `_internal` 文件夹；不要单独拷贝 exe。
- Mac：按 Apple 芯片 / Intel 选择包，解压内层 tar.gz，打开 `WaveDispX-Classroom.app`。
- Linux x64：解压内层 tar.gz，运行 `WaveDispX-Classroom/WaveDispX-Classroom`。本轮为 Ubuntu 22.04 基线，需要正常桌面/图形基础库，中文字体已内置。
- 无需安装 Python、Conda、QED、CPS、Fortran 或 WSL。两个后端路径保持空白即使用内置程序。
- 推荐可用屏幕区域至少 1200×780。较旧系统、其他 CPU 架构和不同缩放比例需要另做实际验证。

选择示例模型或导入六列 CSV，选择一个 backend、频带和模态数，点击计算。默认只画曲线；勾选 ideal 后再计算能量叠加。R/L 在标签间切换，Vp/Vs/ρ 与曲线并列。当前模型可直接导出 CSV；计算结果用 ZIP 导出。后端对比在软件外用各自导出结果完成。

## 检查内容与证据

- 原生 QED forward、CPS sprep96/sdisp96/sdpsrf96 在各平台从固定源码构建；没有用其他求解器替换它们。
- PyInstaller 成品移动到带中文、空格的新目录，清除开发 PATH/PYTHONPATH 与后端路径覆盖，运行内置 `--self-test`。
- 固定四层低速夹层模型；频率 4–40 Hz、73 点、M0–M2；ideal 速度 100–900 m/s、41 点、gamma=0.2 s⁻¹。41 点是对既有 321 点基准每隔 8 列取样。
- 每个 backend 的缺根位置必须与自己的基准相同；速度 atol=0.002 m/s、rtol=2e-6；复响应 atol=1e-18、rtol=1e-7。这里没有要求 QED/CPS 相互完全相等；已知 Love 截止附近差异仍保留。
- 实际执行 GUI 计算与 ideal 勾选，断言叠加前后曲线数组不变，导出模型/曲线/图片/数组，并记录求解器二进制 SHA256。
- 固定版本 Noto Sans SC 字体与 OFL 许可随包携带；检查中文/拉丁/数字/ρ 字形覆盖，查看真实截图。Windows 使用原生 windows Qt 平台；Mac/Linux 的截图使用 offscreen。
- 本机源码回归：15 passed；学校 Linux 原生重新编译后的两套后端 R/L 基准差均为 0。公共 Python 导出逐项核对保持兼容。

详细平台记录、二进制/构建来源、压缩包散列、到期日期见 `packaging_qc/RELEASE_MANIFEST.json`；各平台 `smoke-output/self_test.json`、`visual_review.json` 与 PNG 是直接证据。早期 Windows 无屏幕插件的文字方框问题已记录为失败候选，不能凭它的数值自检 pass 宣称界面通过。

## 发布范围

当前是私有、未签名候选包，没有改主分支或创建公开 Release。尚未完成 Windows 商业签名、Apple 公证及学生 Win10/Win11 实机点击验收。不同历史系统版本不能仅凭云端构建推断兼容。

正式分发前需补齐对应版本的第三方许可与源码提供说明：QED 根 LICENSE 与部分源码头表述不一致，旧 CPS 的对应授权来源也需核对。已保留原始声明与源码快照，不用新版网页许可替代旧源码记录。详情见 `../../packaging/classroom/README_zh.md`。

较大压缩包和源码归档放在学校服务器；本机仅保留源码、基准数组、小型 JSON 和截图。服务器目录：`lx_lzf:/storage/HOME/yaolab/liuxin/wavedispx_classroom_build_20261027/releases/`。
