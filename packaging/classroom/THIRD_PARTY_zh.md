# 教学版第三方组件与源码说明

本页记录 2026-10-27 候选版实际使用的组件。仓库可公开查看，但不将所有组件重新授权为 MIT 或其他统一许可证。原始版权、许可文本及源码头保留。

| 组件 | 用途 | 本仓库记录 |
|---|---|---|
| QEDispInv | R/L 相速度曲线，独立 forward 程序 | `vendor/qed/`、根 LICENSE、源码头及 `vendor/SOURCE_MANIFEST.json` |
| CPS | `sprep96` / `sdisp96` / `sdpsrf96` 原生程序链 | `vendor/cps/`，保留原始源码声明及逐文件散列 |
| fmt、toml11、HighFive、CLI11 等 | QED 的构建依赖 | `vendor/qed/external/` 和 `vendor/qed/include/` 中的原始声明 |
| Python、PySide6 / Qt、Matplotlib、NumPy 等 | 桌面运行环境与绘图 | 固定构建依赖见 `requirements.txt`，实际环境见各平台构建清单；发行包保留所收集的组件元数据及许可文件 |
| Noto Sans SC | 中文界面及图件字体 | `fetch_font.py` 固定上游提交并验证文件身份；字体与 OFL 文本随包携带 |
| 编译器运行库、HDF5 等 | 原生二进制运行依赖 | 各平台原生构建及动态库收集，见工作流、构建清单和包内文件 |

## 固定版本与对应源码

`vendor/SOURCE_MANIFEST.json` 的 `source_sha256` 给出所用 QED/CPS 快照的逐文件 SHA256。构建配方为 `native/CMakeLists.txt`，Windows 平台适配为 `native/windows_headers.hpp`；平台适配不修改求解算法。WaveDispX 核和课堂应用的源代码分别位于 `cpp/` 和 `python_src/wavedispx/`。

公开试用版的源代码可从同名 Release 的 `Source code` 获取，或检出标签 `classroom-2026.10.27-rc1`。该标签在已验收候选源码上补充中文发布文档，不重新编译已有二进制。二进制实际构建提交、平台和散列以[打包验证记录](../../docs/classroom/PACKAGING_VALIDATION_20261027_zh.md)及包内构建清单为准。外部依赖仍需按构建工作流准备，源码 ZIP 不是完整离线工具链。

## 尚未统一的许可记录

- QED 快照的根 LICENSE 为 LGPL-3.0 文本，部分源码头使用 GPL 表述；两者原样保留，不擅自选择其中一种覆盖另一种。
- CPS 来自既有旧版源码快照，包含原始署名与声明，但本快照缺少完整的版本授权配套记录。当前官方网页的新版本许可不能直接作为旧快照的授权证明。
- 本仓库暂无统一顶层许可证，也未完成整个冻结运行环境的逐组件许可汇总。公开试用版不是“所有第三方授权审计已完成”的声明；进一步再分发、修改或商业使用应逐项核对适用条款。

上游来源：[QEDispInv](https://github.com/pan3rock/QEDispInv)、[CPS 官方网站](https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/index.html)、[Google Fonts Noto Sans SC](https://github.com/google/fonts/tree/main/ofl/notosanssc)。
