# WaveDispX DAS 本地证据登记

## STUDIO-CUSTOM-007：CPU选项与非均匀源输入包（2026-10-28）

Studio新增workers、频率/源分块及中文hover和整数范围校验；新增活动表CSV导入/移除，校验成功才更新输入。提供单侧源线阵、双簇源不规则面阵、逐窗强度切换三个完整TOML与坐标/活动CSV及中文字段说明。未修改正演/互相关/成像核、自动参数或物理输入定义；未开放逐源方向或新的事件目录。

相关回归73 passed in 33.46s，保留1条隐藏画布布局警告；workers=1/2小型主动矢量源波形与R/L能量在rtol=1e-12、atol=1e-14内一致。真实Qt参数/预览截图已查看。服务器CPU完成三例12源60s实际混合记录NCF和R/L能量，小图与报告回传，数组留服务器。短记录、小孔径和非均匀照明的宽峰/波动保留，不声称收敛、加速或新的物理验收。报告SHA256本机/服务器一致：288f3e2526766a8dc38513bfe636357884c8d59c02c0241019ffc9d2332616cb。配置/CSV/代码/数组哈希、限制、服务器路径与可查看QC见`docs/studio/custom_inputs_20261028/README_zh.md`、`results/report.json`及`ui_final/UI_VALIDATION.json`。未推送发布。

## STUDIO-WINDOWED-006：普通窗口、源配置保留和R/L实例（2026-10-28）

默认改普通窗口（本机1600×922逻辑像素），不最大化/全屏。新增示例与说明入口、中文源分量/活动表/DAS等错误定位。修复源环/源线/位置CSV清空极化、功率、方向和activity_file；保留用户源数，CSV按行数更新。67项相关回归通过；真实Qt按钮计算主动R/L、无Love原因、仅理论、被动R/L，未启用HVSR；绘图前后能量哈希不变。Qt自动化QTest.qWait导致的后台Python饥饿/90s超时用QEventLoop解决，未改生产线程或内核。保留警告/失败调试记录。

学校CPU补算24源60s与192源600s independent_hv、192源600s directed_hv，均得到R/L有限记录NCF成像；不代表收敛或独立核验证。输入、原生平滑/窗、算子与谱权重未为图形效果更改。模型/PSD/geometry/双边NCF及有无覆盖QC已查看；大数组留服务器，小图与report逐文件SHA256核对回传。运行配置/代码/二进制/数组哈希、截图、测试及范围见`docs/studio/windowed_20261028/README_zh.md`、`ui_verified_final/UI_PRODUCTS.json`、`passive_results/report.json`。源模板SHA256=`bb191f1108df7a35b4a7bf6cb15cbe979423a0f02311433a029a48abbdc18551`；服务器运行脚本SHA256=`3cbec803effcf7884bb0c2d4c4757615ecc740e95c1449833f5cb86295877a04`。未推送/发布。

## STUDIO-HVSR-VIEW-005：多台站HVSR只读展示（2026-10-28）

同日入口调整：HVSR从默认计算目标移至「附加分析」菜单，默认不计算、不展示其紧凑参数和结果页；显式HVSR配置仍保留，关闭后恢复此前主目标，不修改源分量。仅UI修改，无新正演，无输入/物理/谱比算法变更。`test_studio_ux.py test_studio.py test_hvsr_view.py`：19 passed in 20.09s；`git diff --check`通过。源码SHA256：`desktop/studio.py`=`7c13c8f3d8a15eb7d36da789cf9beb1fc21aa732f28f99a7b3c9bcb6827295ef`，`python_tests/test_studio_ux.py`=`c07c80210ec3b8c5867e7537ed5289e1b0a654691a9430df4393ae6b9287126f`。之前可查看的QC图保留为多站显示层证据，不作为新菜单截图；旧运行窗口未重启。

仅显示层修改：单站H/V主曲线、选择1–12站叠加、全台站共同线性色标热图、Geometry接收器点选/修饰键多选、热图行点选。线阵按沿线距离，面阵按输入编号；不平均台站、不逐站归一化，NaN保留灰色。原分量谱/方向诊断单列；没有新增Z/H、NCF谱比或理论椭圆率。物理、HVSR处理和原生数值内核未改。

复用STUDIO-PRODUCTS-004服务器定向噪声例的全部41站统计，仅下载351783-byte显示快照，未重新正演或拉取全波形。Qt实际接收器鼠标点击站10成功，显示前后全部HVSR数组哈希一致。初次检查修复密集台站难点选、颜色重复与底部说明重叠，保留初版截图；最后人工检查完整窗口与曲线/热图。未更改用户旧窗口、未推送GitHub。数据/输入来源、最终代码与快照哈希、鼠标记录和可查看QC见`docs/studio/hvsr_multistation_20261028/README_zh.md`及`screenshots_final/VALIDATION.json`。不继承或增加噪声收敛认证。

## STUDIO-PRODUCTS-004：显式计算目标、矢量源与节点HVSR（2026-10-28）

新增Studio产品层：仅QED理论、所选R/L观测能量、原始XYZ节点HVSR及组合。R/L标签保留未选择/不激发/缺分量/零响应/失败说明。主动`vector`按全局方位/倾角相干投影五项原生Green基响应；被动新增固定总源力PSD的independent_hv/directed_hv模式，旧legacy_xyz系数原样保留。DAS仍限制垂直源，未开放Love-DAS或DAS-HVSR；不更改原生Green、源平滑/频率窗、成像内核或自动参数默认。

HVSR按全时长流式原始混合XYZ独立窗计算，10%Tukey、二次均值H、可选KO振幅谱平滑、几何平均及窗间对数散度；不足10周期/频点、垂直谱下限和少于3窗标无效。不是NCF、不是Rayleigh椭圆率，不宣称完整HVSR质量认证。新加权噪声仅独立平稳高斯源、所有源同一水平力方位；逐源方向与带方向事件目录未实现。新产品不静默继承旧CLI/分阶段工作台，旧入口对非兼容请求明确拒绝；只理论不依赖有效观测系统参数。

本机相关11个测试文件最终结果与实际源码散列见`docs/studio/products_v1_20261028/VALIDATION.json`。此前完整回归116 passed in 47.60s，补充状态/导出后重点22 passed in 13.27s；最终重跑另记，不重复累计。主动quick兼容能量SHA256仍为`e44cc12c2ca058c852e203765ac6f56889d992179365334ca398787723b871d3`。受控恒等测试覆盖矢量线性/旋转、源PSD权重、已知H/V=2、零V掩码及精确零横向投影，非独立物理验收。

学校CPU实际横向/倾斜主动、两种被动极化四例与独立保存成功。横向线阵无R/Z响应、倾斜例R/L均可成像；两种被动HVSR首窗与NCF生产器原始float32预览相对L2约2.54e-8。24源60 s仅为集成检查，不继承此前600/1200 s或其他模型的收敛结论。源/模型/实际轴、统计和原数组保留服务器`wavedispx_classroom_build_20261027/studio_products_20261028/examples/results_final/`，本地仅报告、manifest、输入和QC。

真实Qt后台按钮完成矢量源、纯垂直无Love、仅Love理论、逐站HVSR、首窗XYZ切站检查；1728×1084逻辑全屏。修复Retina上下文图密度副轴未完整重绘并重新检查，不隐藏布局警告。完整窗口、公式、输入示例、保存格式、阶段源码/二进制哈希与限制见`docs/studio/products_v1_20261028/README_zh.md`、`ui_final/`、`ui_render_fix/`、`server_final/`。未安装依赖、未重打发行包、未推送GitHub，未修改已有用户运行窗口。

## STUDIO-UX-003：视觉层级与操作状态改版（2026-10-28）

仅修改Studio显示/交互及截图脚本：文件/输入/计算/结果操作分区，主流程编号与主次按钮颜色，参数语义分组，当前模式标签，CSV菜单，DAS依赖字段灰显，收起参数专注结果，文本+颜色状态及运行锁定。旧canvas/toolbar在deleteLater前立即隐藏，防止新输入搭配旧图；保存成功不再自动弹出旧工作台结果页。科学输入、核、成像色标及数组不改，自动参数和数值收敛未获新认证。

本机`test_studio_ux/test_studio/test_studio_acceptance/test_desktop`：最终39 passed in 37.31s。实际Qt后台CPU active_quick运行，QED开关前后能量SHA256均为`e44cc12c2ca058c852e203765ac6f56889d992179365334ca398787723b871d3`，与此前同例一致。测试还验证收起参数不改输入/数组、CSV派发、状态锁定、旧canvas立即隐藏。被动DAS面阵仅预览，无新增大型计算。

本机16-inch逻辑1728×1084完整窗口已查看，主动/被动输入、成像设置、修改状态、运行中、波形和收起参数截图及代码/样式/二进制/输入散列见`docs/studio/ux_v2_20261028/`与`UI_VALIDATION.json`。保留原截图及数值诊断；一次隐藏画布布局警告仍在日志中，未屏蔽，已检查可见图无缺轴裁切。依然为真实Qt自动化而非声称computer-use鼠标遍历。未增加依赖或推送发布。

## STUDIO-ACCEPTANCE-002：操作回归与CPU数值敏感性（2026-10-28）

修复Studio浮点参数输入、被动Love页、运行日志重复打开/运行中可见、保存前误启动正演及TOML另存相对路径问题；新增实际轴/时长摘要及导入状态保护。中文hover明确区分正演截断与成像上限、周期变化与DWM网格/复频移。正演/成像/互相关/DAS核及自动参数算法不变。本机88 passed in 35.46s，完整Qt窗口截图已查看。

学校CPU现有环境完成主动/被动×line/array×节点/DAS共8模板及5个附加案例的计算/保存；smoke-pass不等于科学准确性。361道被动DAS面阵301.58s，交互性能尚未验收。Mac/Linux同算法active_quick图像相对L2=6.07e-13、ZZ波形=2.56e-8，分别小于预定1e-6/1e-5；非独立物理验证。

固定192源/种子的600→1200s被动对照：图像相对L2=4.885%、NCF=10.148%；预定5%仅图像目标满足，仍未验收源分布/独立种子收敛。Lg4/8/16m对照保留异常尾部，不能将它当作定量标距认证。Lg16m原16Hz截断下T4→8s图像变化11.051%；同4s数据df/dv减半的共同点变化0.0113%，不认证插值点间弱峰。

进一步Ricker f0=7Hz、同模型/阵列/dt，显式对照截断16/24/32Hz：T4s近道末20%能量占比49.78%/0.00909%/0.00196%；T8s占比79.75%/0.02858%/0.000137%。支持正演截断不足是明显晚时振铃的重要原因；未单独分离零截断/指数校正等机制。提高截止后周期图像差仍约3.2%，因此未认证收敛、未复制为全局默认、未隐藏尾部。全部实际数组与原生平滑/频率窗保留。

证据、限制、配置/代码/二进制散列、可见QC及命令见`docs/studio/acceptance_20261028/README_zh.md`、`report.json`、`PROVENANCE.json`及子目录报告。大结果留`lx_lzf:/storage/HOME/yaolab/liuxin/wavedispx_classroom_build_20261027/studio_acceptance_20261028/`；初始脚本错误及绘图恢复经过也保留。未安装依赖、未使用GPU、未发布或推送工作区其他改动。状态：UI可试用；自动参数和数值收敛未完成。

## STUDIO-UI-001：CPU/QED 全屏实验界面（2026-10-28）

新增独立 `wavedispx.desktop.studio` 入口，保留旧工作台和课堂版。变更限界面组织、输入映射、QED 理论后端选择及显示；未修改正演、噪声/互相关、DAS 或成像内核与自动参数算法。复用当前本地适配器，不自动更新远端版本，也不继承其他 QED 项目的稳定性认证。

本机 `test_studio/test_desktop/test_qedisp/test_plotting_axes`：21 passed in 15.06s。真实 active_quick 点击后台 CPU 工作线程，QED 叠加/关闭前后 221×164 能量数组 SHA256 都为 `e44cc12c2ca058c852e203765ac6f56889d992179365334ca398787723b871d3`。覆盖被动字段和 DAS 121 道面阵审核/标距修改，不宣称已新验收被动长记录或 DAS 多标距收敛。

完整窗口 1728×1084 逻辑像素、Retina 2×，模型/Geometry/能量及可选炮集截图已查看。输入、实际 CPU/QED 二进制/适配器/源码哈希与限制见 `docs/studio/UI_VALIDATION.json`、`docs/studio/checked_input.toml` 和 `docs/studio/README_zh.md`。图保持 quick 算例真实旁瓣；不是正式收敛展示。原生截图过程有一次 constrained-layout 警告，已核对可见图无缺轴/裁切；computer-use 未能绑定命令行 Python，因此验证途径为实际 Qt 脚本和截图。未生成大型本地波形文件，未推送或发布发行包。

## PLOT-AXES-003：双边NCF显示（2026-09-17）

仅显示修改：被动横轴−T…+T，纵轴距离从近（下）到远（上）；主动保持offset横轴、0在上且向下增长的时间轴。汇总图及网页/桌面/Notebook共享阶段绘图统一使用此约定。
使用baseline_linear的实际negative/positive分支，反转负支并去掉重复零点后拼接，不镜像symmetric分支；距离排序只用于显示。共用全图99%绝对幅值分位裁色，不逐道归一化。频散仍使用原有symmetric数据，未改变计算。

通过 `scripts/replot_passive_summary.py` 从保存的HDF5重绘手册3600秒案例，无重新正演；重绘前后 results.h5 SHA256均为 `4e9a58fc04ce93dd45b2e559f2e21fa75c0a5edc20437bc8c3b267b5d3cc5379`。
QC `docs/manual_assets/showcase-passive-summary-ncf.png` 已查看；汇总图同步更新。
回归 `test_plotting_axes.py test_workflow.py test_runner.py test_das.py`：**37 passed in 22.27s**，验证实际异幅正负支、唯一零点、距离排序、主动时间轴方向和运行保存链路。
绘图源码SHA256：plotting.py `68d285d10cfcd0c5c3c3b5561da5474b44c98b366ffa80bc80f574d307f775cf`；workflow_plotting.py `9b3edb831631df930055ed6574d6bf4f7c8c8520ae46f206a7db327f01a16595`。

## AUTO-SHOWCASE-002：推荐策略修订与手册实算（2026-09-17）

状态：指定两层算例完成实算与图像检查；**不是完全统计收敛/跨求解器验收**。
用户指出手册60秒短记录不足以展示，自动范围过窄、时长建议不全。
修改仅涉及建议策略、UI默认入口、预检显示诊断与文档，不改正演核、NCF或成像算子。

- 频带保留源功率≥−20 dB及空间采样支持；不以全模态两波长孔径或DAS首零点裁掉频带，相关物理限制保留预检警告。
- 速度边界不再依赖旧dv；dv上限按最慢保留理论相速度的孔径峰宽/8计算，工程取整不得超过候选上限。仍需dv/2独立收敛测试，不能视为严格峰宽界。
- 点击自动按钮才改被动总时长：至少60窗且600秒，整窗取整、保留更长输入；有activity_file不自动延长总时长。此为工程候选而非误差门槛。源数量/位置/模型不变。
- 新会话默认载入384源/1800秒长记录模板，不覆盖已有会话。quick仍可选择。NCF显示FFT网格按保留正lag样点数计算，并另列1/window与1/T_ncf。

实算脚本 `scripts/run_manual_showcase.py`，本机seisloc/4 workers；原始结果
`output_python/manual_showcase_20260917/{active_auto,passive_60,passive_600,passive_1800,passive_3600}`。
两层模型Vp/Vs=380/220、760/440 m/s、h=10 m、rho=2000 kg/m³；主动Ricker20 Hz，1–200 m/1 m，both分量，Hankel；被动384竖直源800–1200 m环带、−300–300 m/5 m、Ricker PSD12 Hz、20秒独立窗、固定空间/时间种子、谱白化NCF/phase shift。
被动仅总时长变化，保留所有频率与速度坐标；无平滑或理论掩膜。相对3600秒，归一化全图L2差为60秒69.013996%、600秒17.930565%、1800秒7.072603%。3600秒只作较长参考，**未设通过阈值，不据此宣布完全收敛**。主动低频自动拾取仍偏离理论；高阶弱信号、源数量和数值收敛尚未在本轮全面检查。

QC图 `docs/manual_assets/showcase-{active,passive}-summary.png` 和
`showcase-duration-comparison.png` 已逐图查看，手册保留旧短记录反例。
逐例配置、输入/结果SHA256及指标见 `docs/manual_assets/showcase-report.json`。
全量回归：`PYTHONPATH=python_src conda run -n seisloc pytest -q --tb=short`，**254 passed in 62.36s**。
本轮computer use两次超时，未宣称网页点击复验；仅截图渲染检查及本地HTML链接检查。

| 文件 | SHA256 |
|---|---|
| survey_design.py | 9815e941166d862bed7e29dce17ce0501b2d3c5baf69fbb17760cf67f95c7b5e |
| app_service.py | 3cefa9ea680663f181cf378225110ea5e7a90cb665f3bc51b0b531e17bdfda5e |
| preflight.py | b86b09d81bdbbaba6dfede9f30907e6fcb3d0ea69ef73aeb656e8240c0f1c836 |
| scripts/run_manual_showcase.py | d2e0e16e7e6d62c500a31c7cfe6552afadae0d93f120986b923a7e43a3f3b1d9 |
| docs/manual_assets/showcase-report.json | f04b9dafa4a47e46a87490f269ccc60536f061d3252eaaca9faf8f17802edcb8 |

更新时间：2026-09-17。此文件是本项目对便携工作流的本地补充，不是跨模型或跨求解器的通用验证证书。

## 状态定义

- `analytic-pass`：解析算子或人工可判定极限已通过自动测试。
- `smoke-pass`：完整输入→正演→记录→相关/成像→落盘链路可运行，输出已人工查看。
- `not-validated`：接口可能存在，但还没有足以支持科研结论的对照证据。

## 当前证据

| ID | 状态 | 输入/测试 | 可复核证据 | 结论边界 |
|---|---|---|---|---|
| DAS-OP-001 | analytic-pass | `python_tests/test_das.py` | 刚体平移为零、均匀轴向应变、单频应变率 `+iω`、第一标距零点 | 只验证有限标距测量算子和 FFT 号约定，不验证 DWM Green 本身 |
| DAS-ACTIVE-LINE-001 | smoke-pass | `examples/active_das_line_quick.toml` | `output_python/active_das_line_quick/results.h5`、`summary.png`、`manifest.json` | 16 道、8 m 标距、竖直点力、应变率、phase-shift；不是宽频/长排列精度证书 |
| DAS-PASSIVE-LINE-001 | smoke-pass | `examples/passive_das_line_quick.toml` | `output_python/passive_das_line_quick/results.h5`、`summary.png`、`manifest.json` | 41 道、48 源、60 s/6 窗；只证明真实有限记录随机混合和内部 DAS–DAS NCF 链路 |
| DAS-PASSIVE-GRID-001 | smoke-pass | `examples/passive_das_grid_quick.toml` | `output_python/passive_das_grid_quick/results.h5`、`summary.png`、`manifest.json` | 19×19=361 个平行直标距通道；二维接收布局不等于二维横向变化介质或一根转弯光缆 |
| DAS-ACTIVE-GRID-001 | smoke-pass | `examples/active_das_grid_quick.toml` | `output_python/active_das_grid_quick/results.h5`、`summary.png`、`manifest.json`、`das_channels.csv` | 11×11=121 个平行有限标距通道；验证规则二维通道布局的主动链路，不代表二维横向非均匀介质 |
| DAS-ACTIVE-CABLE-001 | smoke-pass | `examples/active_das_channel_file_quick.toml`、`examples/das_channels_bent_example.csv` | `output_python/active_das_channel_file_quick/results.h5`、`summary.png`、`manifest.json`、标准化 `das_channels.csv` | 15 道折线光缆、逐道 6–8 m 标距及 0°/45°/90°方向；验证逐端点主动正演和文件往返，不代表连续光纤力学 |
| DAS-PASSIVE-CABLE-001 | smoke-pass | `examples/passive_das_channel_file_quick.toml`、同一折线光缆 CSV | `output_python/passive_das_channel_file_quick/results.h5`、`summary.png`、`manifest.json`、标准化 `das_channels.csv` | 15 道、48 源、60 s/6 窗；验证逐通道 DAS 传递→随机混合记录→DAS–DAS NCF→phase-shift，短阵列结果较宽且不作收敛声明 |
| DAS-NB-001 | smoke-pass | `notebooks/04_das_active_passive.ipynb` | 已自顶向下执行；HTML 为 `output_python/notebook_preview/04_das_active_passive.html`；真实浏览器全页截图为 `output/playwright/das_notebook_channel_file_full.png` | 参数 Cell 可切换 active/passive 与 line/grid/channel_file；本次保存输出仍是默认被动线阵快速案例，Notebook 不替代独立求解器验证 |

本次未提交快照的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `python_src/wavedispx/das.py` | `2f47e7fa54f98d4b947cadd4fec8657a41d5734c672e19409e38d62e29dd5d38` |
| `examples/active_das_line_quick.toml` | `0e1e0ce99312a365dd7a1705df4b5c3569883c669859b06bc7ca5824bc2c37d9` |
| `examples/passive_das_line_quick.toml` | `d28b55c863990f65514428c502aa7eb6ca7992c79db6ef02fc0691ba28658d7f` |
| `examples/passive_das_grid_quick.toml` | `d60fb4abf26a60f8c67a56c0e31d2746b5dcca0fa1f2865f94ac4ec0a6ac9189` |
| `output_python/active_das_line_quick/results.h5` | `0c4a59c1f6693952da4ce6b84779a3a872dc35bb2a347834991058c886246c77` |
| `output_python/passive_das_line_quick/results.h5` | `d87ebdc3ba62d6261701cd2b8f4bdc342530c7852ca22017b944767988d849d7f` |
| `output_python/passive_das_grid_quick/results.h5` | `119f22471550cf12772da781171858239019032b53c90e60fe51e9dc9bc7196e` |
| `notebooks/04_das_active_passive.ipynb` | `fcbef0036744441ce690dd3e43b32fb13ecb442cc821c659ef6eb175629ba02d` |
| `examples/active_das_grid_quick.toml` | `0b6292e212c5c4cc0b2f2d9322d8320081a9f4d4a5df953a9b4045e3cd1f0105` |
| `examples/das_channels_bent_example.csv` | `f4a122b06a0c2ddfa736a8e808ac4eec56c2256c00a6a7eeca4b3f064e1f660b` |
| `examples/active_das_channel_file_quick.toml` | `5d2917ab0bbafd8dcfc2a45c85cf6dd03fba11f28e6a514820dfe82ab1836c28` |
| `examples/passive_das_channel_file_quick.toml` | `8e9c2ed5d8b02d60ceada24a1be80ac0b8e1ba97b6ae4939ce21b5847598fed2` |
| `output_python/active_das_grid_quick/results.h5` | `f21149be292f2c69f7fe34ea27e22970eff2a52bb94665a727cd7dc17573aeb2` |
| `output_python/active_das_channel_file_quick/results.h5` | `fa89f6b1e00a6d41cf9bd70ed2fcfdfb364894b2ca897182c893db1fa676ce38` |
| `output_python/passive_das_channel_file_quick/results.h5` | `8e15aedee0a6dd8417ebed93fc7ec15861a67539225bebb64fb6376cf615ea84` |
| `output_python/notebook_preview/04_das_active_passive.html` | `c7dbd3efb572afa9cb7cf56123b6dad02022573264747d1f58e138f5410f238c` |
| `output/playwright/das_notebook_channel_file_full.png` | `75f554025e80559202d85eaf41cb7816d7a5fd772a9c271214da6cda190446da` |
| `docs/WaveDispX_UI_MANUAL_CN.html` | `0aaace29b73b752e902c2714f80adc9c5d5e514b9ccb2a1a1c67ae3d93f2ca14` |
| `docs/manual_assets/das-active-grid-review-full.png` | `5960272ee5afc36cc37831eebf8421b42e5b71daeb473f01ea87d8a977ee6800` |
| `docs/manual_assets/das-passive-cable-review-full.png` | `dbc8277273d95ed5d810178a18dd0cf151c53489021712ef6b5dfa18dc63449e` |
| `output/playwright/wavedispx_manual_das_section.png` | `7c493e7ca7f77d250aeed45f8dfe1ebd48be2c476bcc227823e6d527b00145d3` |

本次完整回归命令 `PYTHONPATH=python_src conda run -n seisloc pytest -q` 的结果为
`242 passed in 45.40s`；这是上表 DAS 快照当时的结果。浏览器渲染只出现缺失 `favicon.ico` 的 404，不影响 Notebook 内容。

## 时长与 f-v 范围审计（2026-09-17）

本轮把 DWM synthetic period、保存炮集/NCF 窗的 FFT period 和被动总统计时长拆开记录，并增加
基于源谱、多模态相/群速度、孔径、径向采样、DAS 标距与真实 df 的模型感知预检查。建议范围不
静默覆盖显式输入；主动显式记录过短或被动窗长不足属于 required failure。

| ID | 状态 | 输入/测试 | 可复核证据 | 结论边界 |
|---|---|---|---|---|
| RANGE-ACTIVE-001 | smoke-pass | `examples/active_two_layer.toml` | `output_python/active_two_layer_range_validation/results.h5`、`summary.png`、`manifest.json` | DWM 复频率核→有限炮集→Hankel/FJ 图；不是未经波形窗的实频率直接有限阵列产品 |
| RANGE-PASSIVE-001 | smoke-pass | `examples/passive_two_layer.toml` | `output_python/passive_two_layer_range_validation/results.h5`、`summary.png`、`manifest.json` | 384 源、1800 s、90 个 20 s 独立窗；验证当前两层模型的连续基阶能量，不是三求解器证书 |
| RANGE-PREFLIGHT-001 | analytic-pass | `python_tests/test_preflight.py`、`test_processing.py`、`test_active.py` | `output_python/active_quick_range_audit/preflight.png`、`passive_two_layer_range_audit/preflight.png` | 验证范围/时长计算与阻塞逻辑；经验阈值（-20 dB、2 波长、20 频率单元、速度边界系数）仍需按研究目标声明 |

当前 HDF5 根属性使用 `dispersion_product` 区分
`windowed_waveform_image` 与 `finite_record_ncf_waveform_image`，并分别保存 observation/NCF df、
DWM df 和复频移位。直接实频率有限阵列产品尚未实现，不能用当前结果代替。

当前未提交快照 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `python_src/wavedispx/survey_design.py` | `6dbaae6940576f66ff3e7559bd0e4d2c850668d8b77ba9cf3d4809d65cecff34` |
| `python_src/wavedispx/preflight.py` | `39b6332d9606fe9de2a560a27237a7e2d339056b34b1781623dc044f07950a75` |
| `python_src/wavedispx/active.py` | `b339617d0c818fadbb74653a7037368d8275801db54fa784aa0128468a486ec8` |
| `python_src/wavedispx/passive.py` | `b0d7fbf9e46fbd92152d1842cc2bda69c64a8ee7084ab20155ce269c3fb94135` |
| `python_src/wavedispx/artifacts.py` | `581e3dd83a4bdcdbd20ddd3039c7150e93898f98c1c1791458316c363c1d5bc8` |
| `examples/active_two_layer.toml` | `5589e9ed9e79282c7fb1f248e761243074dfb0142460d142eee8fb3cdb0c2d0e` |
| `examples/passive_two_layer.toml` | `ae3a912d77171833d66b51eb810547b3facb78af2be0b332d3d1bfe7df9c3406` |
| `output_python/active_two_layer_range_validation/results.h5` | `bef11d058eab0c909f104dabb00fd607eec0e8f72d68f47ff59d68d4eb6ca058` |
| `output_python/passive_two_layer_range_validation/results.h5` | `faff131489425e56551974f321784f757460dd24b3ac2e83e20791f3ef7361ab` |
| `output_python/active_two_layer_range_validation/summary.png` | `9a7be4f56eeb63dade0eaf377a18ab49b927ecdb558a893202d058a124741811` |
| `output_python/passive_two_layer_range_validation/summary.png` | `331ab4e9b2c2183e7cfcb235945dfbb5bde8cacdc638328071707ef7c043599f` |

当前完整回归命令 `PYTHONPATH=python_src conda run -n seisloc pytest -q` 为
`246 passed in 44.53s`；`python -m compileall -q python_src/wavedispx python_src/nnws scripts` 通过。

## 明确未验证

### UI 自动参数验收（2026-09-17）

`UI-AUTO-001`：smoke-pass，范围仅为当前 quick 主动/被动两层算例的参数交互及完整运行。
通过 computer use 在网页界面点击自动推荐、手改 df/记录时长/lag、恢复模板默认，运行并保存两种模式。
主动记录 FFT 周期 3.5 s（最后样点 3.495 s，dt=0.005 s）、显示 df=0.05 Hz；
被动 60 s / 10 s 独立窗，手改 lag=2.5 s、显示 df=0.1 Hz，均核对 HDF5 实际轴。
QC：`output_python/ui_acceptance_active_20260917/summary.png`、
`output_python/ui_acceptance_passive_20260917/summary.png`。
桌面真实渲染截图 `output_python/ui_acceptance_desktop.png` 已人工查看，工具栏改为两行；
computer use 未能识别本机 Conda Python 桌面进程，原生交互由 Qt 自动测试覆盖，不能称作原生人工点击验收。

本次修复：被动保存变量引用；套用新频带后重算推荐时长；频带/模态变化使依赖时长的正演缓存失效。
新增显式 df 为补零 FFT 复谱插值网格，不能解释为增加物理分辨率。lag 修改影响相关阶段；
当前短记录测试不证明噪声收敛或多求解器一致性。
回归：`test_config.py test_processing.py test_app_service.py test_preflight.py test_desktop.py test_workflow.py test_runner.py`
共 **95 passed in 25.52s**（seisloc，`PYTHONPATH=python_src`）。

| 快照 | SHA-256 |
|---|---|
| `python_src/wavedispx/app_service.py` | `c193d56e4e17cdbcbf11c32d887d4dfe53607cbb10e9591ab1c8c1fdef684646` |
| `python_src/wavedispx/workflow.py` | `d02e6f815900f34874140b187abf69a6faddd29fd23419d573c9f9d281c5f0b5` |
| `python_src/wavedispx/processing/dispersion.py` | `66a6baa2ea7c96d63ed106ecd828be1d193f5cd39942c373f0abbe2e8818b862` |
| `python_src/wavedispx/desktop/app.py` | `03f2e0bd0c7432b5e35c7fc7c39d2fd6c5b965da7280fea848266979fc19b945` |
| `output_python/ui_acceptance_active_20260917/input.toml` | `01057a0e4800de3d96c27389d7adc1e79a246ea9b4dc0a38738896dfdebe4ff5` |
| `output_python/ui_acceptance_active_20260917/results.h5` | `c6788bb9c062ed6cb93698a50d922732d432410330dce3767fa196c5ed91ff02` |
| `output_python/ui_acceptance_passive_20260917/input.toml` | `50b67f52d0a41b2e5bbbc4f3fda1bc68b6d0d536d9abe79a946e09d8a73665fa` |
| `output_python/ui_acceptance_passive_20260917/results.h5` | `c8fb2da4ecbf4b93dc1d349ff733409e05db4b33d52343acc59b0cdd082190b9` |

## 尚未覆盖的科学验证

收尾补充：2026-09-17 全量 `pytest -q --tb=short` 为 **251 passed in 42.90s**。
当前 HTML 操作手册所有本地链接与图片存在性检查通过，浏览器检查结果页完整截图正常。
详见 `docs/RELEASE_20260917_CN.md`。

- 与同物理量 DAS–DAS 主动参考的 NCF/EGF 等价关系；当前不以节点 `ZZ` Green 冒充 DAS 参考。
- 三分量随机力 DAS、FJ/Bessel/Hankel DAS 核、连续弯曲光纤力学、井中 DAS、地形或横向非均匀模型。当前支持的是逐通道离散测量段及逐道方位/标距。
- 60 s 快速被动案例的统计收敛；需要固定源场与随机种子，增加记录时长/独立窗数并保存相邻时长增量指标。
- SPECFEM2D/SASEM/WaveDispX 同模型、同源、同观测量的 ISO 三求解器对照。

## 可复现命令

必须强制使用本工作区源码，因为本机 `seisloc` 的 editable 安装可能指向另一份 WaveDispX 工作区：

```bash
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/active_das_line_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/passive_das_line_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/passive_das_grid_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/active_das_grid_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/active_das_channel_file_quick.toml
PYTHONPATH=python_src conda run -n seisloc wavedispx run examples/passive_das_channel_file_quick.toml
PYTHONPATH=python_src conda run -n seisloc pytest -q
```

每次作为正式证据引用前，应同时登记 Git commit（若已提交）、输入文件 SHA-256、`results.h5` SHA-256、运行平台和完整测试结果；未提交的工作树不能只用分支名标识快照。

## CLASSROOM-001：PySide6 弹性频散教学入口（2026-10-27）

新增独立 `wavedispx.classroom`，不修改旧正演/观测流程。模型为各向同性固体水平层+半空间、Qp/Qs=inf；显式独立 h/Vp/Vs/rho。QEDispInv forward 与原生 CPS 相速度曲线对比；可叠加新命名 `wavedispx-classroom/1` 的 WaveDispX 理想谱域单位表面载荷 compliance 模平方，不是有限阵列 FK 或曲线展宽。

指定范围的 CSV/解析极限/双后端/导出/Qt 状态及原 QEDisp 回归共 **11 passed**；真实 Cocoa 窗口后台计算、R/L 页渲染与 ZIP 导出完成并检查截图。四层低速夹层显示例 f4–40/.5 Hz、c100–900/2.5 m/s、gamma=.2 s^-1，0–2阶共同根最大差 Rayleigh 6.08e-5、Love 5.98e-5 m/s；**4 Hz Love M1 QEDispInv 有根、CPS 无根，保留待查，不能宣称完整一致**。

精确配置、源码/二进制哈希、原数组、QC 与限制见 `docs/classroom/VALIDATION_zh.md`、`docs/classroom/qc/validation.json`、`docs/classroom/qc/low_velocity_layer.zip`。只为教学诊断，不继承三求解器波场或全频域幅值收敛结论。无大型本地实验。Qt 冲突通过当前进程插件路径隔离修复，没有改第三方安装。

## CLASSROOM-002：单后端与按需 ideal（2026-10-27）

按用户新要求，教学界面改为可选 QEDispInv/CPS 单后端，导出独立结果；删除界面双后端曲线/差值。ideal 默认关闭，独立任务计算，保留曲线；失败也不清除曲线。参数区无滚动，模型层表可滚动；Vp/Vs/密度共享深度图，密度独立顶部轴；R/L 两标签。所有参数提供中文 tooltip。

曲线和 ideal 科学内核未改。与 CLASSROOM-001 同例原数组核对，两后端 R/L 曲线及 ideal 复响应/原始功率逐元素相等；真实 UI 的 ideal 开关前后曲线 SHA256 相同。结果包 schema /2 与原 ideal operator /1 分开记录。证据、源码/二进制/结果散列、测试和真实截图见 `docs/classroom/VALIDATION_V2_zh.md`、`docs/classroom/qc_v2/`，不覆盖旧 qc/。原 Love 截止附近根覆盖分歧未修订，未扩大科学验收结论。

## CLASSROOM-PACKAGE-001：四平台候选包（2026-10-27）

范围：PyInstaller 封装、原生 QED/CPS 构建、Windows 路径/运行库/头文件兼容、中文字体与成品移植回归。上游数值源码、输入 h/Vp/Vs/rho、无耗散材料与 ideal 算子未改；包入口不运行反演或材料链式导数。公共 Python 导出改为按需导入并逐项核对兼容，避免引入无关工作流依赖。

Windows x64（原生 Qt windows 平台）、Linux x64、Mac arm64、Mac x64 的实际成品均通过：清理开发环境路径、中文/空格位置迁移、所选后端 R/L、按需 ideal、数组保留、导出与字体字形检查。截图已直接查看。Windows 早期 offscreen 文字方框候选被判为失败，未用于最终交付。

基准为 CLASSROOM-002 四层模型，73 个频率、3 模态；ideal 使用原 321 列速度轴的每第 8 列（41 点）。缺根位置须一致；速度 atol=0.002 m/s、rtol=2e-6；复响应 atol=1e-18、rtol=1e-7。保留 QED/CPS 各自根覆盖差异；不等于跨后端或全模型科学认证。本机相关回归 **15 passed in 15.45s**，学校 Linux 新编译原生两套后端 R/L 与各自基准差均为 0。

构建来源：Mac/Linux `c3b94358e9564faa4bbd8befcce1382361d43a2c`；Windows `fd870bf77d83ffbb947c965948d16cbf1e67b30c`（仅测试平台选择进一步修正）。完整输入/代码/二进制/包散列、平台报告、PNG、下载和服务器位置见 `docs/classroom/PACKAGING_VALIDATION_20261027_zh.md` 与 `docs/classroom/packaging_qc/RELEASE_MANIFEST.json`。当前私有未签名候选包，未覆盖所有旧系统、学生实机或正式再分发许可整理；主分支未更改。

## CLASSROOM-DEMO-001：近邻模态差异教学模型（2026-10-27）

用户请求可展示 QED/CPS 差异的模型。探索性筛选后提供同一四层结构的两个厚度版本：h=10/20/10/inf m（R），或10/20/20/inf m（L）；Vp=700/1600/500/2200 m/s，Vs=300/800/180/1100 m/s，rho=1800/2100/1750/2300 kg/m³，Qp/Qs=inf。R 用25–40 Hz/301点/8模态；L用30–40 Hz/201点/12模态。只算频散曲线，不含ideal、阵列或波形；不调用材料经验关系、敏感核或反演。

当前原生CPS配置对R的26.95–34.5 Hz和L的31.05–35.5 Hz存在根覆盖/编号分歧。R在34.5 Hz的435.2602/435.4382 m/s、L在35.5 Hz的382.0633/382.0922 m/s近邻根未被当前CPS返回；独立动态刚度/SH层传递分别确认这四个局部根。数百m/s同编号差值主要是漏根后的序号错位，不能称为匹配物理模态的速度误差。不声称QED全域根完备，也未测试其他CPS设置能否修复。

Linux原生程序筛选、本机实际应用适配层和已公开Mac ARM包内QED/CPS复算完成；最终两套R/L数组与本机开发程序逐元素一致（含NaN）。本例未重新验收Windows/Intel Mac GUI。源代码及求解器未修改。模型、实际轴、原数组、标准导出ZIP、可见对比图、独立根检查、代码/二进制/文件SHA256见 `docs/classroom/backend_demo_20261027/README_zh.md` 与 `PROVENANCE.json`；完整筛选日志在学校服务器同名项目的 `backend_demo_20261027/`。本地只保留小型检查和教学交付文件。

## CLASSROOM-DEMO-002：恢复用户指定的 DAzi 基阶失败图（2026-10-27）

用户指出所需案例来自 DAzi 测试并提供原图，已在“查找可公开使用的 dazi 包”chat 及原报告确认 d8_s1_c30。恢复原20节点、float32 Vs、91周期及冻结QED/CPS相速度。原DAzi嵌入surfdisp96连续周期调用有37零值；4.389401751s的正值差为1.463865832 km/s（QED2.853847428、CPS4.317713260）。不把原图整段偏离未经分支核验就定为某个高阶。

按冻结QED材料与中点细分/展平规则构造显式77层Rayleigh平面模型；当前教学版同源码原生QED forward与sdisp96在同一91频率均有效，最大差0.000097 m/s。新平面结果不冒充原球形输出的逐位复现；这证实本任务必须区分两个CPS接口。当前GUI最多40层、线性频率轴且无DAzi节点球形预处理，不能声称导入CSV即可复现原图。未修改应用、vendor、冻结数据或重打包。

证据及散列：`docs/classroom/dazi_fundamental_20261027/README_zh.md`、`REPORT.json`、`native_backend_report.json`、`MANIFEST.json`。原样节点/77层模型、原结果、新原生对照及原图已交付；服务器同项目 `dazi_fundamental_20261027/` 保存原始取证与新计算。此前CLASSROOM-DEMO-001只作为高阶案例保留，不能替代本例。

## CLASSROOM-NATIVE-COMPARE-001：QED与原生sdisp96直接基准（2026-10-27）

相速度、无耗散各向同性固体层；同教学版Linux原生二进制，不使用pysurf96/disba或DAzi内嵌surfdisp96代替sdisp96。77个模型/波型/阶数配置，包含28个原DAzi模型R/L基阶5096点、代表多阶、DRadi舍入失败模型、近邻模态与解析半空间。输入层参数和频率共同float32，根匹配阈值预定0.01 m/s；物理/数值/runtime分别保存。未修改生产求解器或GUI，未计算能量、核或反演。

默认sdisp96在d8_s2_c20、d12_s1_c30各一个Rayleigh基阶点漏过两低阶根，错选差172.654060、305.512574 m/s；独立WaveDispX动态刚度确认较低根存在。其余26个Rayleigh和全部28个Love基阶模型根集合匹配。271条QED独有未匹配记录全部通过独立局部验根（含不同阶数请求的重复，非271独立样本）。CPS独有27条中2条为半空间解析真根、4条明显越过半空间Vs、19条局部验根未通过、2条截止边界未定。

保留QED反例：本轮Linux forward在均匀半空间6频率中缺0.05/0.1 Hz，CPS与解析解一致；macOS开发二进制同输入未复现，原因未定。故不认证任何后端全模型完备性。搜索设置补查显示FACR=20恢复两个完整91周期基阶反例，浅层R/L近邻诊断分别在FACR20/FACL100恢复，代价增加；-D和FAC设置分别记录，超时不当作数学失败。

8组配置热调用3次重复、输出逐元素稳定；默认CPS/QED程序链耗时比1.30–74.78，仅此平台/请求，部分多阶输出覆盖不同，不称同精度纯内核或反演加速。证据及原数组/阈值/源码二进制散列/QC见 `docs/classroom/qed_sdisp96_benchmark_20261027/README_zh.md`、`cases.json`、`summary.json`、`independent_checks.json`、`provenance.json`、`MANIFEST.json`。完整原生输入/日志/输出留服务器同名目录。

## CLASSROOM-THREE-BACKEND-001：四层模型的基阶/低阶三方差异（2026-10-27 至 28）

按用户要求设计可导入教学软件的简单模型：h=10/10/10/inf m，Vp=700/1600/650/2200 m/s，Vs=300/800/260/1100 m/s，rho=1800/2100/1750/2300 kg/m³，Q=inf；平面 ISO 固体 Rayleigh，39–41.5 Hz、151点、M0–M3。36组粗筛与20组定向筛选后固定本例重算，属于有目的的示例选择，不是总体成功率评测。不含能量、波形、核或反演，生产算法和GUI未改。

同一输入交给教学版 QED forward、原生 sdisp96（FACR=5）和原 DAzi 冻结 surfdisp96（整段升序周期、按阶次调用）。约40.30 Hz，三者M0分别282.6329、377.1872、377.1875 m/s；默认sdisp96在3/151频点漏过低根、旧surfdisp96在151/151频点同编号明显偏离。这里是根覆盖和编号偏移，不宣称匹配物理模态的速度精度相差33%。FACR20剩1异常点；FACR100恢复全部604输出，与QED最大差0.0000536 m/s，必须同时保留这个控制结果。

独立P/SV全局波振幅匹配矩阵验证两个低根共302/302，最大局部根差4.972e-5 m/s、最大归一化最小奇异值3.092e-13；不构成全域根数证明。动态刚度复核301/302确认、1未定；初版最小绝对特征值检验会切换分支而错误拒绝根，初始和修订结果均保留，最终独立证据以波振幅匹配为准。本机实际教学计算适配层同CSV/GUI频率轴复现3个差异点，未重验各平台发行包。

模型、三方曲线、搜索参数控制、中文说明、可查看QC、脚本/输入/二进制散列、独立验证及MANIFEST见 `docs/classroom/three_backend_model_20261027/`。服务器同名目录保存全部筛选和原始程序输出；本地仅小型交付。未新增第三个GUI后端，未发布新版软件。

### CLASSROOM-THREE-BACKEND-001 显示扩展（2026-10-28）

用户要求按教学软件风格展示完整曲线。模型/物性/版本保持不变，频带扩为1–60 Hz（151等距点+原已定位3频点，共154）；三方重新计算，surfdisp96为完整升序周期向量，未切段或拼接原窗口。软件配色按模态、模型Vp/Vs/密度并列、三后端相同f/c显示范围；不改GUI。代码/轴/日志/QC和绘图前后数组散列一致性在 `docs/classroom/three_backend_model_20261027/full_view_20261028/`。保留所有缺根和跳点，新宽频带不继承原窄频带逐根验证结论。已检查全景及单后端并列图。

## ORIGINAL-PACKAGE-LVZ-001：原反演包正演与一维受控反演（2026-10-28）

用户授权合成观测受控测试。固定 DSurfTomo stable `a651faf92758c1b4c5eb494a4865ef797b9a4b1c`、jointBSVpVs `0424d5e73c26d7029dfb3f576ed5701fa6c5a9f6` 的原始 surfdisp96，以及原 DAzi 冻结同类源码；不修改数值程序，不以 disba 或原生 sdisp96 替代。学校服务器计算，本地保留小型图表与数组。没有运行完整三维层析、体波—面波联合反演或原包优化器。

沿用 CLASSROOM-THREE-BACKEND-001 的四层模型，QED 的39–41.5 Hz/51点基阶合成观测全部核对既有独立P/SV验根。只反演第三层 Vs，其余显式物性固定，不用材料关系导数。统一200–320 m/s/0.5网格再0.01细化：QED恢复260.00 m/s，三份原包均恢复263.84 m/s（真值260）；原包真模型相速度RMSE95.656721 m/s，偏移模型后降至0.680531 m/s。三包原始曲线与网格目标逐元素相同。QED自生成数据的零残差只是内部一致性基线。

另用d8_s1_c30区域20节点，经DSurf/joint原生材料运算、细分、球形入口得到77层。原91周期序列前51点（不超入口60容量）连续调用有35/51大偏差，最大1421.198057 m/s；同周期独立调用最大差降至0.002934 m/s；线性/等比51周期覆盖0.5–20s时最大差<0.003455 m/s。原CPS球形与原sphere导出数组的平面重算逐元素一致。35个较低QED候选均通过独立动态刚度局部验根，支持漏低根判断，不证明全域完备。

解析半空间控制四后端均通过，原包最大差0.0001783 m/s。完整1–60Hz展示另算54点，非反演观测。报告、配置/源码/二进制/交付哈希、原数组、审计与已查看QC见 `docs/classroom/original_packages_test_20261028/README_zh.md`、`report.json`、`MANIFEST.json`、`audit.json`。服务器同目录保留原始输入/输出与运行日志。结论仅限指定模型/调用策略；没有更改GUI或生产后端、没有计算能量或重发安装包。
