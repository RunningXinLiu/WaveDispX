# WaveDispX DAS 本地证据登记

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
