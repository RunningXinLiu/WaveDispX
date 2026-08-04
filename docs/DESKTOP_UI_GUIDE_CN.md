# WaveDispX 原生桌面工作台操作手册

这是一套直接运行源码的 PySide6 桌面界面，不需要浏览器，也不要求封装成 macOS `.app`。界面只组织参数、审核、运行与展示；正演、真实有限记录噪声合成、接收端互相关、频散成像、拾取和理论曲线都调用与 CLI/Notebook 相同的 `CaseSession → WaveDispXWorkflow`。

## 1. 启动

```bash
cd /Volumes/lx_exFAT/PassiveFW_versionstable
conda run -n seisloc python -m pip install -e . --no-build-isolation
conda run -n seisloc wavedispx-desktop
```

也可不依赖命令入口：

```bash
conda run -n seisloc python -m wavedispx.desktop
```

从指定配置启动：

```bash
conda run -n seisloc python -m wavedispx.desktop \
  --template examples/passive_two_layer.toml --page review
```

## 2. 五步工作流

1. **建立算例**：选主动/被动模板，在可视参数树集中编辑所有值；双击“当前值”列。每个参数使用中文名称和独立单位列；悬停或单击任意字段，会显示“控制什么、硬约束、推荐起点、调大/调小影响、当前设置判断”。模式、布局、分量、方法和 true/false 使用带中文解释的下拉选项。
2. **输入审核**：先画 Vp、Vs、密度、完整 source–receiver geometry、子波/噪声样例、PSD 和理论 Rayleigh/Love。`required` 失败时计算按钮锁定，并给出字段路径与改法。
3. **分阶段计算**：依次运行正演/Green、波形或接收端 NCF、频散与拾取；也可一键运行并保存。任务在后台线程中运行，界面不会冻结。
4. **结果浏览**：浏览 `summary.png`、分量/方法对照图以及 HDF5、CSV、manifest、`input.toml` 的实际路径和大小。
5. **流程与方法**：界面内说明真实有限记录处理、phase shift、FJ/Bessel、Hankel 及被动收敛边界。

![被动面阵集中参数页](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_passive_grid_parameters.png)

参数表下方的蓝色帮助卡会保留最近悬停/选中参数的完整说明，避免短暂 Tooltip 消失后无法对照。参数树与完整 TOML 是双向的：

- 修改参数树后，点“参数树 → 刷新 TOML”；
- 直接粘贴或修改 TOML 后，点“TOML → 应用到参数树”；
- 打开外部 TOML 时，相对 CSV 路径按该 TOML 所在目录解析；
- 任何修改都会使输入审核变为过期状态，必须重新审核后才能计算。

## 3. 模型、Geometry、源和文件输入

参数页顶部提供快捷操作：

| 操作 | 写入的配置 | 说明 |
|---|---|---|
| `＋/－ 模型层` | `model.layers[]` | 在半空间前增删有限层，末层 `thickness_m=inf` 始终保留 |
| `线阵` | `geometry.receivers.layout="line"` | 输入 x 起终点、道距和固定 y |
| `二维面阵` | `layout="grid"` | 输入 x/y 起终点与间距；被动还输入径向距离箱 |
| `接收器 CSV` | `layout="file"` | 读取任意二维 `x_m,y_m` |
| `源环/源线` | `ambient_sources` | 被动方位照明或道路噪声带 |
| `噪声源 CSV` | `ambient_sources.layout="file"` | 明确给出每个背景噪声源位置 |
| `炮点 CSV` | `active_source.x_m,y_m` | 从多炮表选择本次的一炮 |
| `PSD CSV` | `noise.psd_shape="file"` | 外部 `frequency_hz,power` 噪声谱 |

可直接试用的文件位于 `examples/observation_system/`：

- `receivers.csv`：21 个不规则二维接收器；
- `shots.csv`：3 个主动炮点；
- `noise_sources.csv`：48 个显式被动噪声源；
- `noise_psd.csv`：外部 PSD；
- `active_selected_shot.toml` 与 `passive_poisson.toml`：完整配置。

## 4. 主动/被动 × 线阵/面阵实测页面

### 4.1 主动源线阵

模板：`examples/active_quick.toml`。两层模型；16 道，x=5–80 m、道距 5 m；炮点 (0,0) m；vertical+horizontal；Ricker 8 Hz；主动默认 Hankel 成像。

![主动线阵输入审核](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_active_review.png)

### 4.2 主动源二维面阵

模板：`examples/manual_active_array.toml`。21×21 共 441 个接收器，显示真实二维炮检距；输入审核同时计算空间混叠和有效径向检距采样。

![主动面阵输入审核](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_active_array_review.png)

### 4.3 被动源线阵

模板：`examples/passive_quick.toml`。41 道、孔径 200 m；24 个分层方位抽样源环；60 s、6 个独立 10 s 窗；Ricker 型 PSD；被动默认 phase shift。它用于快速检查，会明确警告源数和方位覆盖尚未达到正式收敛标准。

![被动线阵输入审核](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_passive_line_review.png)

### 4.4 被动源二维面阵

模板：`examples/passive_grid_quick.toml`。x/y=-25–25 m、间距 5 m，共 121 个接收器；接收器对按 5 m 距离箱形成径向 NCF 道集。

![被动面阵输入审核](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_passive_grid_review.png)

正式被动收敛不应停留在 quick 参数。优先换到 `passive_two_layer.toml`，典型为 192–384 个方位均匀源和 600–1800 s 有限记录。源数改善方位照明，记录时长/独立窗数降低有限样本互相关方差，两者不能互相完全替代。

## 5. 分阶段运行与缓存

- **正演 / Green 传递**：主动产生炮集；被动建立每个源到每个接收器的带限 Green 传递。
- **波形 / 接收端 NCF**：主动沿用炮集；被动在每个独立窗生成各源时间函数，传播后在接收器端混合，再仅对接收器记录做互相关/互谱累积。
- **频散成像与拾取**：从炮集或径向 NCF 道集计算能量谱；理论白线独立计算，不参与能量生成。
- **一键运行并保存全部**：依次执行上述步骤并写完整结果。
- **缓存失效**：只改频散/拾取参数时保留正演和 NCF；改模型、源、Geometry 或同名输入文件内容时，下游缓存按内容哈希失效。

## 6. 结果页

主动 quick 完整结果页如下。左侧切换所有 PNG，右下检查每个输出文件的大小和绝对路径。

![主动结果页](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_active_results.png)

被动结果页会显示有限记录 NCF、频散能量、拾取、理论曲线和收敛指标。

![被动结果页](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_passive_results.png)

## 7. 报错怎么处理

界面不会只显示 Python 异常。错误框包含：

- **参数位置**：例如 `noise.window_s` 或 `geometry.receivers.coordinates_file`；
- **问题**：缺失、类型错误、几何关系不成立或采样约束失败；
- **怎么改**：给出可执行修改；
- **原始技术信息**：用于复现和开发排查。

输入审核表中红色 `FAIL` 是阻塞项；黄色 `WARN` 是建议项。先修复 required，再判断是否接受建议项。

## 8. phase shift、FJ 还是 Hankel

- `processing.dispersion_method="phase_shift"`：按候选相速度施加相移并沿径向距离叠加；被动默认。
- `"bessel"`：FJ/Bessel 核 `J0(ωr/c)`；适合二维轴对称解释，但有限孔径仍有核旁瓣。
- `"hankel"`：出射波 Hankel 核；主动默认。
- `processing.compare_methods=true`：在同一频率—速度网格输出三方法对照，用于辨别方法相关旁瓣/假频。

界面里的“流程与方法”页可随时查看同一说明：

![方法说明页](/Volumes/lx_exFAT/PassiveFW_versionstable/docs/images/desktop_methods_guide.png)

## 9. 自动截图与界面验收

桌面入口可生成真实窗口截图后自动退出：

```bash
QT_QPA_PLATFORM=offscreen conda run -n seisloc python -m wavedispx.desktop \
  --template examples/passive_grid_quick.toml \
  --page review --screenshot /tmp/wavedispx_review.png
```

这也是本手册所有界面截图的生成方式；不是设计稿或手工拼图。
