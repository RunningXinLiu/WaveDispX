"""Build the reader-facing WaveDispX notebooks with nbformat.

Run from the repository root after changing notebook content here.  Keeping the
source in Python avoids hand-editing notebook JSON and makes structural review
straightforward.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {
        "display_name": "Python (seisloc)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    return nb


SETUP = r"""
from pathlib import Path
from IPython.display import Image, display
import matplotlib.pyplot as plt

from wavedispx import CaseSession, WaveDispXWorkflow
from wavedispx.app_service import load_template_mapping
from wavedispx.preflight import checks_markdown, plot_preflight_review
from wavedispx.workflow_plotting import (
    plot_workflow_forward,
    plot_workflow_observations,
    plot_workflow_dispersion,
)

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / 'pyproject.toml').is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError('请从 PassiveFW_versionstable 项目内运行本 Notebook')
    PROJECT_ROOT = PROJECT_ROOT.parent
print(f'Project: {PROJECT_ROOT}')
"""


def active_notebook():
    return notebook(
        [
            markdown(
                r"""
# 主动源面波：一步步运行

## Goal

本 Notebook 只做主动源。流程为：**参数 → 输入图/检查 → 多分量炮集 → Hankel/FJ 频散 → 理论曲线 → 保存**。
第一次使用只需修改下面带“常改”的参数；完整参数可改 `INPUT_TOML`，或使用 `wavedispx-ui`。
"""
            ),
            markdown("## Setup\n\n### 1. 导入同一套 Python/C++ 计算核心"),
            code(SETUP),
            markdown(
                r"""
## Steps

### 2. 输入参数

- `INPUT_TOML=None`：使用本 Cell 参数；改为 TOML 路径则全盘读取文件。
- 距离和速度单位为 m、m/s，密度为 kg/m³，时间为 s。
- `component='both'` 同时输出 Rayleigh（垂直）和 Love（横向）分支。
"""
            ),
            code(
                r"""
INPUT_TOML = None  # 例如 PROJECT_ROOT / 'examples/active_two_layer.toml'

CASE = load_template_mapping(PROJECT_ROOT / 'examples/active_quick.toml')
CASE['run'].update({
    'name': 'notebook_active_quickstart',
    'output_directory': str(PROJECT_ROOT / 'output_python/notebook_active_quickstart'),
    'workers': 0,
})

# 模型：最后一层 thickness_m 必须是 inf（半空间）
CASE['model']['layers'] = [
    {'thickness_m': 10.0, 'vp_mps': 380.0, 'vs_mps': 220.0,
     'density_kgm3': 2000.0, 'qp': float('inf'), 'qs': float('inf')},
    {'thickness_m': float('inf'), 'vp_mps': 760.0, 'vs_mps': 440.0,
     'density_kgm3': 2000.0, 'qp': float('inf'), 'qs': float('inf')},
]

# 观测系统：这里是线阵；也可换成 layout='grid' 或 layout='file'
CASE['geometry']['receivers'] = {
    'layout': 'line', 'start_x_m': 5.0, 'stop_x_m': 80.0,
    'spacing_m': 5.0, 'y_m': 0.0,
}

# 主动源与 Ricker 子波（常改）
CASE['active_source'].update({
    'x_m': 0.0, 'y_m': 0.0, 'component': 'both',
    'dominant_frequency_hz': 8.0,
    'maximum_frequency_hz': 20.0, 'dt_s': 0.005,
})

# 频散范围（主动源默认 Hankel/FJ；可改 phase_shift 做交叉验证）
CASE['processing'].update({
    'dispersion_method': 'hankel',
    'frequency_min_hz': 3.0, 'frequency_max_hz': 18.0,
    'velocity_min_mps': 120.0, 'velocity_max_mps': 500.0,
    'velocity_step_mps': 2.0,
})
"""
            ),
            markdown("### 3. 构建算例并绘图确认（这里只计算理论曲线，不运行波场正演）"),
            code(
                r"""
if INPUT_TOML is None:
    case_session = CaseSession(PROJECT_ROOT)
    config, invalidated = case_session.apply_mapping(CASE)
    workflow = case_session.workflow
else:
    workflow = WaveDispXWorkflow.from_toml(INPUT_TOML)
    config = workflow.config
    invalidated = ['forward', 'analysis']

review = workflow.review_inputs()
print(checks_markdown(review))
figure = plot_preflight_review(review)
display(figure)
plt.close(figure)
assert review.ready, 'required 检查失败：请先修正输入'
"""
            ),
            markdown(
                "## Checks\n\n### 4. 可选分步运行\n\n把相应开关改为 `True`，可在一键运行前查看中间炮集或频散。"
            ),
            code(
                r"""
RUN_FORWARD_STAGE = False
if RUN_FORWARD_STAGE:
    workflow.compute_forward()
    figure = plot_workflow_forward(workflow)
    display(figure)
    plt.close(figure)
    display(workflow.status())
"""
            ),
            code(
                r"""
RUN_DISPERSION_STAGE = False
if RUN_DISPERSION_STAGE:
    workflow.compute_dispersion()
    figure = plot_workflow_dispersion(workflow)
    display(figure)
    plt.close(figure)
    display(workflow.status())
"""
            ),
            markdown(
                "## Results\n\n### 5. 一键全流程\n\n确认输入图后，只改下面第一行为 `True`，运行这一个 Cell 即可正演、成像、拾取、保存并显示完整汇总图。"
            ),
            code(
                r"""
RUN_ALL_NOW = False

if RUN_ALL_NOW:
    artifacts = workflow.run_all(save=True)
    print(f'完成：{artifacts.output_directory}')
    display(Image(filename=str(artifacts.summary_png)))
else:
    print('尚未启动正演。确认输入后把 RUN_ALL_NOW 改为 True。')
"""
            ),
            markdown(
                "## Next Steps\n\n正式高频算例可直接把 `INPUT_TOML` 指向 `examples/active_two_layer.toml`；日常文件导入和多参数编辑推荐运行 `wavedispx-ui`。"
            ),
        ]
    )


def passive_notebook():
    return notebook(
        [
            markdown(
                r"""
# 被动源面波：真实有限记录互相关

## Goal

本 Notebook 只做被动源。流程为：**参数 → 输入图/检查 → 离散波数法 Green 传递函数 → 所有噪声源混合 → 接收器 NCF → 相移频散 → 理论曲线 → 保存**。

处理阶段只接收混合后的接收器波形，不使用源编号，也不做逐源理想互相关。
"""
            ),
            markdown("## Setup\n\n### 1. 导入同一套 Python/C++ 计算核心"),
            code(SETUP),
            markdown(
                r"""
## Steps

### 2. 输入参数

快速模板使用 24 源、60 s，仅用于检查程序；要看收敛频散，请把源数改为 192–384、记录改为 600–1800 s，或读取 `passive_two_layer.toml`。
"""
            ),
            code(
                r"""
INPUT_TOML = None  # 正式算例：PROJECT_ROOT / 'examples/passive_two_layer.toml'

CASE = load_template_mapping(PROJECT_ROOT / 'examples/passive_quick.toml')
CASE['run'].update({
    'name': 'notebook_passive_quickstart',
    'output_directory': str(PROJECT_ROOT / 'output_python/notebook_passive_quickstart'),
    'workers': 0,
})

# 模型：最后一层是半空间
CASE['model']['layers'] = [
    {'thickness_m': 10.0, 'vp_mps': 380.0, 'vs_mps': 220.0,
     'density_kgm3': 2000.0, 'qp': float('inf'), 'qs': float('inf')},
    {'thickness_m': float('inf'), 'vp_mps': 760.0, 'vs_mps': 440.0,
     'density_kgm3': 2000.0, 'qp': float('inf'), 'qs': float('inf')},
]

# 观测系统：line / grid / file 均可；二维被动阵列需 distance_bin_m
CASE['geometry']['receivers'] = {
    'layout': 'line', 'start_x_m': -100.0, 'stop_x_m': 100.0,
    'spacing_m': 5.0, 'y_m': 0.0,
}

# 噪声源空间分布（正式推荐 count=192–384）
CASE['ambient_sources'].update({
    'layout': 'annulus', 'count': 24, 'seed': 42,
    'inner_radius_m': 250.0, 'outer_radius_m': 350.0,
    'angular_sampling': 'stratified', 'component': 'vertical',
})

# 有限时长随机噪声与 PSD（正式推荐 duration_s=600–1800）
CASE['noise'].update({
    'temporal_model': 'stationary_gaussian', 'seed': 314159,
    'duration_s': 60.0, 'window_s': 10.0, 'dt_s': 0.01,
    'psd_shape': 'ricker', 'minimum_frequency_hz': 3.0,
    'dominant_frequency_hz': 7.0, 'maximum_frequency_hz': 10.0,
    'independent_sources': True, 'overlap': 0.0,
})

# NCF 与频散：被动默认 phase_shift
CASE['processing'].update({
    'correlation_normalization': 'spectral_whitening',
    'dispersion_method': 'phase_shift',
    'frequency_min_hz': 3.5, 'frequency_max_hz': 9.5,
    'velocity_min_mps': 150.0, 'velocity_max_mps': 500.0,
    'velocity_step_mps': 2.0, 'distance_taper_fraction': 0.5,
})
"""
            ),
            markdown("### 3. 构建算例并确认 Vp/Vs/ρ、geometry、PSD 和理论频散"),
            code(
                r"""
if INPUT_TOML is None:
    case_session = CaseSession(PROJECT_ROOT)
    config, invalidated = case_session.apply_mapping(CASE)
    workflow = case_session.workflow
else:
    workflow = WaveDispXWorkflow.from_toml(INPUT_TOML)
    config = workflow.config
    invalidated = ['forward', 'observations', 'analysis']

review = workflow.review_inputs()
print(checks_markdown(review))
figure = plot_preflight_review(review)
display(figure)
plt.close(figure)
assert review.ready, 'required 检查失败：请先修正输入'
"""
            ),
            markdown(
                "## Checks\n\n### 4. 可选分步运行\n\n三个开关默认关闭。正式大算例可先只计算传递函数，再决定是否生成长记录 NCF。"
            ),
            code(
                r"""
RUN_TRANSFER_STAGE = False
if RUN_TRANSFER_STAGE:
    workflow.compute_forward()
    figure = plot_workflow_forward(workflow)
    display(figure)
    plt.close(figure)
    display(workflow.status())
"""
            ),
            code(
                r"""
RUN_NCF_STAGE = False
if RUN_NCF_STAGE:
    workflow.simulate_observations()
    figure = plot_workflow_observations(workflow)
    display(figure)
    plt.close(figure)
    display(workflow.status())
"""
            ),
            code(
                r"""
RUN_DISPERSION_STAGE = False
if RUN_DISPERSION_STAGE:
    workflow.compute_dispersion()
    figure = plot_workflow_dispersion(workflow)
    display(figure)
    plt.close(figure)
    display(workflow.status())
"""
            ),
            markdown(
                "## Results\n\n### 5. 一键全流程\n\n确认模型、源—接收器分布、PSD 和 required 检查后，只改下面第一行为 `True`。最终汇总图包含 Vp/Vs、ρ、geometry、混合记录/NCF 和频散。"
            ),
            code(
                r"""
RUN_ALL_NOW = False

if RUN_ALL_NOW:
    artifacts = workflow.run_all(save=True)
    print(f'完成：{artifacts.output_directory}')
    display(Image(filename=str(artifacts.summary_png)))
else:
    print('尚未启动正演。确认输入后把 RUN_ALL_NOW 改为 True。')
"""
            ),
            markdown(
                r"""
## Next Steps

- 快速程序检查：24 源、60 s。
- 可用被动频散：建议至少 192 源、600 s。
- 稳定展示/收敛基准：384 源、1800 s。
- 二维阵列结果是按台间距分箱后的径向平均；方位信息当前不会进入频散图。
"""
            ),
        ]
    )


def methods_notebook():
    return notebook(
        [
            markdown(
                r"""
# WaveDispX 方法、公式与适用边界

## Goal

解释正演、主动/被动记录、互相关、相移法、FJ/Hankel 与理论频散之间的关系，并用同一模型绘制理论曲线和线阵/面阵 geometry。此 Notebook 不运行耗时的波场正演。
"""
            ),
            markdown("## Setup\n\n### 1. 加载示例与绘图函数"),
            code(
                SETUP
                + r"""
from wavedispx.preflight import build_preflight_review
from wavedispx.workflow_plotting import plot_receiver_layout_comparison
"""
            ),
            markdown(
                r"""
## Steps

### 2. 离散波数法（DWM）正演计算什么

对水平层状介质，程序在频率—波数域组装分层刚度矩阵并求 Green 函数。主动源为

$$u_i(\mathbf{x},\omega)=G_{is}(\mathbf{x},\mathbf{x}_s,\omega)S(\omega).$$

Python 管理配置、几何、并行分块和输出，C++17 计算主要离散波数法频率/距离块。它是完整波场响应，不是直接画一条理论频散曲线。

### 3. 被动记录与互相关

每个独立噪声源 $q_s$ 先通过 Green 函数传播，随后在接收器处相加：

$$u_i(t)=\sum_s G_{is}(t)*q_s(t).$$

之后仅从有限时长混合波形计算

$$C_{ij}(\tau)=\langle u_i(t)u_j(t+\tau)\rangle_t.$$

互相关函数不接收源坐标或源编号。因此它模拟的是有限记录的接收端处理，而不是逐源理想 NCF 捷径。

`ambient_sources.component='three_component'` 时，每个源生成独立全局 $F_x,F_y,F_z$。程序按源—接收器方位将其分解为 $F_r,F_t$，通过 $G_{rr},G_{tt},G_{zr},G_{rz},G_{zz}$ 传播并旋回 X/Y/Z；互相关阶段再按台站对方向形成 RR/TT/ZZ。这是完整 Green 张量上的传播，不是频散图后加能量。
"""
            ),
            markdown(
                r"""
### 4. 相移法和 FJ/Hankel

相移成像对距离道集按候选速度补偿相位：

$$E(f,c)=\left|\sum_r \hat C(f,r)\exp\left(i\frac{2\pi fr}{c}\right)\right|.$$

它对有限孔径线阵和未完全扩散的被动 NCF 更直接，因此是当前被动默认。

FJ/Bessel 使用 $J_0(kr)$ 核，Hankel 使用柱面波核，适合径向波场或近似各向同性噪声场。主动炮集默认 Hankel/FJ；被动 Bessel/Hankel 建议作为方法敏感性对照。有限孔径、离散道距和距离窗会产生空间点扩散旁瓣，旁瓣不等于新的物理模态。
"""
            ),
            markdown("### 5. 同时查看线阵、二维面阵和理论频散"),
            code(
                r"""
line_case = load_template_mapping(PROJECT_ROOT / 'examples/passive_quick.toml')
grid_case = load_template_mapping(PROJECT_ROOT / 'examples/passive_grid_quick.toml')
line_review = build_preflight_review(
    CaseSession(PROJECT_ROOT).apply_mapping(line_case)[0]
)
grid_review = build_preflight_review(
    CaseSession(PROJECT_ROOT).apply_mapping(grid_case)[0]
)

figure = plot_receiver_layout_comparison({'Line array': line_review, '2-D grid': grid_review})
display(figure)
plt.close(figure)

figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
for mode in range(line_review.theory.rayleigh_mps.shape[1]):
    axis.plot(
        line_review.theory.frequency_hz,
        line_review.theory.rayleigh_mps[:, mode],
        label=f'Rayleigh mode {mode}',
    )
axis.set(xlabel='Frequency (Hz)', ylabel='Phase velocity (m/s)', title='Independent layered-model theory')
axis.grid(alpha=0.25)
axis.legend()
display(figure)
plt.close(figure)
"""
            ),
            markdown(
                r"""
## Checks

### 6. 结果解释边界

| 环节 | 当前稳定实现 | 不能直接解释为 |
|---|---|---|
| 模型 | 水平分层、各向同性、地表源/接收器 | 侧向非均匀或地形 |
| 主动 | 垂直/水平 Green 分量，Rayleigh/Love | 任意方位水平力未经旋转的面阵结果 |
| 被动 | 竖直快速路径，或物理 Fx/Fy/Fz 经五分量 Green 张量传播、有限混合记录、RR/TT/ZZ NCF | 逐源理想互相关或频散图后加能量 |
| 二维阵列 | 台站对按距离分箱、径向平均 | 方位各向异性成像 |
| 频散 | phase-shift / Bessel / Hankel 能量 | 每条亮纹都是物理模态 |

旁瓣和假频主要受孔径、道距、距离窗、噪声场方位覆盖和记录时长共同控制；增加噪声源只能改善照明，不能消除有限阵列的确定性旁瓣。
"""
            ),
            markdown(
                "## Next Steps\n\n主动操作见 `01_active_quickstart.ipynb`，被动真实处理见 `02_passive_quickstart.ipynb`；大量参数和文件导入使用 `wavedispx-ui`。"
            ),
        ]
    )


def index_notebook():
    return notebook(
        [
            markdown(
                r"""
# WaveDispX 工作流入口

原来的单本 Notebook 同时混合主动/被动、线阵/面阵、Cell/TOML 和分步/一键，阅读负担较大。现在按任务拆开：

1. **[01_active_quickstart.ipynb](01_active_quickstart.ipynb)**：主动源参数、炮集、Hankel/FJ、Rayleigh/Love。
2. **[02_passive_quickstart.ipynb](02_passive_quickstart.ipynb)**：真实有限时长混合噪声、接收端 NCF、相移频散。
3. **[03_methods_and_validation.ipynb](03_methods_and_validation.ipynb)**：离散波数法（DWM）、互相关、phase shift、FJ/Hankel、理论曲线与适用边界。

日常图形操作、TOML/CSV 上传、线阵/面阵/不规则阵列和参数提示，运行：

```bash
conda run -n seisloc wavedispx-ui
```

三个入口调用同一个 `WaveDispXWorkflow`，不会产生两套正演算法。
"""
            ),
            markdown(
                r"""
## Goal

- 想逐步理解或修改公式：打开对应 Notebook。
- 想快速输入参数、上传观测系统并运行：使用本地界面。
- 想批量复现：使用 `wavedispx run your_case.toml`。

完整参数定义见 [`../docs/INPUT_PARAMETERS_CN.md`](../docs/INPUT_PARAMETERS_CN.md)。
"""
            ),
        ]
    )


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "WaveDispX_workflow.ipynb": index_notebook(),
        "01_active_quickstart.ipynb": active_notebook(),
        "02_passive_quickstart.ipynb": passive_notebook(),
        "03_methods_and_validation.ipynb": methods_notebook(),
    }
    for name, nb in outputs.items():
        path = NOTEBOOKS / name
        # The checked-in tutorials contain the newer single-PARAMETERS-cell
        # layout and expanded equation notes. Preserve that reader-facing
        # structure when this legacy builder refreshes notebook metadata/index.
        if path.is_file() and name in {
            "01_active_quickstart.ipynb",
            "02_passive_quickstart.ipynb",
            "03_methods_and_validation.ipynb",
        }:
            current = nbf.read(path, as_version=4)
            if name == "03_methods_and_validation.ipynb" or any(
                "PARAMETERS" in cell.source for cell in current.cells
            ):
                nb = current
        nbf.write(nb, path)
        print(path)


if __name__ == "__main__":
    main()
