# 单后端简洁界面验收 · 2026-10-27

> 历史阶段记录：后续跨平台成品检查见 [打包验证记录](PACKAGING_VALIDATION_20261027_zh.md)，不扩大本页的科学验证范围。

本轮按用户要求调整界面与运行编排：一次选一个 backend，界面不再比较两求解器；ideal 默认关闭，勾选时独立计算叠加。物理模型、求根适配器、全局刚度核、ideal 算子和逐频归一化公式未改。

## 功能与输出

- 参数区不使用 QScrollArea，只有层模型表格滚动。默认 1440×900，最小 1200×780；两个尺寸均通过真实 Cocoa 窗口截图检查。
- 左侧为 Vp/Vs/ρ 共用深度的阶梯图：速度使用底部 m/s 轴，密度使用顶部 kg/m³ 轴和紫色虚线；右侧只有 R/L 两个标签。
- 所有模型字段、计算参数、显示参数、路径配置、关键按钮均有中文 tooltip；表头与单元格也有说明。
- QEDispInv 或 CPS 单独执行；未选中的后端不探测、不调用。曲线导出 CSV 字段固定为 frequency_hz,mode,phase_velocity_mps,backend。
- ideal 单独任务保留 curve 数组；关闭只隐藏，再次开启可复用；改变 ideal 参数重算响应而不重算曲线。失败保留曲线和导出功能。
- 结果包 schema 改为 wavedispx-classroom/2，明确 backend；ideal 算子保持 wavedispx-classroom/1。历史 v1 QC 保留在 qc/，本轮另存 qc_v2/。

## 验证

命令：

```sh
PYTHONPATH=python_src conda run --no-capture-output -n seisloc pytest -q python_tests/test_classroom.py python_tests/test_qedisp.py --tb=short
PYTHONPATH=python_src conda run --no-capture-output -n seisloc python scripts/check_classroom_ui.py
```

测试日志见 `qc_v2/tests.txt`：本版 11 项测试加既有 QEDisp 4 项回归。
新增检查包括：未选中后端被故意设为不可调用仍能完成选择后端计算；勾选 ideal 时禁止曲线求解器再执行；ideal 失败保留曲线；开关缓存复用；修改 ideal 参数不改变曲线；导出仅含所选后端；无参数滚动区域；所有控件可见；中文 tooltip；密度轴数据与深度范围正确。

`qc_v2/validation.json` 保存源码、输出包 SHA256 和旧版数组回归结果。使用同一四层低速夹层、73频率、三阶，与首版 `qc/low_velocity_layer.zip` 比较：

- QEDispInv R/L 曲线与旧版逐元素相等（含 NaN 位置）。
- CPS R/L 曲线与旧版逐元素相等（含 NaN 位置）。
- ideal 两波型原始复响应与原始功率逐元素相等。
- 真实桌面后台计算中，勾选 ideal 前后两波型曲线 SHA256 相同，记录在 `qc_v2/ui_checks.json`。

可查看结果：`desktop_curves.png`（默认不叠加）、`desktop_rayleigh_ideal.png`、`desktop_love_ideal.png`、`desktop_compact.png`（1200×780）、`desktop_cps.png`。
独立输出包：`qedisp_curves.zip`、`qedisp_ideal.zip`、`cps_curves.zip`。图片均经过渲染检查。

这是指定案例的运行和显示回归，不扩大数值验收范围。首版低速夹层的 Love 4 Hz/M1 根覆盖分歧仍保留在各自导出结果中；本轮未更改求根或判定原因。gamma/网格全带收敛、任意多层高阶完备性、有限 Q、跨平台发行包仍未验收。
