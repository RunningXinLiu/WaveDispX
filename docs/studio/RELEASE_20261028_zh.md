# Studio阶段收尾｜2026-10-28

用户完成交互试用并确认没有操作阻断，授权同步GitHub及完善图文介绍。本次发布源码、示例、操作说明和小型QC证据；不是新的独立安装包，不更新既有Classroom教学版Release，不把界面验收等同于科学验证。

## 本阶段交付

- 普通可缩放Qt窗口：模型、Geometry、结果同屏，中文hover、错误定位与修改建议，输入→预览→计算→保存。
- Rayleigh/Love及计算产品显式选择；不可用结果页解释未选择、不激发、缺分量、零响应、未支持或失败。
- 主动矢量力；被动三分量独立/定向H/V源功率；真实有限混合记录互相关。空间方位与力方向分开输入。
- 可选DAS有限标距垂直源路径；可选原始XYZ噪声HVSR及逐站/多站/热图显示，默认关闭HVSR。
- CPU workers、频率/源分块；源台CSV和活动表导入；三套非均匀源完整输入与参数说明。
- [中文操作手册](README_zh.md)、[自定义输入包](../../examples/custom_geometry/README_CN.md)、[完整窗口与结果图](custom_inputs_20261028/README_zh.md)。仓库首页区分Studio、Classroom、Notebook/API，保留教学版下载和许可说明。

## 验证记录

发布前在GitHub最新main（70afecc）基础上单独整合Studio，保留Classroom和第三方固定源码，不覆盖原工作目录或用户窗口。上一轮73项相关测试及三例服务器CPU实际计算见[输入包验证](custom_inputs_20261028/README_zh.md)。发布整合回归：**126 passed，1 warning，60.92s**；覆盖Studio、源产品、HVSR、输入包、主动源、DAS、绘图轴及Classroom，不累计历史通过数量。警告为隐藏画布constrained_layout尺寸不足，未屏蔽。首次命令误列不存在的test_passive.py，未执行测试；纠正后完成下列实际测试集。

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=python_src \
  conda run --no-capture-output -n seisloc pytest -q \
  python_tests/test_custom_geometry.py python_tests/test_studio_ux.py \
  python_tests/test_studio.py python_tests/test_studio_acceptance.py \
  python_tests/test_app_service.py python_tests/test_desktop.py \
  python_tests/test_source_products.py python_tests/test_hvsr_view.py \
  python_tests/test_classroom.py python_tests/test_active.py \
  python_tests/test_das.py python_tests/test_plotting_axes.py --tb=short
```

运行前显式设置QEDISPINV_FORWARD为本机已有可执行文件路径；此命令不是无依赖环境的一键安装。

本轮仅整理/同步已实现代码及文档，没有重新设计模型、源分布、物理算子、权重或数值默认，没有安装依赖、重新编译或更换QED。测试复用本机已有C++/QED二进制；大型波形仍在服务器，仓库只包含说明所需小图、报告和小型回归快照。个人AGENTS配置、编译模块、缓存、交付ZIP及无关模型试验不加入本次提交。

## 明确保留的边界

- 自动f/v/df/dv和记录时长是候选建议，不保证所有模型、阵列、模态收敛。
- 小孔径、非均匀照明和有限记录的宽峰/旁瓣仍保留；高阶弱峰不自动补齐，QED理论不替代观测能量。
- Love-DAS、水平/倾斜力DAS、逐源不同方向、带方向事件目录、多炮批量、远程提交及强制取消尚未提供。
- HVSR来自原始XYZ，不是NCF谱比或理论Rayleigh椭圆率；独立H/V和定向H/V新极化只支持现有独立平稳高斯噪声组合。
- Studio新计算产品不支持多方法比较/自动拾取；旧流程继续保留。Classroom的ideal响应与有限阵列炮集/NCF能量不混称。
- 隐藏/小画布的布局警告仍可能出现；当前Studio并非已经完成四平台安装包验收。第三方许可待核对项沿用现有声明，不新增统一许可证。

后续优先修复真实使用问题；模型/源/算子变化需要重新做对应验证，不继承其他案例的通过结论。
