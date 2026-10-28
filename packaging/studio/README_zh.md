# WaveDispX Studio 桌面试用包

开发者主页：https://runningxinliu.github.io/

联系邮箱：xinliu_geo@outlook.com

感谢 Pan Lei 的 [QEDispInv](https://github.com/pan3rock/QEDispInv)，提供理论频散计算后端。

这是综合Studio，不是Classroom教学版。包内包含Python/Qt运行环境、CPU正演扩展、理论程序及群速度库、示例和中文字体；用户不需要安装Conda、Python或编译器，不使用GPU。

## 使用

- Windows x64：完整解压文件夹，双击`WaveDispX-Studio.exe`；保留`_internal`，不要只复制exe。
- macOS：分别提供Apple芯片和Intel构建，解压后打开`WaveDispX-Studio.app`。当前候选未签名、未公证；不要通过全局关闭系统安全检查来运行。
- Linux x64：解压完整目录，运行`WaveDispX-Studio`；测试基线为Ubuntu 22.04，不承诺所有发行版。
- 默认普通窗口；软件名后的网站可点击。「关于」内有个人网站、邮箱、版本与致谢。
- 内置示例默认结果路径为系统文档目录下`WaveDispX/results/案例名`，不写入应用目录。用户自己导入TOML的输出路径原样保留；参数可另存，结果需单独保存。
- 不要覆盖自己的输入文件；将内置CSV例子复制到可写工作目录再修改。详细字段与限制见包内`docs/studio/README_zh.md`及`examples/custom_geometry/README_CN.md`。
- 保存计算结果时，外部坐标/活动/PSD输入会复制到结果目录的`inputs/`，input.toml改为相对于该目录引用，并记录SHA256；请整体保留结果文件夹。不会移动或修改原CSV。

## 打包与验证

使用各自操作系统原生构建；工作流`.github/workflows/studio-build.yml`。独立环境安装`packaging/studio/requirements.txt`，不更改本机seisloc/base；沿用已声明依赖，不引入新数值后端。

```sh
cmake -S packaging/studio/native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --parallel 2 --target forward qedisp_group
python setup.py build_ext --inplace
python packaging/classroom/fetch_font.py
STUDIO_NATIVE="$PWD/native-build/solvers" python packaging/studio/freeze.py
python packaging/studio/check_distribution.py --label linux-x64
```

Windows需现有MSYS2工具链和CLASSROOM_MINGW_BIN；原生源码保持不变。成品迁移至含中文与空格的路径，移除开发环境PATH和求解器设置，再执行实际GUI计算、保存、重新读取与数值对照。只在验收通过后生成候选压缩包。

固定8个包装回归：主动矢量、被动三分量、主动/被动DAS线阵与3×3小面阵、非均匀活动源面阵、HVSR。使用已有模板，DAS面阵明确缩为9通道包装fixture，workers=2；不修改发布模板，不称大型面阵验收。另执行默认主动兼容流程。源环境基准由服务器生成，原始输入/数组保留，包内参考和SHA256可追溯。比较无增益/时间/相位拟合：原始波形/NCF/能量/HVSR相对L2≤1e-5；理论相速度atol=.002m/s、rtol=2e-6；缺失掩码相同，坐标轴rtol/atol=1e-12。失败不放宽阈值或生成候选。

这是平台移植回归，不是所有模型或统计收敛认证。Love-DAS、逐源方向目录、远程任务提交及强制取消仍未实现。没有为使截图更漂亮改变物理模型、源窗或成像算子。

## 来源与许可

保留原程序Readme.txt、QEDispInv及Python/Qt等依赖的声明；与已有教学版第三方记录一并随包提供。仓库尚无统一顶层许可证，上游快照根LICENSE与源码头的不同表述仍需核对；不宣称完整许可审计已通过。正式再分发策略、平台签名/公证和更多实机试用与数值验收分别记录，不以打包成功代替它们。
