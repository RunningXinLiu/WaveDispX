# 教学版跨平台打包

本目录将 PySide6 界面、WaveDispX C++ 核、当前固定源码的 QEDispInv forward，以及 CPS sprep96/sdisp96/sdpsrf96 一起打包。学生端不需要 Python、Conda、编译器或配置 PATH。

2026-10-27 候选包的下载链接、逐平台结果、截图及限制见
[`PACKAGING_VALIDATION_20261027_zh.md`](../../docs/classroom/PACKAGING_VALIDATION_20261027_zh.md)。
运行验证使用对应平台的成品；Windows 使用原生窗口插件，避免 Qt offscreen 插件不支持字体的问题。

## 平台与分发

- Windows x64：ZIP 解压后运行 WaveDispX-Classroom.exe，保留整个文件夹。
- macOS arm64 / x64：分别构建 .app，以 tar.gz 保存权限和符号链接。
- Linux x64：在 Ubuntu 22.04 构建，tar.gz 解压后运行 WaveDispX-Classroom。
- 不承诺所有历史操作系统版本通用；目标机器还须满足所用 Qt/Python 的最低系统要求。Windows ARM、32 位系统、旧版 Linux 和旧版 macOS 不在本轮验证矩阵内。
- 当前为私有候选包；未做 Windows 商业代码签名或 Apple 公证。干净虚拟构建机的测试不能代替每种学生机器的人工试用。

GitHub Actions 工作流为 `.github/workflows/classroom-build.yml`，仅在独立构建分支 push 或手动调用时执行。不开通公开 Release；候选包、截图、自检 JSON 和 SHA256 在私有 Actions artifact，保留 14 天。正式给学生发包前须另存到课程共享目录。

## 验证门槛

先编译原生求解器，再冻结 Python 应用。将成品复制到含中文和空格的新路径，移除开发环境 PATH、PYTHONPATH 和后端覆盖设置，直接运行发行版 `--self-test OUTPUT_DIRECTORY`。

自检实际点击 GUI 的后端与计算按钮，分别计算 R/L 三个模态，与既有 v2 四层案例比较（缺根位置必须相同，速度容差 atol=0.002 m/s、rtol=2e-6）。随后勾选 ideal，比较复数响应（rtol=1e-7、atol=1e-18），验证曲线数组没有改变，并保存 R/L 截图、紧凑布局截图及结果 ZIP。失败的构建只上传诊断，不生成学生候选压缩包。

这属于既有案例的移植回归，不是全部模型、模态或极点行为的物理认证。既有 QED/CPS 高阶根差异保留；不填补缺根。

## 源码与许可

`vendor/SOURCE_MANIFEST.json` 记录精确源文件 SHA256。QED 来源为本地既有固定版本，不更新远端 main；CPS 为本地既有源码子集。上游数值代码不修改，单独的 CMake 配方只负责构建及现有平台分支。

当前 QED 根目录 LICENSE 与部分源码头的许可证表述存在差异，旧 CPS 源码也需要补齐对应版本的分发说明；不能用当前网页上的新版本许可证替代旧源码的授权记录。私有候选测试可以继续，正式学生分发前应明确所用版本的授权与随包源码/许可证要求，并包含 Python、Qt/PySide、数值库及编译器运行库的第三方声明。

## 本机复现

需要 C/C++/Fortran 编译器、Eigen3、HDF5 及独立 Python 3.12 环境。不要修改 Conda base。

```sh
cmake -S packaging/classroom/native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --parallel 2
python -m pip install -r packaging/classroom/requirements.txt
python setup.py build_ext --inplace
python packaging/classroom/fetch_font.py
CLASSROOM_NATIVE="$PWD/native-build/solvers" python packaging/classroom/freeze.py
python packaging/classroom/check_distribution.py --label local-test
```

Windows 构建配方使用 MSYS2 MinGW 原生求解器和 CPython/MSVC 的 Python 扩展；发行时包含所需 DLL，运行不依赖学生安装 MSYS2/WSL。所有较大的构建目录和候选包留在云端/学校服务器，本机仅保留代码、基准数组和少量 QC。

构建前运行 `python packaging/classroom/fetch_font.py` 下载固定提交的 Noto Sans SC 与 OFL 许可；脚本验证 Git blob 身份并保存 SHA256。字体仅下载到构建机，不提交大字体文件。Windows 冻结时还需设置 `CLASSROOM_MINGW_BIN` 为实际 MinGW bin 目录，工作流会自动传递。
