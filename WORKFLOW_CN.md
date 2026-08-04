# WaveDispX 主动/被动波场与频散工作流

> 新的日常入口是本地界面 `conda run -n seisloc wavedispx-ui`。Notebook 已拆为主动源、被动源
> 和方法说明三本，`notebooks/WaveDispX_workflow.ipynb` 仅作为入口导航。本文件后续内容主要
> 记录原 MATLAB 基准流程。

## 环境

- MATLAB R2026a（需要 Parallel Computing Toolbox、Signal Processing Toolbox）
- `seisloc` Conda 环境（理论频散使用其中的 `disba`）

在 MATLAB 中先进入本目录：

```matlab
cd('/Volumes/lx_exFAT/PassiveFW_versionstable')
addpath(pwd, fullfile(pwd,'lx-passive seismic modeling by DWM'))
```

## 被动源端到端流程

快速回归模型：

```matlab
results = run_passive_end_to_end('Quick',true,'Plot',true);
```

正式模型（384 个分层均匀方位环形源、800--1200 m 源环、统一 12 Hz
Ricker 源谱、每个位置 24 次独立事件、121 道、600 m 孔径、60 s）：

```matlab
results = run_passive_end_to_end('Quick',false,'Plot',true);
```

默认流程依次执行：

1. 两层介质离散波数法正演；
2. 将所有源的随机时刻、随机权重响应叠加成有限时长混合记录 `traces.z`；
3. 把混合记录分窗、谱白化，再直接计算接收点互相关；处理阶段不使用源编号；
4. 正负时延对称叠加、台间距分箱与时间窗平均；
5. phase-shift 频散能量，并在相位归一化后施加 Tukey 0.5 距离窗；
6. 动态规划基阶脊线拾取；
7. `disba` 独立理论 Rayleigh/Love 频散；
8. 保存 MAT、CSV、PNG 和可选的逐台站对 NCF。PNG 同时包含 Vp/Vs、密度、
   source-receiver geometry、NCF 道集和频散能量。

默认 `CorrelationMode='mixed'`。逐源分离的理想集合期望只保留为算法上限对照：

```matlab
ideal = run_passive_end_to_end( ...
    'CorrelationMode','source-separated','Plot',true);
```

它利用模拟中已知的源编号，不应当作为现实被动处理结果。

输出位于：

```text
lx-passive seismic modeling by DWM/output/passive_时间戳/
```

## 主动源流程

```matlab
run(fullfile(pwd,'active seismic modeling by DWM','main_active_twolayer.m'))
```

输出包括 Rayleigh/Love 频散能量、理论模态、自动拾取和完整波场：

```text
active seismic modeling by DWM/output/active_时间戳/
```

## 独立调用主要模块

```matlab
% NCF 距离分箱
[gather,t,r,count] = assemble_ncf_gather(ncf_folder, ...
    'Component','z','Branch','symmetric','DistanceBin',5);

% 正式处理：只输入已经混合的有限记录，不使用源编号
[gather,t,r,count] = compute_ncf_gather_from_traces( ...
    traces.z,SR.rx,dt,'SegmentLength',10,'Overlap',0.5, ...
    'MaxLag',4,'DistanceBin',5, ...
    'Normalize','spectral-whitening','Algorithm','auto');

% 仅用于理想上限对照，不作为现实处理结果
[ideal_gather,ideal_t,ideal_r] = compute_source_separated_ncf_gather( ...
    TT.z,SR.rx,dt,'MaxLag',4,'DistanceBin',5, ...
    'SpectrumNormalization','phase','WaterLevel',0);

% phase / bessel / hankel 频散能量
[energy,f,v] = surface_wave_dispersion(gather,r,dt,150:2:500,20, ...
    'Method','phase','Component','vertical', ...
    'DistanceTaperFraction',0.5);

% 如果资料满足近似各向同性噪声场，可对照 F-Bessel；空间窗可减弱旁瓣
[energy_bessel,f_bessel] = surface_wave_dispersion( ...
    gather,r,dt,150:2:500,20,'Method','bessel', ...
    'DistanceTaperFraction',0.25);

% 自动脊线拾取
picks = pick_dispersion_ridge(energy,f,v, ...
    'FrequencyRange',[3 18.8],'VelocityRange',[170 460]);

% 理论频散
theory = compute_theoretical_dispersion(model,f, ...
    'RayleighModes',4,'LoveModes',4);
```

## 真实混合噪声时长收敛

端到端正演后，可以导出混合记录和垂直 DWM 传递响应：

```matlab
input_file = export_realistic_forward_inputs(results);
```

随后生成独立平稳随机源时间函数。每个时间窗内先把全部源传播到接收端并相加，
再对混合后的 121 道数据做现场式互相关；处理中不读取源编号：

```matlab
[cases,metrics] = run_realistic_ambient_convergence(input_file, ...
    'Durations',[60 600 1800], ...
    'WindowLength',20, ...
    'Seed',314159);
```

当前固定随机序列结果为：60/600/1800 s 的基阶支持率分别为
96.3%/100%/100%，能量集中度为 0.266/0.474/0.578，中位半功率宽度为
16/14/14 m/s。600 s 相对 1800 s 的整幅频散能量 L2 变化约 9.85%。

## 测试

```matlab
results = runtests(fullfile(pwd,'tests'),'IncludeSubfolders',true);
assertSuccess(results)
```

当前回归覆盖几何可复现性、记录长度保护、磁盘/内存 NCF 距离叠加、
逐源集合互相关（含 DWM 极小振幅）、混合记录空间 FFT 与逐对 `xcorr`
逐元素一致性、
Tukey 距离窗旁瓣抑制、已知速度波场拾取、频散图像质量指标和理论曲线复核。

## 被动频散连续性诊断

旧默认把每个源位置只激发一次随机主频 Ricker 脉冲。它是稀疏瞬态叠加，
不是平稳环境噪声；短记录中的源间交叉项、随频率改变的方位照明、有限孔径
F-Bessel 旁瓣会使伪脊间歇性强于真实脊。现在用以下量化指标检查：

```matlab
m = dispersion_continuity_metrics(energy,f,v, ...
    theory.frequency,theory.rayleigh(:,1), ...
    'FrequencyRange',[3 18.8]);

q = dispersion_image_quality_metrics(energy,f,v, ...
    theory.frequency,theory.rayleigh(:,1), ...
    'FrequencyRange',[3 18.8]);
```

主要问题由三部分组成：有限混合记录中的源间交叉项、有限阵列点扩散函数、
以及真实高阶模态/少量体波近场能量。正式合成默认直接处理混合有限记录，
600 m 孔径对应 3.939 s NCF，并在相位归一化以后施加 Tukey 0.5 距离窗。
此前把空间窗乘在相位归一化之前会把窗权重除掉，现已修复并有回归测试。

逐源理想上限在 3--18.8 Hz 内的基阶占优率为 100%、集中度 0.726；它不再
作为正式结果。真实混合噪声的 60 s 记录仍有明显随机交叉项，600 s 后主脊
连续稳定，1800 s 进一步降低随机能量。

## 物理限制

- 模型为水平层状、各向同性、地表源/地表接收；Python/C++ 版接收器可为规则线阵或 CSV
  输入的任意二维阵列，但被动频散会按台间距作径向平均；
- 5 m 道间距下，当前模型建议把被动频散限制在约 20 Hz 以下；
- 600 m 孔径按至少约 3 个波长的稳健准则，当前 QC 下限约为 2.2 Hz；
  由于源谱从 3 Hz 开始，正式图使用 3 Hz 作为下限；
  全频能量仍保存在 MAT 中，但最终图和自动拾取不把低频当成已分辨结果；
- F-Bessel 需要更接近各向同性、充分收敛的噪声场；有限事件模拟优先使用
  phase-shift，并把 F-Bessel 当作方法敏感性对照；
- `Damp_p/Damp_s` 非零时，弹性理论曲线仅作为近似；
- 自动拾取的不确定性是局部半功率宽度，正式研究应增加时间窗、源和台站对 bootstrap。
