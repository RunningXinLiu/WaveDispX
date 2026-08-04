function tests = test_surface_wave_pipeline
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
repository_root = fileparts(fileparts(mfilename('fullpath')));
testCase.TestData.repository_root = repository_root;
addpath(repository_root, ...
    fullfile(repository_root,'lx-passive seismic modeling by DWM'));
end

function testGeometryIsReproducibleAndAreaUniform(testCase)
SR1 = S_R_geometry(2000,'Seed',7,'InnerRadius',100,'OuterRadius',200);
SR2 = S_R_geometry(2000,'Seed',7,'InnerRadius',100,'OuterRadius',200);
verifyEqual(testCase,SR1.sx,SR2.sx);
verifyEqual(testCase,SR1.sy,SR2.sy);
r2 = SR1.sx.^2+SR1.sy.^2;
verifyGreaterThanOrEqual(testCase,min(r2),100^2);
verifyLessThanOrEqual(testCase,max(r2),200^2);
% Uniform-area annulus has mean squared radius (r1^2+r2^2)/2.
verifyLessThan(testCase,abs(mean(r2)-25000),800);

stratified=S_R_geometry(96,'Seed',9,'InnerRadius',100, ...
    'OuterRadius',200,'AngularSampling','stratified');
azimuth=sort(mod(atan2(stratified.sy,stratified.sx),2*pi));
azimuth=azimuth(:);
maximum_gap=max(diff([azimuth;azimuth(1)+2*pi]));
verifyLessThanOrEqual(testCase,maximum_gap,4*pi/96);
end

function testShortPassiveRecordFailsBeforeForwardSolve(testCase)
model = reference_model;
source.number=1; source.recordlength=1; source.dt=.01;
source.maxfre=10; source.fmin=3; source.fmax=8;
source.Fx=0; source.Fy=0; source.Fz=-1;
SR.sx=300; SR.sy=0; SR.rx=-50:5:50; SR.ry=zeros(size(SR.rx));
verifyError(testCase,@() PassiveFW(model,source,SR,'v'), ...
    'PassiveFW:RecordTooShort');
end

function testNcfFilesAssembleByDistance(testCase)
dt=.01; t=(0:dt:4)';
traces.x=zeros(numel(t),3); traces.y=traces.x; traces.z=traces.x;
for station=1:3
    traces.z(:,station)=exp(-((t-(1+(station-1)*.1))/.03).^2);
end
traces.t=t';
SR.rx=[0 5 10]; SR.ry=[0 0 0]; source.dt=dt;
output_folder=[tempname '_ncf']; mkdir(output_folder);
cleanup=onCleanup(@() rmdir(output_folder,'s')); %#ok<NASGU>
summary=compute_ncf(traces,SR,source,output_folder, ...
    'SegmentLength',4,'Overlap',.5,'MaxLag',.5);
[gather,lag,distance,counts]=assemble_ncf_gather( ...
    output_folder,'DistanceBin',5);
[memory_gather,memory_lag,memory_distance,memory_counts]= ...
    compute_ncf_gather_from_traces(traces.z,SR.rx,dt, ...
    'SegmentLength',4,'Overlap',.5,'MaxLag',.5,'DistanceBin',5);
verifyEqual(testCase,summary.file_count,3);
verifySize(testCase,gather,[51 2]);
verifyEqual(testCase,distance,[5 10]);
verifyEqual(testCase,counts,[2 1]);
verifyEqual(testCase,lag(end),.5,'AbsTol',1e-12);
verifyEqual(testCase,memory_lag,lag,'AbsTol',1e-12);
verifyEqual(testCase,memory_distance,distance);
verifyEqual(testCase,memory_counts,counts);
verifyEqual(testCase,memory_gather,gather,'RelTol',2e-6,'AbsTol',1e-12);
end

function testPhaseShiftAndPickerRecoverKnownVelocity(testCase)
dt=.002; t=0:dt:3; distance=5:5:200; true_velocity=250;
data=zeros(numel(t),numel(distance));
for trace_index=1:numel(distance)
    data(:,trace_index)=rickerkai( ...
        t-distance(trace_index)/true_velocity,10);
end
velocity=(150:2:350)';
[energy,frequency]=surface_wave_dispersion( ...
    data,distance,dt,velocity,20,'Method','phase', ...
    'FrequencyMin',2,'ZeroPadding',4);
picks=pick_dispersion_ridge(energy,frequency,velocity, ...
    'FrequencyRange',[5 15],'VelocityRange',[180 320], ...
    'Smoothness',.2,'MaxJump',20);
verifyLessThanOrEqual(testCase, ...
    abs(median(picks.phase_velocity_mps)-true_velocity),2);
end

function testPhaseDistanceTaperSuppressesFarSidelobes(testCase)
dt=.002;
time=(0:dt:4)';
distance=5:5:600;
true_velocity=250;
data=zeros(numel(time),numel(distance));
for distance_index=1:numel(distance)
    data(:,distance_index)=rickerkai( ...
        time-(.5+distance(distance_index)/true_velocity),10);
end
velocity=(150:500)';
[rectangular,frequency]=surface_wave_dispersion( ...
    data,distance,dt,velocity,15,'Method','phase', ...
    'FrequencyMin',5,'ZeroPadding',4,'DistanceTaperFraction',0);
[hann,frequency_hann]=surface_wave_dispersion( ...
    data,distance,dt,velocity,15,'Method','phase', ...
    'FrequencyMin',5,'ZeroPadding',4,'DistanceTaperFraction',1);
[~,frequency_index]=min(abs(frequency-10));
far=abs(velocity-true_velocity)>=60;
verifyEqual(testCase,frequency_hann,frequency,'AbsTol',1e-12);
verifyLessThan(testCase,max(hann(far,frequency_index)), ...
    0.5*max(rectangular(far,frequency_index)));
end

function testLinearFftNcfMatchesPairwiseReference(testCase)
rng(17,'twister');
dt=.01;
receiver_x=-15:5:15;
source_response=randn(129,numel(receiver_x),3);
[pairwise_gather,pairwise_lag,pairwise_distance,pairwise_counts, ...
    pairwise_spectrum]=compute_source_separated_ncf_gather( ...
    source_response,receiver_x,dt,'MaxLag',.4,'DistanceBin',5, ...
    'SpectrumNormalization','phase','WaterLevel',0, ...
    'Algorithm','pairwise');
[fast_gather,fast_lag,fast_distance,fast_counts,fast_spectrum]= ...
    compute_source_separated_ncf_gather( ...
    source_response,receiver_x,dt,'MaxLag',.4,'DistanceBin',5, ...
    'SpectrumNormalization','phase','WaterLevel',0, ...
    'Algorithm','linear-fft');
verifyEqual(testCase,fast_lag,pairwise_lag,'AbsTol',1e-12);
verifyEqual(testCase,fast_distance,pairwise_distance,'AbsTol',1e-12);
verifyEqual(testCase,fast_counts,pairwise_counts);
verifyEqual(testCase,fast_spectrum,pairwise_spectrum, ...
    'RelTol',1e-10,'AbsTol',1e-10);
verifyEqual(testCase,fast_gather,pairwise_gather, ...
    'RelTol',1e-10,'AbsTol',1e-10);
end

function testMixedTraceLinearFftMatchesPairwiseReference(testCase)
rng(23,'twister');
dt=.01;
receiver_x=-15:5:15;
data=randn(501,numel(receiver_x));
[pairwise_gather,pairwise_lag,pairwise_distance,pairwise_counts]= ...
    compute_ncf_gather_from_traces(data,receiver_x,dt, ...
    'SegmentLength',2,'Overlap',.5,'MaxLag',.4, ...
    'Normalize','spectral-whitening','Branch','symmetric', ...
    'DistanceBin',5,'Algorithm','pairwise');
[fast_gather,fast_lag,fast_distance,fast_counts]= ...
    compute_ncf_gather_from_traces(data,receiver_x,dt, ...
    'SegmentLength',2,'Overlap',.5,'MaxLag',.4, ...
    'Normalize','spectral-whitening','Branch','symmetric', ...
    'DistanceBin',5,'Algorithm','linear-fft');
verifyEqual(testCase,fast_lag,pairwise_lag,'AbsTol',1e-12);
verifyEqual(testCase,fast_distance,pairwise_distance,'AbsTol',1e-12);
verifyEqual(testCase,fast_counts,pairwise_counts);
verifyEqual(testCase,fast_gather,pairwise_gather, ...
    'RelTol',1e-9,'AbsTol',1e-11);
end

function testIndependentTheoryMatchesBundledCurves(testCase)
model=reference_model;
frequency=(.2:.2:20)';
theory=compute_theoretical_dispersion(model,frequency, ...
    'RayleighModes',2,'LoveModes',2);
reference=load(fullfile(testCase.TestData.repository_root, ...
    'active seismic modeling by DWM','Rayleigh_Love_DC.mat'),'vr','vl');
rayleigh_error=max(abs(theory.rayleigh(:,1)- ...
    reference.vr(1:numel(frequency),2)),[],'omitnan');
love_error=max(abs(theory.love(:,1)- ...
    reference.vl(1:numel(frequency),2)),[],'omitnan');
verifyLessThan(testCase,rayleigh_error,1e-3);
verifyLessThan(testCase,love_error,1e-3);
end

function testContinuityMetricDetectsCompetingBrokenRidge(testCase)
frequency=1:20;
velocity=(100:400)';
theory_velocity=250*ones(size(frequency));
good_energy=exp(-0.5*((velocity-theory_velocity)/4).^2);
broken_energy=good_energy;
for frequency_index=2:2:numel(frequency)
    false_velocity=160+180*mod(frequency_index/2,2);
    broken_energy(:,frequency_index)=broken_energy(:,frequency_index)+ ...
        1.5*exp(-0.5*((velocity-false_velocity)/3).^2);
end
good=dispersion_continuity_metrics(good_energy,frequency,velocity, ...
    frequency,theory_velocity,'VelocityTolerance',12);
broken=dispersion_continuity_metrics(broken_energy,frequency,velocity, ...
    frequency,theory_velocity,'VelocityTolerance',12);
verifyEqual(testCase,good.dominant_ridge_fraction,1,'AbsTol',1e-12);
verifyLessThan(testCase,broken.dominant_ridge_fraction,0.75);
verifyGreaterThan(testCase,broken.global_large_jump_fraction,0.75);
sidelobe_good=dispersion_sidelobe_metrics(good_energy,frequency,velocity, ...
    frequency,theory_velocity,'ModeTolerance',12);
sidelobe_broken=dispersion_sidelobe_metrics( ...
    broken_energy,frequency,velocity,frequency,theory_velocity, ...
    'ModeTolerance',12);
verifyGreaterThan(testCase,sidelobe_good.explained_energy_fraction,0.95);
verifyLessThan(testCase,sidelobe_good.p90_outside_peak,0.05);
verifyGreaterThan(testCase,sidelobe_broken.p90_outside_peak,0.95);
end

function testSourceSeparatedNcfHandlesTinyDwmAmplitudes(testCase)
dt=.004;
time=(0:dt:5)';
receiver_x=0:10:200;
true_velocity=250;
source_count=8;
response=zeros(numel(time),numel(receiver_x),source_count);
for source_index=1:source_count
    main_frequency=5+source_index;
    if mod(source_index,2)==1
        arrival=.6+receiver_x/true_velocity;
    else
        arrival=.6+(max(receiver_x)-receiver_x)/true_velocity;
    end
    for receiver_index=1:numel(receiver_x)
        response(:,receiver_index,source_index)=1e-20*rickerkai( ...
            time-arrival(receiver_index),main_frequency);
    end
end
[gather,~,distance]=compute_source_separated_ncf_gather( ...
    response,receiver_x,dt,'MaxLag',2,'DistanceBin',10, ...
    'SpectrumNormalization','phase','WaterLevel',0);
verifyGreaterThan(testCase,max(abs(gather),[],'all'),0);
velocity=(180:2:320)';
[energy,frequency]=surface_wave_dispersion(gather,distance,dt,velocity, ...
    15,'Method','phase','FrequencyMin',4,'ZeroPadding',4);
picks=pick_dispersion_ridge(energy,frequency,velocity, ...
    'FrequencyRange',[6 13],'VelocityRange',[200 300], ...
    'Smoothness',.2,'MaxJump',20);
verifyLessThanOrEqual(testCase,abs( ...
    median(picks.phase_velocity_mps)-true_velocity),4);
end

function model=reference_model
model.vs=[220 440]; model.vp=[380 760];
model.dns=[2000 2000]; model.thk=[10 inf];
model.Damp_s=[0 0]; model.Damp_p=[0 0];
end
