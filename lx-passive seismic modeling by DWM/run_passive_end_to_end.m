function results = run_passive_end_to_end(varargin)
%RUN_PASSIVE_END_TO_END Passive simulation through theory/picking validation.
%
% results = run_passive_end_to_end('Quick',true) runs a small regression
% model.  The default full configuration uses 384 stratified annular source
% locations, repeated independent events, a 600 m receiver aperture, and a
% 60 s record.

p = inputParser;
addParameter(p, 'Quick', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Plot', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'OutputFolder', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'Seed', 42, @(x) isscalar(x) && isfinite(x));
addParameter(p, 'SourceCount', [], ...
    @(x) isempty(x) || (isscalar(x) && x>=1 && fix(x)==x));
addParameter(p, 'EventsPerSource', [], ...
    @(x) isempty(x) || (isscalar(x) && x>=1 && fix(x)==x));
addParameter(p, 'AngularSampling', 'stratified', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'SourceMainFrequency', 12, ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p, 'SourceInnerRadius', [], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>=0));
addParameter(p, 'SourceOuterRadius', [], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p, 'ReceiverHalfLength', [], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p, 'ReceiverSpacing', [], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p, 'ComputeMixedNcf', false, ...
    @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'DistanceTaperFraction', 0.5, ...
    @(x) isscalar(x) && isfinite(x) && x>=0 && x<=1);
addParameter(p, 'CorrelationMode', 'mixed', ...
    @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opts = p.Results;
correlation_mode=lower(string(opts.CorrelationMode));
if ~any(correlation_mode==["mixed","source-separated"])
    error('run_passive_end_to_end:UnknownCorrelationMode', ...
        'CorrelationMode must be mixed or source-separated.');
end

script_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(script_dir);
addpath(repository_root, script_dir);
rng(opts.Seed, 'twister');

if strlength(string(opts.OutputFolder)) == 0
    run_id = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
    output_folder = fullfile(script_dir, 'output', ['passive_',run_id]);
else
    output_folder = char(opts.OutputFolder);
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
ncf_folder = fullfile(output_folder, 'ncf');

% Elastic two-layer reference model.
model.vs = [220 440];
model.vp = [380 760];
model.dns = [2000 2000];
model.thk = [10 inf];
model.Damp_s = [0 0];
model.Damp_p = [0 0];

if logical(opts.Quick)
    source.number = 8;
    source.recordlength = 8;
    source.events_per_source = 1;
    source.dt = 0.01;
    source.maxfre = 12;
    source.fmin = 3;
    source.fmax = 10;
    source_inner_radius = 250;
    source_outer_radius = 350;
    receiver_half_length = 100;
    receiver_spacing = 5;
    segment_length = 4;
else
    source.number = 384;
    source.recordlength = 60;
    source.events_per_source = 24;
    source.dt = 0.005;
    source.maxfre = 20;
    source.fmin = 3;
    source.fmax = 18;
    source_inner_radius = 800;
    source_outer_radius = 1200;
    receiver_half_length = 300;
    receiver_spacing = 5;
    segment_length = 10;
end

if ~isempty(opts.SourceCount)
    source.number=opts.SourceCount;
end
if ~isempty(opts.EventsPerSource)
    source.events_per_source=opts.EventsPerSource;
end
if ~isempty(opts.SourceMainFrequency)
    source.fm=opts.SourceMainFrequency;
end
if ~isempty(opts.SourceInnerRadius)
    source_inner_radius=opts.SourceInnerRadius;
end
if ~isempty(opts.SourceOuterRadius)
    source_outer_radius=opts.SourceOuterRadius;
end
if source_outer_radius<=source_inner_radius
    error('run_passive_end_to_end:InvalidSourceAnnulus', ...
        'SourceOuterRadius must exceed SourceInnerRadius.');
end
if ~isempty(opts.ReceiverHalfLength)
    receiver_half_length=opts.ReceiverHalfLength;
end
if ~isempty(opts.ReceiverSpacing)
    receiver_spacing=opts.ReceiverSpacing;
end
geometry_options = {'InnerRadius',source_inner_radius, ...
    'OuterRadius',source_outer_radius, ...
    'ReceiverHalfLength',receiver_half_length, ...
    'ReceiverSpacing',receiver_spacing};

source.Fx = zeros(1,source.number);
source.Fy = zeros(1,source.number);
source.Fz = sign(randn(1,source.number)) .* ...
    (0.5+0.5*rand(1,source.number));
flag = 'v';
SR = S_R_geometry(source.number, 'Distribution','annulus', ...
    'AngularSampling',opts.AngularSampling,geometry_options{:});

fprintf('\n=== Passive forward modelling ===\n');
forward_start = tic;
[traces, TT] = PassiveFW(model, source, SR, flag);
forward_seconds = toc(forward_start);

% Use the full aperture and a conservative Rayleigh lower velocity.
array_aperture = max(SR.rx)-min(SR.rx);
minimum_phase_velocity = 0.90*min(model.vs);
finite_record_max_lag = 1.3*array_aperture/minimum_phase_velocity;
if finite_record_max_lag >= segment_length
    error('run_passive_end_to_end:WindowTooShort', ...
        'SegmentLength must exceed MaxLag.');
end

fprintf('\n=== Noise correlations ===\n');
ncf_start = tic;
if logical(opts.ComputeMixedNcf)
    ncf_summary = compute_ncf(traces, SR, source, ncf_folder, ...
        'Component','z', 'SegmentLength',segment_length, 'Overlap',0.5, ...
        'MaxLag',finite_record_max_lag, 'Normalize','spectral-whitening');
    [mixed_ncf_gather, mixed_lag_time, mixed_distances, ...
        mixed_pair_counts, mixed_gather_metadata] = ...
        assemble_ncf_gather(ncf_folder, 'Component','z', ...
        'Branch','symmetric', 'DistanceBin',SR.receiver_spacing, ...
        'MinimumDistance',SR.receiver_spacing);
else
    ncf_summary = struct('skipped',true, ...
        'reason','ComputeMixedNcf=false');
    mixed_ncf_gather = [];
    mixed_lag_time = [];
    mixed_distances = [];
    mixed_pair_counts = [];
    mixed_gather_metadata = struct('skipped',true);
end

switch correlation_mode
    case "mixed"
        % Field-style default: all unknown sources are already mixed in the
        % finite record.  Window, whiten, correlate receiver pairs, bin by
        % distance, and stack without using simulated source identities.
        ensemble_max_lag=finite_record_max_lag;
        [ncf_gather,lag_time,distances,pair_counts]= ...
            compute_ncf_gather_from_traces(traces.z,SR.rx,source.dt, ...
            'SegmentLength',segment_length,'Overlap',0.5, ...
            'MaxLag',ensemble_max_lag, ...
            'Normalize','spectral-whitening','Branch','symmetric', ...
            'DistanceBin',SR.receiver_spacing,'Algorithm','auto');
        ensemble_cross_spectrum=[];
        correlation_mode_label='mixed finite-record';
    otherwise
        % Controlled synthetic reference: correlate every known source
        % separately before source stacking.  This is an ideal ensemble
        % expectation and is intentionally not the default field workflow.
        ensemble_max_lag=min(max(2,finite_record_max_lag),TT.time(end));
        [ncf_gather,lag_time,distances,pair_counts, ...
            ensemble_cross_spectrum]= ...
            compute_source_separated_ncf_gather( ...
            TT.z,SR.rx,source.dt,'MaxLag',ensemble_max_lag, ...
            'Branch','symmetric','DistanceBin',SR.receiver_spacing, ...
            'SpectrumNormalization','phase','WaterLevel',0);
        correlation_mode_label='source-separated ensemble';
end
gather_metadata=struct('component','z','branch','symmetric', ...
    'distance_bin',SR.receiver_spacing, ...
    'correlation_mode',correlation_mode_label, ...
    'source_count',source.number, ...
    'segment_length',segment_length);
ncf_seconds = toc(ncf_start);

fprintf('\n=== Dispersion image and ridge picking ===\n');
velocity = (150:2:500)';
spatial_alias_limit = minimum_phase_velocity/(2*SR.receiver_spacing);
dispersion_fmax = min(source.maxfre, 0.95*spatial_alias_limit);
[energy, frequency] = surface_wave_dispersion( ...
    ncf_gather, distances, source.dt, velocity, dispersion_fmax, ...
    'Method','phase','Component','vertical','FrequencyMin',1, ...
    'ZeroPadding',8,'DistanceTaperFraction', ...
    opts.DistanceTaperFraction,'Plot',false);

% Require about three wavelengths across the aperture for a narrow, unique
% ridge.  Lower-frequency energy remains saved in the MAT result but is not
% presented as quantitatively resolved dispersion.
aperture_frequency_floor = 3*max(model.vs)/array_aperture;
frequency_range = [max(source.fmin,aperture_frequency_floor), ...
    dispersion_fmax];
picks = pick_dispersion_ridge(energy, frequency, velocity, ...
    'FrequencyRange',frequency_range, 'VelocityRange',[170 460], ...
    'Smoothness',0.35, 'MaxJump',30, 'MinimumEnergy',0.05);

fprintf('\n=== Independent theoretical dispersion ===\n');
theory = compute_theoretical_dispersion(model, frequency, ...
    'RayleighModes',4,'LoveModes',4);
continuity_metrics = dispersion_continuity_metrics(energy,frequency,velocity, ...
    theory.frequency,theory.rayleigh(:,1), ...
    'FrequencyRange',frequency_range);
image_quality_metrics = dispersion_image_quality_metrics( ...
    energy,frequency,velocity,theory.frequency,theory.rayleigh(:,1), ...
    'FrequencyRange',frequency_range);

results = struct;
results.model = model;
results.source = source;
results.geometry = SR;
results.traces = traces;
results.single_source = TT;
results.ncf_summary = ncf_summary;
results.ncf_gather = ncf_gather;
results.lag_time = lag_time;
results.distances = distances;
results.pair_counts = pair_counts;
results.gather_metadata = gather_metadata;
results.ensemble_cross_spectrum = ensemble_cross_spectrum;
results.mixed_ncf_gather = mixed_ncf_gather;
results.mixed_lag_time = mixed_lag_time;
results.mixed_distances = mixed_distances;
results.mixed_pair_counts = mixed_pair_counts;
results.mixed_gather_metadata = mixed_gather_metadata;
results.energy = energy;
results.frequency = frequency;
results.velocity = velocity;
results.picks = picks;
results.theory = theory;
results.dispersion_method = 'phase';
results.distance_taper_fraction = opts.DistanceTaperFraction;
results.continuity_metrics = continuity_metrics;
results.image_quality_metrics = image_quality_metrics;
results.correlation_mode = correlation_mode_label;
results.max_lag = ensemble_max_lag;
results.finite_record_max_lag = finite_record_max_lag;
results.spatial_alias_limit_hz = spatial_alias_limit;
results.aperture_frequency_floor_hz = aperture_frequency_floor;
results.reliable_frequency_range_hz = frequency_range;
results.forward_seconds = forward_seconds;
results.ncf_seconds = ncf_seconds;
results.output_folder = output_folder;

save(fullfile(output_folder,'passive_end_to_end_results.mat'), ...
    'results','-v7.3');
writetable(picks, fullfile(output_folder,'picked_rayleigh.csv'));

if logical(opts.Plot)
    summary_figure=plot_passive_surface_wave_summary(results);
    exportgraphics(summary_figure, ...
        fullfile(output_folder,'passive_end_to_end.png'), ...
        'Resolution',180);
end

fprintf('\nCompleted. Results: %s\n', output_folder);
end
