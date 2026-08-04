function results=process_mixed_passive_record(data,model,source,geometry,varargin)
%PROCESS_MIXED_PASSIVE_RECORD Field-style NCF and dispersion processing.
%
% All sources must already be mixed in DATA.  No source identity is used.

p=inputParser;
addParameter(p,'SegmentLength',10, ...
    @(x) isscalar(x) && isfinite(x) && x>0);
addParameter(p,'Overlap',0.5, ...
    @(x) isscalar(x) && isfinite(x) && x>=0 && x<1);
addParameter(p,'MaxLag',[], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p,'Normalize','spectral-whitening', ...
    @(x) ischar(x) || isstring(x));
addParameter(p,'DistanceTaperFraction',0.5, ...
    @(x) isscalar(x) && isfinite(x) && x>=0 && x<=1);
addParameter(p,'OutputFolder','',@(x) ischar(x) || isstring(x));
addParameter(p,'OutputTag','mixed_record',@(x) ischar(x) || isstring(x));
addParameter(p,'Plot',true,@(x) islogical(x) || isnumeric(x));
parse(p,varargin{:});
opts=p.Results;

validateattributes(data,{'numeric'},{'2d','nonempty','finite'});
dt=source.dt;
record_length=(size(data,1)-1)*dt;
source.recordlength=record_length;
array_aperture=max(geometry.rx)-min(geometry.rx);
minimum_phase_velocity=0.90*min(model.vs);
if isempty(opts.MaxLag)
    max_lag=1.3*array_aperture/minimum_phase_velocity;
else
    max_lag=opts.MaxLag;
end
if opts.SegmentLength<=max_lag
    error('process_mixed_passive_record:WindowTooShort', ...
        'SegmentLength must exceed MaxLag.');
end

[ncf_gather,lag_time,distances,pair_counts]= ...
    compute_ncf_gather_from_traces(data,geometry.rx,dt, ...
    'SegmentLength',opts.SegmentLength,'Overlap',opts.Overlap, ...
    'MaxLag',max_lag,'Normalize',opts.Normalize, ...
    'Branch','symmetric','DistanceBin',geometry.receiver_spacing, ...
    'Algorithm','auto');

velocity=(150:2:500)';
spatial_alias_limit=minimum_phase_velocity/(2*geometry.receiver_spacing);
dispersion_fmax=min(source.maxfre,0.95*spatial_alias_limit);
[energy,frequency]=surface_wave_dispersion( ...
    ncf_gather,distances,dt,velocity,dispersion_fmax, ...
    'Method','phase','Component','vertical','FrequencyMin',1, ...
    'ZeroPadding',8,'DistanceTaperFraction', ...
    opts.DistanceTaperFraction,'Plot',false);
aperture_frequency_floor=3*max(model.vs)/array_aperture;
frequency_range=[max(source.fmin,aperture_frequency_floor), ...
    dispersion_fmax];
picks=pick_dispersion_ridge(energy,frequency,velocity, ...
    'FrequencyRange',frequency_range,'VelocityRange',[170 460], ...
    'Smoothness',0.35,'MaxJump',30,'MinimumEnergy',0.05);
theory=compute_theoretical_dispersion(model,frequency, ...
    'RayleighModes',4,'LoveModes',4);
continuity_metrics=dispersion_continuity_metrics( ...
    energy,frequency,velocity,theory.frequency,theory.rayleigh(:,1), ...
    'FrequencyRange',frequency_range);
image_quality_metrics=dispersion_image_quality_metrics( ...
    energy,frequency,velocity,theory.frequency,theory.rayleigh(:,1), ...
    'FrequencyRange',frequency_range);
sidelobe_metrics=dispersion_sidelobe_metrics( ...
    energy,frequency,velocity,theory.frequency,theory.rayleigh, ...
    'FrequencyRange',frequency_range,'ModeTolerance',15);

results=struct;
results.model=model;
results.source=source;
results.geometry=geometry;
results.ncf_gather=ncf_gather;
results.lag_time=lag_time;
results.distances=distances;
results.pair_counts=pair_counts;
results.energy=energy;
results.frequency=frequency;
results.velocity=velocity;
results.picks=picks;
results.theory=theory;
results.dispersion_method='phase';
results.distance_taper_fraction=opts.DistanceTaperFraction;
results.correlation_mode='mixed finite-record';
results.max_lag=max_lag;
results.segment_length=opts.SegmentLength;
results.overlap=opts.Overlap;
results.normalization=char(opts.Normalize);
results.reliable_frequency_range_hz=frequency_range;
results.spatial_alias_limit_hz=spatial_alias_limit;
results.aperture_frequency_floor_hz=aperture_frequency_floor;
results.continuity_metrics=continuity_metrics;
results.image_quality_metrics=image_quality_metrics;
results.sidelobe_metrics=sidelobe_metrics;

if strlength(string(opts.OutputFolder))>0
    output_folder=char(opts.OutputFolder);
    if ~exist(output_folder,'dir')
        mkdir(output_folder);
    end
    output_tag=char(opts.OutputTag);
    save(fullfile(output_folder,[output_tag,'_results.mat']), ...
        'results','-v7.3');
    writetable(picks,fullfile(output_folder, ...
        [output_tag,'_picked_rayleigh.csv']));
    if logical(opts.Plot)
        summary_figure=plot_passive_surface_wave_summary(results);
        exportgraphics(summary_figure,fullfile(output_folder, ...
            [output_tag,'.png']),'Resolution',180);
    end
end
end
