function metrics = dispersion_image_quality_metrics(energy, frequency, ...
    velocity, theory_frequency, theory_velocity, varargin)
%DISPERSION_IMAGE_QUALITY_METRICS Measure ridge sharpness and uniqueness.
%
% Unlike a picker-error metric, this function penalizes broad energy,
% competing off-ridge maxima, and frequency columns without a unique mode.

p = inputParser;
addParameter(p,'FrequencyRange',[-inf inf], ...
    @(x) isnumeric(x) && numel(x)==2 && x(1)<x(2));
addParameter(p,'NearTolerance',12, ...
    @(x) isscalar(x) && isfinite(x) && x>0);
addParameter(p,'FarGuard',30, ...
    @(x) isscalar(x) && isfinite(x) && x>0);
parse(p,varargin{:});
opts=p.Results;

frequency=frequency(:)';
velocity=velocity(:);
theory_frequency=theory_frequency(:);
theory_velocity=theory_velocity(:);
validateattributes(energy,{'numeric'},{'2d','nonempty','finite'});
if size(energy,1)~=numel(velocity) || size(energy,2)~=numel(frequency)
    error('dispersion_image_quality_metrics:SizeMismatch', ...
        'Energy must be velocity-by-frequency.');
end

reference_velocity=interp1(theory_frequency,theory_velocity,frequency, ...
    'linear',nan);
selected=frequency>=opts.FrequencyRange(1) & ...
    frequency<=opts.FrequencyRange(2) & isfinite(reference_velocity);
selected_index=find(selected);
normalized=energy(:,selected_index);
normalized=normalized./max(max(normalized,[],1),eps);
reference_velocity=reference_velocity(selected_index);

n_frequency=numel(selected_index);
concentration=nan(1,n_frequency);
half_power_width=nan(1,n_frequency);
dominance_margin=nan(1,n_frequency);
dominant_near_reference=false(1,n_frequency);
velocity_increment=median(diff(velocity));
for frequency_index=1:n_frequency
    near=abs(velocity-reference_velocity(frequency_index))<= ...
        opts.NearTolerance;
    far=abs(velocity-reference_velocity(frequency_index))>=opts.FarGuard;
    column=normalized(:,frequency_index);
    concentration(frequency_index)=sum(column(near))/max(sum(column),eps);
    half_power_width(frequency_index)= ...
        sum(column>=0.5)*velocity_increment;
    dominance_margin(frequency_index)=max(column(near))-max(column(far));
    [~,peak_index]=max(column);
    dominant_near_reference(frequency_index)= ...
        abs(velocity(peak_index)-reference_velocity(frequency_index))<= ...
        opts.NearTolerance;
end

metrics=struct;
metrics.frequency_hz=frequency(selected_index);
metrics.reference_velocity_mps=reference_velocity;
metrics.ridge_concentration=concentration;
metrics.half_power_width_mps=half_power_width;
metrics.dominance_margin=dominance_margin;
metrics.dominant_ridge_fraction=mean(dominant_near_reference);
metrics.median_ridge_concentration=median(concentration,'omitnan');
metrics.median_half_power_width_mps=median(half_power_width,'omitnan');
metrics.p90_half_power_width_mps=prctile(half_power_width,90);
metrics.p10_dominance_margin=prctile(dominance_margin,10);
metrics.ambiguous_frequency_fraction=mean(dominance_margin<0.25);
metrics.clean_ridge_pass=metrics.dominant_ridge_fraction>=0.95 && ...
    metrics.median_ridge_concentration>=0.32 && ...
    metrics.median_half_power_width_mps<=30 && ...
    metrics.p10_dominance_margin>=0;
metrics.settings=opts;
end
