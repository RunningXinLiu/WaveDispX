function metrics=dispersion_sidelobe_metrics(energy,frequency,velocity, ...
    theory_frequency,theory_modes,varargin)
%DISPERSION_SIDELOBE_METRICS Quantify energy outside theoretical modes.

p=inputParser;
addParameter(p,'FrequencyRange',[-inf inf], ...
    @(x) isnumeric(x) && numel(x)==2 && x(1)<x(2));
addParameter(p,'ModeTolerance',15, ...
    @(x) isscalar(x) && isfinite(x) && x>0);
parse(p,varargin{:});
opts=p.Results;

frequency=frequency(:)';
velocity=velocity(:);
theory_frequency=theory_frequency(:);
if isvector(theory_modes)
    theory_modes=theory_modes(:);
end
if size(theory_modes,1)~=numel(theory_frequency)
    error('dispersion_sidelobe_metrics:TheorySizeMismatch', ...
        'Theory modes must have one row per theory frequency.');
end
selected=frequency>=opts.FrequencyRange(1) & ...
    frequency<=opts.FrequencyRange(2);
selected_frequency=frequency(selected);
normalized=energy(:,selected);
normalized=normalized./max(max(normalized,[],1),eps);
interpolated_modes=interp1(theory_frequency,theory_modes, ...
    selected_frequency,'linear',nan);
if size(theory_modes,2)==1
    interpolated_modes=interpolated_modes(:);
elseif size(interpolated_modes,1)~=numel(selected_frequency) && ...
        size(interpolated_modes,2)==numel(selected_frequency)
    interpolated_modes=interpolated_modes.';
end

explained=false(size(normalized));
for frequency_index=1:numel(selected_frequency)
    for mode_index=1:size(interpolated_modes,2)
        mode_velocity=interpolated_modes(frequency_index,mode_index);
        if isfinite(mode_velocity)
            explained(:,frequency_index)= ...
                explained(:,frequency_index) | ...
                abs(velocity-mode_velocity)<=opts.ModeTolerance;
        end
    end
end
outside=normalized;
outside(explained)=0;
outside_peak=max(outside,[],1);

metrics=struct;
metrics.frequency_hz=selected_frequency;
metrics.explained_energy_fraction=sum(normalized(explained))/ ...
    max(sum(normalized,'all'),eps);
metrics.unexplained_hot_pixel_fraction=mean( ...
    normalized(~explained)>=0.5);
metrics.median_outside_peak=median(outside_peak,'omitnan');
metrics.p90_outside_peak=prctile(outside_peak,90);
metrics.mode_mask=explained;
metrics.settings=opts;
end
