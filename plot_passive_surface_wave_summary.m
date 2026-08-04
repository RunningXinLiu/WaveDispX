function summary_figure = plot_passive_surface_wave_summary(results)
%PLOT_PASSIVE_SURFACE_WAVE_SUMMARY Model, geometry, NCF, and dispersion QC.

model=results.model;
source=results.source;
SR=results.geometry;
distances=results.distances;
ncf_gather=results.ncf_gather;
lag_time=results.lag_time;
energy=results.energy;
frequency=results.frequency;
velocity=results.velocity;
theory=results.theory;
picks=results.picks;

summary_figure=figure('Position',[60 60 1600 900]);
layout=tiledlayout(summary_figure,2,3,'TileSpacing','compact', ...
    'Padding','compact');
velocity_axes=nexttile(layout,1);
density_axes=nexttile(layout,2);
plot_layer_model(velocity_axes,density_axes,model);

geometry_axes=nexttile(layout,3);
plot_source_receiver_geometry(geometry_axes,SR.sx,SR.sy,SR.rx,SR.ry, ...
    'Title','Passive source-receiver geometry');

gather_axes=nexttile(layout,4);
scale=max(distances)*0.025;
hold(gather_axes,'on');
for trace_index=1:numel(distances)
    trace=ncf_gather(:,trace_index);
    trace_amplitude=max(abs(trace));
    if trace_amplitude>0
        trace=trace/trace_amplitude;
    end
    plot(gather_axes,lag_time,distances(trace_index)+scale*trace,'k');
end
xlabel(gather_axes,'Lag time (s)');
ylabel(gather_axes,'Interstation distance (m)');
if isfield(results,'correlation_mode') && ...
        strcmpi(results.correlation_mode,'mixed finite-record')
    title(gather_axes,'Direct mixed-record symmetric NCF gather');
else
    title(gather_axes,'Source-ensemble symmetric NCF gather');
end
grid(gather_axes,'on');

dispersion_axes=nexttile(layout,5,[1 2]);
imagesc(dispersion_axes,frequency,velocity,energy);
axis(dispersion_axes,'xy');
hold(dispersion_axes,'on');
plot(dispersion_axes,theory.frequency,theory.rayleigh(:,2:end), ...
    'w:','LineWidth',1.0);
plot(dispersion_axes,theory.frequency,theory.rayleigh(:,1), ...
    'w--','LineWidth',1.5);
if ~isempty(picks)
    plot(dispersion_axes,picks.frequency_hz,picks.phase_velocity_mps, ...
        'k-','LineWidth',1.5);
end
xlabel(dispersion_axes,'Frequency (Hz)');
ylabel(dispersion_axes,'Phase velocity (m/s)');
method_label=ucfirst(results.dispersion_method);
if isfield(results,'distance_taper_fraction')
    method_label=sprintf('%s, Tukey %.2f',method_label, ...
        results.distance_taper_fraction);
end
title(dispersion_axes,sprintf( ...
    ['%s energy: theory (white), pick (black), support %.1f%%, ' ...
     'width %.0f m/s'], ...
    method_label, ...
    100*results.image_quality_metrics.dominant_ridge_fraction, ...
    results.image_quality_metrics.median_half_power_width_mps));
colormap(dispersion_axes,turbo);
colorbar(dispersion_axes);
clim(dispersion_axes,[0 1]);
xlim(dispersion_axes,results.reliable_frequency_range_hz);
if isfield(results,'correlation_mode') && ...
        strcmpi(results.correlation_mode,'mixed finite-record')
    header=sprintf([ ...
        'Passive surface waves | directly correlated mixed record | %.0f s'], ...
        source.recordlength);
else
    header=sprintf([ ...
        'Passive surface waves | %d-source ensemble correlation | %.0f s finite record saved'], ...
        source.number,source.recordlength);
end
sgtitle(layout,header);
end

function value=ucfirst(value)
value=char(value);
value(1)=upper(value(1));
end
