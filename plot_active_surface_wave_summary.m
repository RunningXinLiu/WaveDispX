function fig = plot_active_surface_wave_summary(model, source, time, gather, ...
    energy, frequency, velocity, theory, picks, wave_name)
%PLOT_ACTIVE_SURFACE_WAVE_SUMMARY Active geometry, model, data, and dispersion.

fig=figure('Position',[60 60 1600 900]);
layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');

velocity_axes=nexttile(layout,1);
density_axes=nexttile(layout,2);
plot_layer_model(velocity_axes,density_axes,model);

geometry_axes=nexttile(layout,3);
plot_source_receiver_geometry(geometry_axes,0,0,source.offset, ...
    zeros(size(source.offset)),'Title','Active source-receiver geometry');

gather_axes=nexttile(layout,4);
column_scale=max(abs(gather),[],1);
column_scale(column_scale==0)=1;
normalized_gather=gather./column_scale;
imagesc(gather_axes,source.offset,time,normalized_gather);
axis(gather_axes,'xy');
xlabel(gather_axes,'Offset (m)');
ylabel(gather_axes,'Time (s)');
title(gather_axes,sprintf('%s shot gather (trace-normalized)',wave_name));
colormap(gather_axes,gray);
colorbar(gather_axes);

dispersion_axes=nexttile(layout,5,[1 2]);
imagesc(dispersion_axes,frequency,velocity,energy);
axis(dispersion_axes,'xy');
hold(dispersion_axes,'on');
plot(dispersion_axes,theory(:,1),theory(:,2:end),'w.','MarkerSize',5);
if ~isempty(picks)
    plot(dispersion_axes,picks.frequency_hz,picks.phase_velocity_mps, ...
        'k-','LineWidth',1.3);
end
xlabel(dispersion_axes,'Frequency (Hz)');
ylabel(dispersion_axes,'Phase velocity (m/s)');
title(dispersion_axes,sprintf( ...
    '%s dispersion: theory (white), pick (black)',wave_name));
colormap(dispersion_axes,turbo);
colorbar(dispersion_axes);
clim(dispersion_axes,[0 1]);
sgtitle(layout,sprintf('Active %s-wave modelling and dispersion QC',wave_name));
end
