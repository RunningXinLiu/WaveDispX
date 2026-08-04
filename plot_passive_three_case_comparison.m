function fig=plot_passive_three_case_comparison(model,geometry,cases, ...
    theory,velocity,frequency_range,figure_title)
%PLOT_PASSIVE_THREE_CASE_COMPARISON Model-aware three-panel diagnostic.

fig=figure('Position',[50 50 1700 950]);
layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
velocity_axes=nexttile(layout,1);
density_axes=nexttile(layout,2);
plot_layer_model(velocity_axes,density_axes,model);
geometry_axes=nexttile(layout,3);
plot_source_receiver_geometry(geometry_axes,geometry.sx,geometry.sy, ...
    geometry.rx,geometry.ry,'Title','Source-receiver geometry');

for case_index=1:3
    ax=nexttile(layout,3+case_index);
    imagesc(ax,cases(case_index).frequency,velocity,cases(case_index).energy);
    axis(ax,'xy');
    hold(ax,'on');
    plot(ax,theory.frequency,theory.rayleigh(:,2:end), ...
        'w:','LineWidth',1.0);
    plot(ax,theory.frequency,theory.rayleigh(:,1), ...
        'w--','LineWidth',1.5);
    xlim(ax,frequency_range);
    ylim(ax,[min(velocity) max(velocity)]);
    clim(ax,[0 1]);
    xlabel(ax,'Frequency (Hz)');
    ylabel(ax,'Phase velocity (m/s)');
    title(ax,cases(case_index).title);
    colormap(ax,turbo);
end
sgtitle(layout,figure_title);
end
