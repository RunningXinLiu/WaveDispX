function plot_layer_model(velocity_axes, density_axes, model, varargin)
%PLOT_LAYER_MODEL Plot stepwise Vp, Vs, and density profiles.

p = inputParser;
addParameter(p, 'DepthMax', [], ...
    @(x) isempty(x) || (isscalar(x) && x > 0));
parse(p, varargin{:});

finite_thickness = model.thk(isfinite(model.thk));
if isempty(p.Results.DepthMax)
    if isempty(finite_thickness)
        depth_max = 50;
    else
        depth_max = max(40, 3*sum(finite_thickness));
    end
else
    depth_max = p.Results.DepthMax;
end

[vp_step, depth_step] = layer_steps(model.vp, model.thk, depth_max);
vs_step = layer_steps(model.vs, model.thk, depth_max);
rho_step = layer_steps(model.dns, model.thk, depth_max);

plot(velocity_axes, vs_step, depth_step, 'b-', 'LineWidth', 2, ...
    'DisplayName', 'V_s');
hold(velocity_axes, 'on');
plot(velocity_axes, vp_step, depth_step, 'r-', 'LineWidth', 2, ...
    'DisplayName', 'V_p');
set(velocity_axes, 'YDir', 'reverse');
ylim(velocity_axes, [0 depth_max]);
xlabel(velocity_axes, 'Velocity (m/s)');
ylabel(velocity_axes, 'Depth (m)');
title(velocity_axes, 'Layer velocity model');
legend(velocity_axes, 'Location', 'southeast');
grid(velocity_axes, 'on');

plot(density_axes, rho_step, depth_step, 'k-', 'LineWidth', 2);
set(density_axes, 'YDir', 'reverse');
ylim(density_axes, [0 depth_max]);
xlabel(density_axes, 'Density (kg/m^3)');
ylabel(density_axes, 'Depth (m)');
title(density_axes, 'Density model');
grid(density_axes, 'on');
end

function [value_step, depth_step] = layer_steps(values, thickness, depth_max)
values = values(:)';
thickness = thickness(:)';
if numel(values) ~= numel(thickness)
    error('plot_layer_model:LayerCountMismatch', ...
        'Property and thickness arrays must have equal lengths.');
end
finite_thickness = thickness(1:end-1);
finite_thickness(~isfinite(finite_thickness)) = 0;
edges = [0, cumsum(finite_thickness), depth_max];
edges = edges(1:numel(values)+1);
edges(end) = depth_max;
value_step = repelem(values,2).';
depth_step = [edges(1:end-1); edges(2:end)];
depth_step = depth_step(:);
end
