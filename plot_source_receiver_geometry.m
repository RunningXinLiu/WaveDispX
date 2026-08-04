function plot_source_receiver_geometry(ax, source_x, source_y, ...
    receiver_x, receiver_y, varargin)
%PLOT_SOURCE_RECEIVER_GEOMETRY Plot source and receiver acquisition layout.

p = inputParser;
addParameter(p, 'Title', 'Source-receiver geometry', ...
    @(x) ischar(x) || isstring(x));
parse(p, varargin{:});

source_x = source_x(:);
source_y = source_y(:);
receiver_x = receiver_x(:);
receiver_y = receiver_y(:);
if numel(source_x) ~= numel(source_y) || ...
        numel(receiver_x) ~= numel(receiver_y)
    error('plot_source_receiver_geometry:SizeMismatch', ...
        'Each coordinate pair must have equal lengths.');
end

scatter(ax, source_x, source_y, 36, [0.85 0.15 0.12], 'filled', ...
    'Marker', 'p', 'DisplayName', sprintf('Sources (n=%d)',numel(source_x)));
hold(ax, 'on');
scatter(ax, receiver_x, receiver_y, 24, [0.05 0.35 0.85], 'filled', ...
    'Marker', 'v', 'DisplayName', sprintf('Receivers (n=%d)',numel(receiver_x)));
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x (m)');
ylabel(ax, 'y (m)');
title(ax, char(p.Results.Title));
legend(ax, 'Location', 'best');

all_x = [source_x; receiver_x];
all_y = [source_y; receiver_y];
x_span = max(all_x)-min(all_x);
y_span = max(all_y)-min(all_y);
if y_span < 0.05*max(x_span,1)
    x_mid = 0.5*(max(all_x)+min(all_x));
    y_mid = 0.5*(max(all_y)+min(all_y));
    xlim(ax, x_mid+[-1 1]*0.55*max(x_span,1));
    ylim(ax, y_mid+[-1 1]*0.35*max(x_span,1));
end
end
