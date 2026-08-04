function summary = compute_ncf(traces, SR, source, output_folder, varargin)
%COMPUTE_NCF Compute and save noise correlation functions for station pairs.
%
% summary = compute_ncf(traces, SR, source, output_folder, Name, Value)
%
% Options:
%   SegmentLength - processing-window length [s], [] uses the full trace
%   Overlap       - window overlap fraction in [0,1), default 0.5
%   MaxLag        - maximum absolute lag [s], default half a window
%   Normalize     - none, one-bit, rms, running-abs-mean, spectral-whitening
%   Component     - x, y, z, or all
%   OriginLat/Lon - geographic origin used only in output headers

p = inputParser;
addParameter(p, 'SegmentLength', [], @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'Overlap', 0.5, @(x) isscalar(x) && x >= 0 && x < 1);
addParameter(p, 'MaxLag', [], @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'Normalize', 'none', @(x) ischar(x) || isstring(x));
addParameter(p, 'Component', 'z', @(x) ischar(x) || isstring(x));
addParameter(p, 'OriginLat', 30, @(x) isscalar(x) && isfinite(x));
addParameter(p, 'OriginLon', 100, @(x) isscalar(x) && isfinite(x));
parse(p, varargin{:});
opts = p.Results;

validateattributes(source.dt, {'numeric'}, {'scalar', 'positive', 'finite'});
rec_num = numel(SR.rx);
if numel(SR.ry) ~= rec_num
    error('compute_ncf:GeometryMismatch', 'SR.rx and SR.ry must have equal length.');
end

component = lower(string(opts.Component));
if component == "all"
    components = ["x", "y", "z"];
elseif any(component == ["x", "y", "z"])
    components = component;
else
    error('compute_ncf:UnknownComponent', 'Component must be x, y, z, or all.');
end

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

[sta_lat, sta_lon] = station_coordinates( ...
    SR, opts.OriginLat, opts.OriginLon);
pairs = nchoosek(1:rec_num, 2);

summary = struct;
summary.output_folder = output_folder;
summary.components = cellstr(components);
summary.station_pairs = pairs;
summary.file_count = 0;
summary.dt = source.dt;

for component_index = 1:numel(components)
    comp = components(component_index);
    field_name = char(comp);
    if ~isfield(traces, field_name)
        error('compute_ncf:MissingComponent', ...
            'traces.%s does not exist.', field_name);
    end
    data = traces.(field_name);
    if size(data, 2) ~= rec_num
        error('compute_ncf:TraceGeometryMismatch', ...
            'traces.%s has %d columns but geometry has %d receivers.', ...
            field_name, size(data,2), rec_num);
    end

    fprintf('Computing %s-component NCFs for %d station pairs...\n', ...
        upper(field_name), size(pairs,1));
    for pair_index = 1:size(pairs,1)
        i = pairs(pair_index,1);
        j = pairs(pair_index,2);
        [lag_time, ncf_pos, ncf_neg] = compute_pair_ncf( ...
            data(:,i), data(:,j), source.dt, ...
            opts.SegmentLength, opts.Overlap, opts.MaxLag, opts.Normalize);

        filename = fullfile(output_folder, ...
            sprintf('ncf_%s_%d_%d.dat', field_name, i, j));
        save_ncf_file(filename, ...
            sta_lon(i), sta_lat(i), 0, sta_lon(j), sta_lat(j), 0, ...
            lag_time, ncf_pos, ncf_neg);
        summary.file_count = summary.file_count + 1;
    end
end

fprintf('NCF computation complete: %d files in %s\n', ...
    summary.file_count, output_folder);
end

function [lat, lon] = station_coordinates(SR, origin_lat, origin_lon)
% Local Cartesian offsets [m] to geographic coordinates.
lat = origin_lat + SR.ry(:)'/111320;
lon = origin_lon + SR.rx(:)'/(111320*cosd(origin_lat));
end
