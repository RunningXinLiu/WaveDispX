function [gather, lag_time, distances, counts, metadata] = ...
    assemble_ncf_gather(ncf_folder, varargin)
%ASSEMBLE_NCF_GATHER Bin station-pair NCFs into a time-distance gather.
%
% The default symmetric branch averages positive and time-reversed negative
% lag.  All station pairs at the same binned separation are then stacked.

p = inputParser;
addParameter(p, 'Component', 'z', @(x) ischar(x) || isstring(x));
addParameter(p, 'Branch', 'symmetric', @(x) ischar(x) || isstring(x));
addParameter(p, 'DistanceBin', 5, @(x) isscalar(x) && x > 0);
addParameter(p, 'VirtualSource', [], ...
    @(x) isempty(x) || (isscalar(x) && x >= 1 && fix(x)==x));
addParameter(p, 'MinimumDistance', 0, @(x) isscalar(x) && x >= 0);
parse(p, varargin{:});
opts = p.Results;

component = lower(char(opts.Component));
branch = lower(string(opts.Branch));
if ~any(branch == ["symmetric", "positive", "negative"])
    error('assemble_ncf_gather:UnknownBranch', ...
        'Branch must be symmetric, positive, or negative.');
end

files = dir(fullfile(ncf_folder, sprintf('ncf_%s_*.dat', component)));
if isempty(files)
    error('assemble_ncf_gather:NoFiles', ...
        'No %s-component NCF files found in %s.', component, ncf_folder);
end

records = struct('distance', {}, 'trace', {}, 'pair', {});
lag_time = [];
for file_index = 1:numel(files)
    tokens = regexp(files(file_index).name, ...
        'ncf_\w+_(\d+)_(\d+)\.dat', 'tokens', 'once');
    if isempty(tokens)
        continue;
    end
    pair = [str2double(tokens{1}), str2double(tokens{2})];
    if ~isempty(opts.VirtualSource) && ~any(pair == opts.VirtualSource)
        continue;
    end

    filename = fullfile(ncf_folder, files(file_index).name);
    [lon1, lat1, ~, lon2, lat2, ~, lag, pos, neg] = read_ncf_file(filename);
    if isempty(lag_time)
        lag_time = lag(:);
    elseif numel(lag) ~= numel(lag_time) || ...
            max(abs(lag(:)-lag_time)) > 10*eps(max(lag_time))
        error('assemble_ncf_gather:LagMismatch', ...
            'NCF files do not share an identical lag-time axis.');
    end

    switch branch
        case "symmetric"
            trace = 0.5*(pos(:)+neg(:));
        case "positive"
            trace = pos(:);
        case "negative"
            trace = neg(:);
    end

    distance = haversine_distance(lat1, lon1, lat2, lon2);
    if distance < opts.MinimumDistance
        continue;
    end
    records(end+1).distance = distance; %#ok<AGROW>
    records(end).trace = trace;
    records(end).pair = pair;
end

if isempty(records)
    error('assemble_ncf_gather:NoSelectedPairs', ...
        'No station pairs satisfy the requested selection.');
end

raw_distance = [records.distance];
distance_label = round(raw_distance/opts.DistanceBin)*opts.DistanceBin;
distances = unique(distance_label);
distances = distances(distances > 0);
gather = zeros(numel(lag_time), numel(distances));
counts = zeros(1, numel(distances));

for distance_index = 1:numel(distances)
    selected = find(distance_label == distances(distance_index));
    traces_at_distance = cat(2, records(selected).trace);
    gather(:,distance_index) = mean(traces_at_distance, 2, 'omitnan');
    counts(distance_index) = numel(selected);
end

metadata = struct;
metadata.component = component;
metadata.branch = char(branch);
metadata.distance_bin = opts.DistanceBin;
metadata.virtual_source = opts.VirtualSource;
metadata.raw_pair_count = numel(records);
metadata.pairs = vertcat(records.pair);
end

function distance = haversine_distance(lat1, lon1, lat2, lon2)
earth_radius = 6371000;
dlat = deg2rad(lat2-lat1);
dlon = deg2rad(lon2-lon1);
a = sin(dlat/2)^2 + cosd(lat1)*cosd(lat2)*sin(dlon/2)^2;
distance = earth_radius*2*atan2(sqrt(a), sqrt(max(0,1-a)));
end
