function plot_ncf(ncf_folder, varargin)
% PLOT_NCF Plot NCF files from folder
%
% Usage:
%   plot_ncf(ncf_folder)
%   plot_ncf(ncf_folder, 'Component', 'z', 'SortBy', 'distance')
%
% Optional Parameters:
%   'Component' - 'z', 'x', 'y' (default: 'z')
%   'SortBy'    - 'distance', 'pair', 'none' (default: 'distance')
%   'Normalize' - true/false, normalize each trace for display (default: true)

p = inputParser;
addParameter(p, 'Component', 'z');
addParameter(p, 'SortBy', 'distance');
addParameter(p, 'Normalize', true);
parse(p, varargin{:});

opts = p.Results;

% Find NCF files
pattern = fullfile(ncf_folder, sprintf('ncf_%s_*.dat', opts.Component));
files = dir(pattern);

if isempty(files)
    error('No NCF files found matching pattern: %s', pattern);
end

% Read all NCF files
n_files = length(files);
ncf_data = cell(n_files, 1);
distances = zeros(n_files, 1);

for i = 1:n_files
    filepath = fullfile(ncf_folder, files(i).name);
    [~, ~, ~, ~, ~, ~, lag_time, ncf_pos, ncf_neg] = read_ncf_file(filepath);
    
    ncf_data{i}.lag_time = lag_time;
    ncf_data{i}.ncf_pos = ncf_pos;
    ncf_data{i}.ncf_neg = ncf_neg;
    ncf_data{i}.filename = files(i).name;
    
    % Calculate distance from filename or coordinates
    % Parse station numbers from filename
    tokens = regexp(files(i).name, 'ncf_\w+_(\d+)_(\d+)\.dat', 'tokens');
    if ~isempty(tokens)
        ncf_data{i}.sta1 = str2double(tokens{1}{1});
        ncf_data{i}.sta2 = str2double(tokens{1}{2});
    end
    
    % Read coordinates to compute distance
    [lon1, lat1, ~, lon2, lat2, ~, ~, ~, ~] = read_ncf_file(filepath);
    distances(i) = compute_distance(lat1, lon1, lat2, lon2);
    ncf_data{i}.distance = distances(i);
end

% Sort by distance if requested
if strcmpi(opts.SortBy, 'distance')
    [~, sort_idx] = sort(distances);
    ncf_data = ncf_data(sort_idx);
    distances = distances(sort_idx);
end

% Create figure
figure('Position', [100, 100, 1200, 800]);

% Get max lag time for x-axis
max_lag = max(cellfun(@(x) max(x.lag_time), ncf_data));

% Plot each NCF
hold on;
for i = 1:n_files
    lag = ncf_data{i}.lag_time;
    ncf_sym = [flipud(ncf_data{i}.ncf_neg(2:end)); ncf_data{i}.ncf_pos];
    t_sym = [-flipud(lag(2:end)); lag];
    
    % Normalize for display
    if opts.Normalize
        max_amp = max(abs(ncf_sym));
        if max_amp > 0
            ncf_sym = ncf_sym / max_amp;
        end
    end
    
    % Plot with offset
    offset = distances(i);
    plot(t_sym, ncf_sym * (distances(end) - distances(1)) * 0.05 + offset, 'k', 'LineWidth', 0.5);
end
hold off;

xlabel('Lag Time (s)');
ylabel('Inter-station Distance (m)');
title(sprintf('NCF - %s Component', upper(opts.Component)));
xlim([-max_lag, max_lag]);
grid on;

end

%% Helper function to compute distance
function dist = compute_distance(lat1, lon1, lat2, lon2)
    % Haversine formula for distance in meters
    R = 6371000; % Earth radius in meters
    
    dlat = deg2rad(lat2 - lat1);
    dlon = deg2rad(lon2 - lon1);
    
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    
    dist = R * c;
end