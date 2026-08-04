function [lon1, lat1, elev1, lon2, lat2, elev2, lag_time, ncf_pos, ncf_neg] = read_ncf_file(filename)
% READ_NCF_FILE Read NCF file in specified format
%
% Input:
%   filename - Path to NCF file
%
% Outputs:
%   lon1, lat1, elev1 - Coordinates of station 1
%   lon2, lat2, elev2 - Coordinates of station 2
%   lag_time - Lag time vector
%   ncf_pos  - NCF for positive lags
%   ncf_neg  - NCF for negative lags

fid = fopen(filename, 'r');

if fid == -1
    error('Cannot open file: %s', filename);
end

try
    % Read station 1 coordinates
    line1 = fgetl(fid);
    coords1 = sscanf(line1, '%e %e %e');
    lon1 = coords1(1);
    lat1 = coords1(2);
    elev1 = coords1(3);
    
    % Read station 2 coordinates
    line2 = fgetl(fid);
    coords2 = sscanf(line2, '%e %e %e');
    lon2 = coords2(1);
    lat2 = coords2(2);
    elev2 = coords2(3);
    
    % Read data
    data = [];
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(strtrim(line))
            values = sscanf(line, '%e %e %e');
            if length(values) == 3
                data = [data; values'];
            end
        end
    end
    
    fclose(fid);
    
    lag_time = data(:, 1);
    ncf_pos = data(:, 2);
    ncf_neg = data(:, 3);
    
catch ME
    fclose(fid);
    rethrow(ME);
end

end