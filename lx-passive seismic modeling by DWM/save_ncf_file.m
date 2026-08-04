function save_ncf_file(filename, lon1, lat1, elev1, lon2, lat2, elev2, ...
    lag_time, ncf_pos, ncf_neg)
% SAVE_NCF_FILE Save NCF to file in specified format
%
% Format:
%   Row 1: lon1, lat1, elev1 (station 1)
%   Row 2: lon2, lat2, elev2 (station 2)
%   Row 3+: lag_time, ncf_pos, ncf_neg
%
% Inputs:
%   filename - Output file path
%   lon1, lat1, elev1 - Coordinates of station 1
%   lon2, lat2, elev2 - Coordinates of station 2
%   lag_time - Lag time vector (column)
%   ncf_pos  - NCF for positive lags (column)
%   ncf_neg  - NCF for negative lags (column)

fid = fopen(filename, 'w');

if fid == -1
    error('Cannot open file for writing: %s', filename);
end

try
    % Row 1: station 1 coordinates
    fprintf(fid, '%.7e %.7e %.7e\n', lon1, lat1, elev1);
    
    % Row 2: station 2 coordinates
    fprintf(fid, '%.7e %.7e %.7e\n', lon2, lat2, elev2);
    
    % Row 3 onwards: lag_time, ncf_pos, ncf_neg
    for k = 1:length(lag_time)
        fprintf(fid, '%.7e %.7e %.7e\n', lag_time(k), ncf_pos(k), ncf_neg(k));
    end
    
    fclose(fid);
catch ME
    fclose(fid);
    rethrow(ME);
end

end