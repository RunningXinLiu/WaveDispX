function [lag_time, ncf_pos, ncf_neg] = compute_pair_ncf(trace1, trace2, dt, ...
    segment_length, overlap, max_lag, normalize_method)
% COMPUTE_PAIR_NCF Compute NCF for a station pair with optional segmentation
%
% Inputs:
%   trace1          - Time series from station 1 (column vector)
%   trace2          - Time series from station 2 (column vector)
%   dt              - Sampling interval (seconds)
%   segment_length  - Length of each segment in seconds ([] for full trace)
%   overlap         - Overlap fraction 0-1
%   max_lag         - Maximum lag time in seconds ([] for auto)
%   normalize_method - 'none', 'one-bit', or 'rms'
%
% Outputs:
%   lag_time - Lag time vector (positive values only, starting from 0)
%   ncf_pos  - NCF for positive lag times (causal part)
%   ncf_neg  - NCF for negative lag times (acausal part, flipped)

validateattributes(dt, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(overlap, {'numeric'}, {'scalar', '>=', 0, '<', 1});

% Ensure column vectors
trace1 = trace1(:);
trace2 = trace2(:);

if length(trace1) ~= length(trace2)
    error('compute_pair_ncf:LengthMismatch', ...
        'The two traces must contain the same number of samples.');
end

npts = length(trace1);
total_time = (npts - 1) * dt;

% Set default segment length (full trace if not specified)
if isempty(segment_length) || segment_length <= 0
    segment_length = total_time;
end

% Set default max lag
if isempty(max_lag) || max_lag <= 0
    max_lag = segment_length / 2;
end

% Calculate segment parameters
seg_samples = max(2, round(segment_length / dt));
step_samples = round(seg_samples * (1 - overlap));
max_lag_samples = round(max_lag / dt);

% Ensure segment fits in data
if seg_samples > npts
    seg_samples = npts;
    step_samples = npts;
    warning('Segment length exceeds data length. Using full trace.');
end

% Ensure step is at least 1 sample
if step_samples < 1
    step_samples = 1;
end

if max_lag_samples > seg_samples-1
    warning('compute_pair_ncf:LagClipped', ...
        'MaxLag exceeds the window length and was clipped to %.3f s.', ...
        (seg_samples-1)*dt);
    max_lag_samples = seg_samples-1;
end

% Number of segments
n_segments = max(1, floor((npts - seg_samples) / step_samples) + 1);

% Initialize stacked correlation
ncf_length = 2 * max_lag_samples + 1;
ncf_stacked = zeros(ncf_length, 1);
valid_segments = 0;
segment_taper = tukeywin(seg_samples, 0.1);

for seg = 1:n_segments
    idx_start = (seg - 1) * step_samples + 1;
    idx_end = idx_start + seg_samples - 1;
    
    if idx_end > npts
        break;
    end
    
    seg1 = trace1(idx_start:idx_end);
    seg2 = trace2(idx_start:idx_end);
    
    % Skip segments with all zeros or NaNs
    if all(seg1 == 0) || all(seg2 == 0) || any(isnan(seg1)) || any(isnan(seg2))
        continue;
    end
    
    % Apply normalization
    seg1 = normalize_trace(seg1, normalize_method);
    seg2 = normalize_trace(seg2, normalize_method);
    
    % Remove trend and taper each window before correlation.
    seg1 = detrend(seg1, 'linear') .* segment_taper;
    seg2 = detrend(seg2, 'linear') .* segment_taper;
    
    % Skip if zero variance after mean removal
    if std(seg1) == 0 || std(seg2) == 0
        continue;
    end
    
    % Cross-correlation: xcorr(seg1, seg2) gives correlation with seg2 shifted
    % Positive lags: seg2 is delayed relative to seg1 (energy travels from 1 to 2)
    % Negative lags: seg2 is advanced relative to seg1 (energy travels from 2 to 1)
    [xcorr_full, lags] = xcorr(seg1, seg2, max_lag_samples, 'none');
    xcorr_full = xcorr_full(:);
    lags = lags(:);
    
    % Normalize by number of overlapping points (unbiased estimate)
    norm_factor = seg_samples - abs(lags);
    norm_factor(norm_factor <= 0) = 1; % Avoid division by zero
    xcorr_full = xcorr_full ./ norm_factor;
    
    % Stack
    ncf_stacked = ncf_stacked + xcorr_full;
    valid_segments = valid_segments + 1;
end

% Average over valid segments
if valid_segments > 0
    ncf_stacked = ncf_stacked / valid_segments;
else
    warning('No valid segments found for correlation.');
end

% Create lag time vector (positive values only)
lag_time = (0:max_lag_samples)' * dt;

% Split into positive and negative parts
% Center of xcorr output corresponds to lag = 0
center_idx = max_lag_samples + 1;

% Positive and negative refer to MATLAB's xcorr(trace1,trace2) convention.
% Do not label either branch causal until the virtual-source orientation is
% selected by the caller.
ncf_pos = ncf_stacked(center_idx:end);

% Flip the negative branch so both arrays start at zero lag.
ncf_neg = flipud(ncf_stacked(1:center_idx));

end
