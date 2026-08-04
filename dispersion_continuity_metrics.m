function metrics = dispersion_continuity_metrics(energy, frequency, velocity, ...
    theory_frequency, theory_velocity, varargin)
%DISPERSION_CONTINUITY_METRICS Quantify energy continuity near a reference mode.
%
% The image is normalized by frequency before measuring how consistently a
% high-energy ridge supports the supplied theoretical dispersion curve.

p = inputParser;
addParameter(p, 'FrequencyRange', [-inf inf], ...
    @(x) isnumeric(x) && numel(x) == 2 && x(1) < x(2));
addParameter(p, 'VelocityTolerance', 12, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'SupportThreshold', 0.6, ...
    @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'JumpThreshold', 12, ...
    @(x) isscalar(x) && isfinite(x) && x > 0);
parse(p, varargin{:});
opts = p.Results;

validateattributes(energy, {'numeric'}, {'2d', 'nonempty', 'finite'});
validateattributes(frequency, {'numeric'}, {'vector', 'finite'});
validateattributes(velocity, {'numeric'}, {'vector', 'finite'});
validateattributes(theory_frequency, {'numeric'}, {'vector'});
validateattributes(theory_velocity, {'numeric'}, {'vector'});

frequency = frequency(:)';
velocity = velocity(:);
theory_frequency = theory_frequency(:);
theory_velocity = theory_velocity(:);
if size(energy,1) ~= numel(velocity) || size(energy,2) ~= numel(frequency)
    error('dispersion_continuity_metrics:SizeMismatch', ...
        'Energy must be velocity-by-frequency.');
end
if numel(theory_frequency) ~= numel(theory_velocity)
    error('dispersion_continuity_metrics:TheorySizeMismatch', ...
        'Theory frequency and velocity must have equal lengths.');
end

column_max = max(energy, [], 1);
column_max(column_max <= 0) = 1;
normalized_energy = energy ./ column_max;
reference_velocity = interp1(theory_frequency, theory_velocity, frequency, ...
    'linear', nan);
valid = frequency >= opts.FrequencyRange(1) & ...
    frequency <= opts.FrequencyRange(2) & isfinite(reference_velocity);

support = nan(size(frequency));
ridge_velocity = nan(size(frequency));
for frequency_index = find(valid)
    local = abs(velocity-reference_velocity(frequency_index)) <= ...
        opts.VelocityTolerance;
    if any(local)
        candidate_velocity = velocity(local);
        candidate_energy = normalized_energy(local,frequency_index);
        [support(frequency_index), peak_index] = max(candidate_energy);
        ridge_velocity(frequency_index) = candidate_velocity(peak_index);
    end
end

valid_index = find(valid & isfinite(support));
supported = support(valid_index) >= opts.SupportThreshold;
run_lengths = contiguous_run_lengths(supported);
if isempty(run_lengths)
    longest_run_samples = 0;
else
    longest_run_samples = max(run_lengths);
end

ridge_jump = abs(diff(ridge_velocity(valid_index)));
[~, global_peak_index] = max(normalized_energy, [], 1);
global_ridge_velocity = reshape(velocity(global_peak_index), 1, []);
global_error = abs(global_ridge_velocity(valid_index)- ...
    reference_velocity(valid_index));
global_jump = abs(diff(global_ridge_velocity(valid_index)));
metrics = struct;
metrics.frequency_hz = frequency;
metrics.reference_velocity_mps = reference_velocity;
metrics.support = support;
metrics.local_ridge_velocity_mps = ridge_velocity;
metrics.global_ridge_velocity_mps = global_ridge_velocity;
metrics.valid_sample_count = numel(valid_index);
metrics.supported_fraction = mean(supported);
metrics.longest_supported_fraction = longest_run_samples / max(1,numel(valid_index));
metrics.longest_supported_samples = longest_run_samples;
metrics.median_support = median(support(valid_index),'omitnan');
metrics.median_local_velocity_error_mps = median(abs( ...
    ridge_velocity(valid_index)-reference_velocity(valid_index)),'omitnan');
metrics.median_adjacent_jump_mps = median(ridge_jump,'omitnan');
metrics.large_jump_fraction = mean(ridge_jump > opts.JumpThreshold);
metrics.dominant_ridge_fraction = mean(global_error <= opts.VelocityTolerance);
metrics.median_global_velocity_error_mps = median(global_error,'omitnan');
metrics.global_large_jump_fraction = mean(global_jump > opts.JumpThreshold);
metrics.settings = opts;
end

function lengths = contiguous_run_lengths(mask)
mask = logical(mask(:));
if isempty(mask) || ~any(mask)
    lengths = [];
    return;
end
edges = diff([false; mask; false]);
starts = find(edges == 1);
stops = find(edges == -1)-1;
lengths = stops-starts+1;
end
