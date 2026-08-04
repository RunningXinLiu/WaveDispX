function picks = pick_dispersion_ridge(energy, frequency, velocity, varargin)
%PICK_DISPERSION_RIDGE Pick a smooth maximum-energy dispersion ridge.
%
% The dynamic-programming objective combines log energy with a quadratic
% inter-frequency velocity-change penalty.  The returned uncertainty is a
% local half-power width and should be supplemented by bootstrap tests for
% publication-quality uncertainty.

p = inputParser;
addParameter(p, 'FrequencyRange', [min(frequency), max(frequency)], ...
    @(x) isnumeric(x) && numel(x)==2 && x(2)>x(1));
addParameter(p, 'VelocityRange', [min(velocity), max(velocity)], ...
    @(x) isnumeric(x) && numel(x)==2 && x(2)>x(1));
addParameter(p, 'Smoothness', 1.5, @(x) isscalar(x) && x >= 0);
addParameter(p, 'MaxJump', inf, @(x) isscalar(x) && x > 0);
addParameter(p, 'MinimumEnergy', 0.1, @(x) isscalar(x) && x >= 0 && x <= 1);
parse(p, varargin{:});
opts = p.Results;

frequency = frequency(:)';
velocity = velocity(:);
if ~isequal(size(energy), [numel(velocity), numel(frequency)])
    error('pick_dispersion_ridge:DimensionMismatch', ...
        'energy must be velocity-by-frequency.');
end

frequency_keep = frequency >= opts.FrequencyRange(1) & ...
    frequency <= opts.FrequencyRange(2);
velocity_keep = velocity >= opts.VelocityRange(1) & ...
    velocity <= opts.VelocityRange(2);
f = frequency(frequency_keep);
v = velocity(velocity_keep);
e = energy(velocity_keep, frequency_keep);
if isempty(f) || isempty(v)
    error('pick_dispersion_ridge:EmptySearchWindow', ...
        'FrequencyRange and VelocityRange select no samples.');
end

if numel(v) > 1
    dv = median(diff(v));
else
    dv = 1;
end
velocity_difference = v-v';
transition_penalty = opts.Smoothness*(velocity_difference/dv).^2;
transition_penalty(abs(velocity_difference) > opts.MaxJump) = inf;

score = -inf(size(e));
back_pointer = ones(size(e), 'uint32');
score(:,1) = log(max(e(:,1), eps));
for frequency_index = 2:numel(f)
    candidate = score(:,frequency_index-1)' - transition_penalty;
    [best_previous, previous_index] = max(candidate, [], 2);
    score(:,frequency_index) = log(max(e(:,frequency_index),eps)) + best_previous;
    back_pointer(:,frequency_index) = uint32(previous_index);
end

[~, current_index] = max(score(:,end));
ridge_index = zeros(1, numel(f));
ridge_index(end) = current_index;
for frequency_index = numel(f):-1:2
    ridge_index(frequency_index-1) = ...
        back_pointer(ridge_index(frequency_index), frequency_index);
end

linear_index = sub2ind(size(e), ridge_index, 1:numel(f));
ridge_energy = e(linear_index);
ridge_velocity = v(ridge_index);
ridge_velocity = ridge_velocity(:);
uncertainty = zeros(size(ridge_velocity));
for frequency_index = 1:numel(f)
    threshold = 0.5*ridge_energy(frequency_index);
    above_threshold = e(:,frequency_index) >= threshold;
    left_index = ridge_index(frequency_index);
    right_index = left_index;
    while left_index > 1 && above_threshold(left_index-1)
        left_index = left_index-1;
    end
    while right_index < numel(v) && above_threshold(right_index+1)
        right_index = right_index+1;
    end
    uncertainty(frequency_index) = 0.5* ...
        (v(right_index)-v(left_index));
end

picked_frequency = f(:);
ridge_energy = ridge_energy(:);
uncertainty = uncertainty(:);
valid = ridge_energy >= opts.MinimumEnergy;
picks = table(picked_frequency(valid), ridge_velocity(valid), ...
    ridge_energy(valid), uncertainty(valid), 'VariableNames', ...
    {'frequency_hz', 'phase_velocity_mps', 'normalized_energy', ...
     'half_power_uncertainty_mps'});
end
