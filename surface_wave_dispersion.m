function [energy, frequency, velocity, raw_amplitude] = ...
    surface_wave_dispersion(data, distance, dt, velocity, fmax, varargin)
%SURFACE_WAVE_DISPERSION Compute phase-shift, F-Bessel, or F-Hankel energy.
%
% data is time-by-distance.  distance and velocity are in metres and m/s;
% dt is seconds.  Output energy is velocity-by-frequency and is normalized
% independently at each frequency.

p = inputParser;
addParameter(p, 'Method', 'phase', @(x) ischar(x) || isstring(x));
addParameter(p, 'Component', 'vertical', @(x) ischar(x) || isstring(x));
addParameter(p, 'FrequencyMin', 0, @(x) isscalar(x) && x >= 0);
addParameter(p, 'ZeroPadding', 4, @(x) isscalar(x) && x >= 1);
addParameter(p, 'TaperFraction', 0.1, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'DistanceTaperFraction', 0, ...
    @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'SpectrumNormalization', 'none', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'RadialWeighting', 'area', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'Plot', false, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
opts = p.Results;

validateattributes(data, {'numeric'}, {'2d', 'nonempty', 'finite'});
validateattributes(distance, {'numeric'}, {'vector', 'finite', 'nonnegative'});
validateattributes(velocity, {'numeric'}, {'vector', 'finite', 'positive'});
validateattributes(dt, {'numeric'}, {'scalar', 'finite', 'positive'});
validateattributes(fmax, {'numeric'}, {'scalar', 'finite', 'positive'});

distance = distance(:)';
velocity = velocity(:);
if size(data,2) ~= numel(distance)
    error('surface_wave_dispersion:GeometryMismatch', ...
        'The number of data columns must equal the number of distances.');
end

keep = distance > 0;
data = data(:,keep);
distance = distance(keep);
[distance, order] = sort(distance);
data = data(:,order);
if numel(distance) < 2
    error('surface_wave_dispersion:TooFewDistances', ...
        'At least two positive offsets are required.');
end
if any(diff(distance) <= 0)
    error('surface_wave_dispersion:DuplicateDistance', ...
        'Distances must be unique. Bin or stack duplicate offsets first.');
end

method = lower(string(opts.Method));
if ~any(method == ["phase", "bessel", "hankel"])
    error('surface_wave_dispersion:UnknownMethod', ...
        'Method must be phase, bessel, or hankel.');
end
component = lower(string(opts.Component));
is_radial = any(component == ["radial", "r"]);
bessel_order = double(is_radial);
spectrum_normalization = lower(string(opts.SpectrumNormalization));
if ~any(spectrum_normalization == ["none", "phase"])
    error('surface_wave_dispersion:UnknownSpectrumNormalization', ...
        'SpectrumNormalization must be none or phase.');
end
radial_weighting = lower(string(opts.RadialWeighting));
if ~any(radial_weighting == ["area", "uniform"])
    error('surface_wave_dispersion:UnknownRadialWeighting', ...
        'RadialWeighting must be area or uniform.');
end

% Remove the temporal mean/trend and suppress FFT edge leakage.
data = detrend(data, 'linear');
taper = tukeywin(size(data,1), opts.TaperFraction);
data = data .* taper;
distance_taper = ones(1,numel(distance));
if opts.DistanceTaperFraction > 0
    distance_taper = tukeywin(numel(distance), ...
        opts.DistanceTaperFraction).';
end

nfft = 2^nextpow2(ceil(size(data,1)*opts.ZeroPadding));
spectrum = fft(data, nfft, 1);
all_frequency = (0:floor(nfft/2))/(nfft*dt);
frequency_mask = all_frequency >= opts.FrequencyMin & ...
    all_frequency <= min(fmax, 0.5/dt) & all_frequency > 0;
frequency = all_frequency(frequency_mask);
spectrum = spectrum(find(frequency_mask), :); %#ok<FNDSB>
if spectrum_normalization == "phase"
    spectrum = spectrum ./ max(abs(spectrum), eps);
end

raw_amplitude = zeros(numel(velocity), numel(frequency));
omega = 2*pi*frequency;

if method == "phase"
    phase_only = spectrum ./ max(abs(spectrum), eps);
    % Apply spatial weights after phase normalization.  Applying them to
    % data beforehand would be cancelled by the unit-amplitude operation.
    phase_only = phase_only .* distance_taper;
    for frequency_index = 1:numel(frequency)
        steering = exp(1i*omega(frequency_index) .* ...
            (1./velocity)*distance);
        raw_amplitude(:,frequency_index) = abs( ...
            steering * phase_only(frequency_index,:).');
    end
else
    % Area weights for the radial integral integral r*C(r)*J_n(kr) dr.
    midpoint = 0.5*(distance(1:end-1)+distance(2:end));
    left_edge = max(0, distance(1)-0.5*(distance(2)-distance(1)));
    right_edge = distance(end)+0.5*(distance(end)-distance(end-1));
    edges = [left_edge, midpoint, right_edge];
    if radial_weighting == "area"
        radial_weight = 0.5*(edges(2:end).^2-edges(1:end-1).^2);
    else
        radial_weight = ones(size(distance));
    end

    for frequency_index = 1:numel(frequency)
        kr = (omega(frequency_index)./velocity)*distance;
        if method == "bessel"
            kernel = besselj(bessel_order, kr);
        else
            kernel = besselh(bessel_order, 1, kr);
        end
        weighted_spectrum = radial_weight .* distance_taper .* ...
            spectrum(frequency_index,:);
        raw_amplitude(:,frequency_index) = abs(kernel * weighted_spectrum.');
    end
end

column_max = max(raw_amplitude, [], 1);
column_max(column_max == 0) = 1;
energy = raw_amplitude ./ column_max;

if logical(opts.Plot)
    figure;
    imagesc(frequency, velocity, energy);
    axis xy;
    xlabel('Frequency (Hz)');
    ylabel('Phase velocity (m/s)');
    title(sprintf('%s dispersion energy', upper(char(method))));
    colormap(turbo);
    colorbar;
end
end
