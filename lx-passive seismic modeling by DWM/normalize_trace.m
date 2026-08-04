function trace_out = normalize_trace(trace_in, method)
% NORMALIZE_TRACE Apply normalization to a trace
%
% Inputs:
%   trace_in - Input time series (column vector)
%   method   - Normalization method: 'none', 'one-bit', 'rms', 'running-abs-mean'
%
% Output:
%   trace_out - Normalized trace

trace_in = trace_in(:);

switch lower(method)
    case 'one-bit'
        % One-bit normalization: only keep sign
        trace_out = sign(trace_in);
        
    case 'rms'
        % RMS normalization: divide by RMS amplitude
        rms_val = sqrt(mean(trace_in.^2));
        if rms_val > eps
            trace_out = trace_in / rms_val;
        else
            trace_out = trace_in;
        end
        
    case 'running-abs-mean'
        % Running absolute mean normalization
        % Use a window of ~1 second or 10% of trace length
        win_length = max(10, round(length(trace_in) * 0.1));
        if mod(win_length, 2) == 0
            win_length = win_length + 1; % Make odd
        end
        
        % Compute running mean of absolute values
        abs_trace = abs(trace_in);
        running_mean = movmean(abs_trace, win_length);
        running_mean(running_mean < eps) = eps; % Avoid division by zero
        
        trace_out = trace_in ./ running_mean;
        
    case 'spectral-whitening'
        % Spectral whitening: flatten amplitude spectrum
        n = length(trace_in);
        nfft = 2^nextpow2(n);
        
        % FFT
        spec = fft(trace_in, nfft);
        
        % Get amplitude and phase
        amp = abs(spec);
        phase = angle(spec);
        
        % Whiten (set amplitude to 1, keep phase)
        amp_white = ones(size(amp));
        
        % Apply water level to avoid amplifying noise
        water_level = max(amp) * 0.01;
        amp_white(amp < water_level) = amp(amp < water_level) / water_level;
        
        % Reconstruct
        spec_white = amp_white .* exp(1i * phase);
        trace_out = real(ifft(spec_white, nfft));
        trace_out = trace_out(1:n);
        
    otherwise % 'none'
        trace_out = trace_in;
end

end