function [gather, lag_time, distances, counts] = ...
    compute_ncf_gather_from_traces(data, receiver_x, dt, varargin)
%COMPUTE_NCF_GATHER_FROM_TRACES Correlate and distance-bin a linear array.
%
% This in-memory path is useful for deterministic processing diagnostics and
% avoids changing the on-disk NCF format used by compute_ncf.

p = inputParser;
addParameter(p, 'SegmentLength', [], ...
    @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'Overlap', 0.5, ...
    @(x) isscalar(x) && x >= 0 && x < 1);
addParameter(p, 'MaxLag', [], ...
    @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'Normalize', 'none', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'Branch', 'symmetric', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'DistanceBin', [], ...
    @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'Algorithm', 'auto', ...
    @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opts = p.Results;

validateattributes(data, {'numeric'}, {'2d', 'nonempty', 'finite'});
validateattributes(receiver_x, {'numeric'}, {'vector', 'finite'});
validateattributes(dt, {'numeric'}, {'scalar', 'positive', 'finite'});
receiver_x = receiver_x(:)';
if size(data,2) ~= numel(receiver_x)
    error('compute_ncf_gather_from_traces:GeometryMismatch', ...
        'Data columns must equal the number of receiver coordinates.');
end
if numel(receiver_x) < 2
    error('compute_ncf_gather_from_traces:TooFewReceivers', ...
        'At least two receivers are required.');
end

branch = lower(string(opts.Branch));
if ~any(branch == ["symmetric", "positive", "negative"])
    error('compute_ncf_gather_from_traces:UnknownBranch', ...
        'Branch must be symmetric, positive, or negative.');
end
algorithm=lower(string(opts.Algorithm));
if ~any(algorithm==["auto","pairwise","linear-fft"])
    error('compute_ncf_gather_from_traces:UnknownAlgorithm', ...
        'Algorithm must be auto, pairwise, or linear-fft.');
end
if isempty(opts.DistanceBin)
    sorted_x = unique(sort(receiver_x));
    opts.DistanceBin = min(diff(sorted_x));
end

pairs = nchoosek(1:numel(receiver_x),2);
pair_distance = abs(receiver_x(pairs(:,2))-receiver_x(pairs(:,1)));
pair_bin = round(pair_distance/opts.DistanceBin)*opts.DistanceBin;
distances = reshape(unique(pair_bin(pair_bin > 0)),1,[]);
gather = [];
counts = zeros(1,numel(distances));
lag_time = [];

spacing=diff(receiver_x);
spacing_tolerance=max(1,max(abs(receiver_x)))*1e-10;
uniform_linear=all(spacing>0) && ...
    max(abs(spacing-spacing(1)))<=spacing_tolerance;
unit_distance_bins=uniform_linear && ...
    abs(opts.DistanceBin-spacing(1))<=spacing_tolerance;
linear_fft_eligible=unit_distance_bins;
if algorithm=="linear-fft" && ~linear_fft_eligible
    error('compute_ncf_gather_from_traces:LinearFftNotEligible', ...
        ['linear-fft requires increasing, equally spaced receivers and a ' ...
         'matching DistanceBin.']);
end
use_linear_fft=algorithm=="linear-fft" || ...
    (algorithm=="auto" && linear_fft_eligible);
if use_linear_fft
    [gather,lag_time,distances,counts]=linear_fft_ncf_gather( ...
        data,receiver_x,dt,opts);
    return
end

for pair_index = 1:size(pairs,1)
    i = pairs(pair_index,1);
    j = pairs(pair_index,2);
    [pair_lag, pos, neg] = compute_pair_ncf(data(:,i), data(:,j), dt, ...
        opts.SegmentLength, opts.Overlap, opts.MaxLag, opts.Normalize);
    if isempty(gather)
        lag_time = pair_lag;
        gather = zeros(numel(lag_time),numel(distances));
    end
    switch branch
        case "positive"
            trace = pos;
        case "negative"
            trace = neg;
        otherwise
            trace = 0.5*(pos+neg);
    end
    distance_index = find(distances == pair_bin(pair_index),1);
    gather(:,distance_index) = gather(:,distance_index)+trace;
    counts(distance_index) = counts(distance_index)+1;
end

gather = gather ./ max(counts,1);
end

function [gather,lag_time,distances,counts]=linear_fft_ncf_gather( ...
    data,receiver_x,dt,opts)
npts=size(data,1);
receiver_count=numel(receiver_x);
total_time=(npts-1)*dt;
segment_length=opts.SegmentLength;
if isempty(segment_length) || segment_length<=0
    segment_length=total_time;
end
seg_samples=max(2,round(segment_length/dt));
if seg_samples>npts
    seg_samples=npts;
end
step_samples=max(1,round(seg_samples*(1-opts.Overlap)));
if isempty(opts.MaxLag) || opts.MaxLag<=0
    max_lag_samples=round(0.5*segment_length/dt);
else
    max_lag_samples=round(opts.MaxLag/dt);
end
max_lag_samples=min(max_lag_samples,seg_samples-1);
n_segments=max(1,floor((npts-seg_samples)/step_samples)+1);
nfft=2^nextpow2(2*seg_samples-1);
spatial_nfft=2^nextpow2(2*receiver_count-1);
counts=receiver_count-(1:receiver_count-1);
distances=(1:receiver_count-1)*(receiver_x(2)-receiver_x(1));
gather=zeros(max_lag_samples+1,receiver_count-1);
segment_taper=tukeywin(seg_samples,0.1);
normalization=char(lower(string(opts.Normalize)));
valid_segments=0;

for segment_index=1:n_segments
    first_sample=(segment_index-1)*step_samples+1;
    sample_index=first_sample:first_sample+seg_samples-1;
    segment=data(sample_index,:);
    if any(all(segment==0,1))
        continue
    end
    for receiver_index=1:receiver_count
        segment(:,receiver_index)=normalize_trace( ...
            segment(:,receiver_index),normalization);
    end
    segment=detrend(segment,'linear').*segment_taper;
    if any(std(segment,0,1)==0)
        continue
    end
    temporal_spectrum=fft(segment,nfft,1);
    spatial_spectrum=fft(temporal_spectrum,spatial_nfft,2);
    spatial_correlation=ifft( ...
        spatial_spectrum.*conj(spatial_spectrum),[],2);
    distance_cross_spectrum=conj( ...
        spatial_correlation(:,2:receiver_count))./counts;
    correlation=real(ifft(distance_cross_spectrum,nfft,1));
    positive=correlation(1:max_lag_samples+1,:);
    negative=[correlation(1,:); ...
        correlation(end:-1:end-max_lag_samples+1,:)];
    unbiased_denominator=(seg_samples-(0:max_lag_samples)).';
    positive=positive./unbiased_denominator;
    negative=negative./unbiased_denominator;
    switch lower(string(opts.Branch))
        case "positive"
            segment_gather=positive;
        case "negative"
            segment_gather=negative;
        otherwise
            segment_gather=0.5*(positive+negative);
    end
    gather=gather+segment_gather;
    valid_segments=valid_segments+1;
end
if valid_segments>0
    gather=gather/valid_segments;
end
lag_time=(0:max_lag_samples)'*dt;
end
