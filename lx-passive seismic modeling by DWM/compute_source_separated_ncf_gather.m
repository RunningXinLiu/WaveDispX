function [gather,lag_time,distances,counts,cross_spectrum] = ...
    compute_source_separated_ncf_gather(source_response,receiver_x,dt,varargin)
%COMPUTE_SOURCE_SEPARATED_NCF_GATHER Ensemble NCF without source cross terms.
%
% source_response is time-by-receiver-by-source.  Correlations are formed
% for each independent source before the source ensemble is stacked.  This
% evaluates the infinite-record expectation of the synthetic noise field
% without finite-record cross terms between unrelated sources.

p=inputParser;
addParameter(p,'MaxLag',[], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p,'Branch','symmetric', ...
    @(x) ischar(x) || isstring(x));
addParameter(p,'DistanceBin',[], ...
    @(x) isempty(x) || (isscalar(x) && isfinite(x) && x>0));
addParameter(p,'SpectrumNormalization','none', ...
    @(x) ischar(x) || isstring(x));
addParameter(p,'WaterLevel',0.01, ...
    @(x) isscalar(x) && isfinite(x) && x>=0 && x<1);
addParameter(p,'TaperFraction',0.1, ...
    @(x) isscalar(x) && isfinite(x) && x>=0 && x<=1);
addParameter(p,'Algorithm','auto', ...
    @(x) ischar(x) || isstring(x));
parse(p,varargin{:});
opts=p.Results;

validateattributes(source_response,{'numeric'}, ...
    {'3d','nonempty','finite'});
validateattributes(receiver_x,{'numeric'},{'vector','finite'});
validateattributes(dt,{'numeric'},{'scalar','finite','positive'});
receiver_x=receiver_x(:)';
if size(source_response,2)~=numel(receiver_x)
    error('compute_source_separated_ncf_gather:GeometryMismatch', ...
        'Response columns must equal receiver coordinates.');
end

branch=lower(string(opts.Branch));
if ~any(branch==["symmetric","positive","negative"])
    error('compute_source_separated_ncf_gather:UnknownBranch', ...
        'Branch must be symmetric, positive, or negative.');
end
normalization=lower(string(opts.SpectrumNormalization));
if ~any(normalization==["none","phase"])
    error('compute_source_separated_ncf_gather:UnknownNormalization', ...
        'SpectrumNormalization must be none or phase.');
end
algorithm=lower(string(opts.Algorithm));
if ~any(algorithm==["auto","pairwise","linear-fft"])
    error('compute_source_separated_ncf_gather:UnknownAlgorithm', ...
        'Algorithm must be auto, pairwise, or linear-fft.');
end
if isempty(opts.DistanceBin)
    opts.DistanceBin=min(diff(unique(sort(receiver_x))));
end

n_time=size(source_response,1);
n_source=size(source_response,3);
nfft=2^nextpow2(2*n_time-1);
if isempty(opts.MaxLag)
    max_lag_samples=n_time-1;
else
    max_lag_samples=min(round(opts.MaxLag/dt),n_time-1);
end
lag_time=(0:max_lag_samples)'*dt;

pairs=nchoosek(1:numel(receiver_x),2);
pair_distance=abs(receiver_x(pairs(:,2))-receiver_x(pairs(:,1)));
pair_bin=round(pair_distance/opts.DistanceBin)*opts.DistanceBin;
distances=reshape(unique(pair_bin(pair_bin>0)),1,[]);
counts=zeros(1,numel(distances));
cross_spectrum=zeros(nfft,numel(distances));
taper=tukeywin(n_time,opts.TaperFraction);

spacing=diff(receiver_x);
spacing_tolerance=max(1,max(abs(receiver_x)))*1e-10;
uniform_linear=numel(receiver_x)>1 && all(spacing>0) && ...
    max(abs(spacing-spacing(1)))<=spacing_tolerance;
unit_distance_bins=uniform_linear && ...
    abs(opts.DistanceBin-spacing(1))<=spacing_tolerance;
separable_normalization=normalization=="none" || ...
    (normalization=="phase" && opts.WaterLevel==0);
linear_fft_eligible=unit_distance_bins && separable_normalization;
if algorithm=="linear-fft" && ~linear_fft_eligible
    error('compute_source_separated_ncf_gather:LinearFftNotEligible', ...
        ['linear-fft requires increasing, equally spaced receivers, a ' ...
         'matching DistanceBin, and none normalization or phase with zero ' ...
         'water level.']);
end
use_linear_fft=algorithm=="linear-fft" || ...
    (algorithm=="auto" && linear_fft_eligible);

if use_linear_fft
    receiver_count=numel(receiver_x);
    spatial_nfft=2^nextpow2(2*receiver_count-1);
    distances=(1:receiver_count-1)*spacing(1);
    counts=n_source*(receiver_count-(1:receiver_count-1));
    cross_spectrum=zeros(nfft,receiver_count-1);
    for source_index=1:n_source
        response=detrend(source_response(:,:,source_index),'linear').*taper;
        spectrum=fft(response,nfft,1);
        if normalization=="phase"
            amplitude=abs(spectrum);
            reliable=amplitude>0;
            spectrum(reliable)=spectrum(reliable)./amplitude(reliable);
            spectrum(~reliable)=0;
        end
        spatial_spectrum=fft(spectrum,spatial_nfft,2);
        spatial_correlation=ifft( ...
            spatial_spectrum.*conj(spatial_spectrum),[],2);
        cross_spectrum=cross_spectrum+ ...
            conj(spatial_correlation(:,2:receiver_count));
    end
else
    for source_index=1:n_source
        response=detrend(source_response(:,:,source_index),'linear').*taper;
        spectrum=fft(response,nfft,1);
        for distance_index=1:numel(distances)
            selected=find(pair_bin==distances(distance_index));
            pair_spectrum=spectrum(:,pairs(selected,1)).* ...
                conj(spectrum(:,pairs(selected,2)));
            if normalization=="phase"
                amplitude=abs(pair_spectrum);
                threshold=opts.WaterLevel*max(amplitude,[],1);
                reliable=amplitude>=threshold & amplitude>0;
                pair_spectrum(reliable)=pair_spectrum(reliable)./ ...
                    amplitude(reliable);
                pair_spectrum(~reliable)=0;
            end
            cross_spectrum(:,distance_index)= ...
                cross_spectrum(:,distance_index)+sum(pair_spectrum,2);
            counts(distance_index)=counts(distance_index)+numel(selected);
        end
    end
end
cross_spectrum=cross_spectrum./max(counts,1);

correlation=real(ifft(cross_spectrum,nfft,1));
positive=correlation(1:max_lag_samples+1,:);
negative=[correlation(1,:); ...
    correlation(end:-1:end-max_lag_samples+1,:)];
switch branch
    case "positive"
        gather=positive;
    case "negative"
        gather=negative;
    otherwise
        gather=0.5*(positive+negative);
end
end
