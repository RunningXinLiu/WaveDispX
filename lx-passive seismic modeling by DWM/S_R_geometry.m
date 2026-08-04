function SR = S_R_geometry(source_number, varargin)
%S_R_GEOMETRY Build reproducible ambient-source and receiver geometry.
%
% SR = S_R_geometry(N) keeps the historical annular-source default.
% Name-value options:
%   Distribution       - 'annulus' (default), 'traffic', or 'love'
%   InnerRadius        - inner annulus radius [m], default 500
%   OuterRadius        - outer annulus radius [m], default 1000
%   SourceHalfLength   - half-length for traffic/love sources [m], default 300
%   RoadOffset         - nominal road offset [m], default 10
%   LoveSourceX        - x coordinate of Love-wave source line [m], default -150
%   ReceiverHalfLength - receiver-line half aperture [m], default 100
%   ReceiverSpacing    - receiver spacing [m], default 5
%   AngularSampling    - random (default), stratified, or uniform
%   Seed               - optional local RNG seed

validateattributes(source_number, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative'}, mfilename, 'source_number');

p = inputParser;
addParameter(p, 'Distribution', 'annulus', @(x) ischar(x) || isstring(x));
addParameter(p, 'InnerRadius', 500, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'OuterRadius', 1000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'SourceHalfLength', 300, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'RoadOffset', 10, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'LoveSourceX', -150, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'ReceiverHalfLength', 100, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'ReceiverSpacing', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'AngularSampling', 'random', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'Seed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});
opts = p.Results;

if ~isempty(opts.Seed)
    rng(opts.Seed, 'twister');
end

distribution = lower(string(opts.Distribution));
switch distribution
    case "annulus"
        if opts.OuterRadius <= opts.InnerRadius
            error('S_R_geometry:InvalidAnnulus', ...
                'OuterRadius must be greater than InnerRadius.');
        end
        % Uniform source density per unit area, not merely uniform radius.
        u = rand(1, source_number);
        r = sqrt(opts.InnerRadius^2 + ...
            (opts.OuterRadius^2 - opts.InnerRadius^2) .* u);
        angular_sampling = lower(string(opts.AngularSampling));
        switch angular_sampling
            case "random"
                theta = 2*pi*rand(1, source_number);
            case "stratified"
                theta = 2*pi*((0:source_number-1)+ ...
                    rand(1,source_number))/max(source_number,1);
                theta = theta + 2*pi*rand;
            case "uniform"
                theta = 2*pi*(0:source_number-1)/max(source_number,1) + ...
                    2*pi*rand;
            otherwise
                error('S_R_geometry:UnknownAngularSampling', ...
                    'AngularSampling must be random, stratified, or uniform.');
        end
        [sourceX, sourceY] = pol2cart(theta, r);

    case "traffic"
        sourceX = 2*opts.SourceHalfLength*rand(1, source_number) ...
            - opts.SourceHalfLength;
        sourceY = opts.RoadOffset*(0.5 + 1.5*rand(1, source_number));

    case "love"
        sourceX = opts.LoveSourceX*ones(1, source_number);
        sourceY = 2*opts.SourceHalfLength*rand(1, source_number) ...
            - opts.SourceHalfLength;

    otherwise
        error('S_R_geometry:UnknownDistribution', ...
            'Distribution must be annulus, traffic, or love.');
end

receiverX = -opts.ReceiverHalfLength:opts.ReceiverSpacing:opts.ReceiverHalfLength;
receiverY = zeros(size(receiverX));

SR.sx = sourceX;
SR.sy = sourceY;
SR.rx = receiverX;
SR.ry = receiverY;
SR.source_distribution = char(distribution);
SR.receiver_spacing = opts.ReceiverSpacing;
SR.angular_sampling = char(opts.AngularSampling);
end
