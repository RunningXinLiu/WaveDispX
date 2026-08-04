function theory = compute_theoretical_dispersion(model, frequency, varargin)
%COMPUTE_THEORETICAL_DISPERSION Elastic Rayleigh/Love curves via disba.
%
% The wrapper sends the MATLAB layer model to the repository's Python
% helper in the seisloc Conda environment. Model units are m, m/s, and
% kg/m^3 (densities in g/cm^3 are also accepted for legacy examples).

p = inputParser;
addParameter(p, 'RayleighModes', 4, ...
    @(x) isscalar(x) && x >= 1 && fix(x)==x);
addParameter(p, 'LoveModes', 4, ...
    @(x) isscalar(x) && x >= 1 && fix(x)==x);
addParameter(p, 'CondaEnvironment', 'seisloc', ...
    @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opts = p.Results;

required_fields = {'vp','vs','dns','thk'};
for field_index = 1:numel(required_fields)
    if ~isfield(model, required_fields{field_index})
        error('compute_theoretical_dispersion:MissingModelField', ...
            'model.%s is required.', required_fields{field_index});
    end
end

vp = model.vp(:);
vs = model.vs(:);
density = model.dns(:);
thickness = model.thk(:);
frequency = frequency(:);
layer_count = numel(vs);
if any([numel(vp),numel(density),numel(thickness)] ~= layer_count)
    error('compute_theoretical_dispersion:DimensionMismatch', ...
        'vp, vs, dns, and thk must have equal lengths.');
end
if ~isinf(thickness(end))
    error('compute_theoretical_dispersion:MissingHalfspace', ...
        'The last model thickness must be Inf.');
end
validateattributes(frequency, {'numeric'}, ...
    {'vector','positive','finite','increasing'});

repository_root = fileparts(mfilename('fullpath'));
python_script = fullfile(repository_root, 'theoretical_dispersion_disba.py');
input_file = [tempname, '.mat'];
output_file = [tempname, '.mat'];
cleanup = onCleanup(@() cleanup_files(input_file, output_file));
save(input_file, 'vp', 'vs', 'density', 'thickness', 'frequency');

command = sprintf([ ...
    'conda run -n %s python "%s" "%s" "%s" ' ...
    '--rayleigh-modes %d --love-modes %d'], ...
    char(opts.CondaEnvironment), python_script, input_file, output_file, ...
    opts.RayleighModes, opts.LoveModes);
[status, command_output] = system(command);
if status ~= 0 || ~isfile(output_file)
    error('compute_theoretical_dispersion:PythonFailure', ...
        'The theoretical solver failed:\n%s', command_output);
end

result = load(output_file, 'frequency', 'rayleigh', 'love');
theory = struct;
theory.frequency = result.frequency(:);
theory.rayleigh = result.rayleigh;
theory.love = result.love;
theory.model = model;
theory.solver = 'disba/dunkin';
end

function cleanup_files(varargin)
for file_index = 1:nargin
    if isfile(varargin{file_index})
        delete(varargin{file_index});
    end
end
end
