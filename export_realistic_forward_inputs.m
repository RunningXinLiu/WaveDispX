function output_file=export_realistic_forward_inputs(results_input,varargin)
%EXPORT_REALISTIC_FORWARD_INPUTS Save mixed traces and vertical responses.

p=inputParser;
addParameter(p,'OutputFile','',@(x) ischar(x) || isstring(x));
parse(p,varargin{:});

if ischar(results_input) || isstring(results_input)
    loaded=load(char(results_input),'results');
    results=loaded.results;
else
    results=results_input;
end
if strlength(string(p.Results.OutputFile))==0
    output_file=fullfile(results.output_folder, ...
        'realistic_forward_inputs.mat');
else
    output_file=char(p.Results.OutputFile);
end

model=results.model;
source=results.source;
geometry=results.geometry;
traces_z=results.traces.z;
trace_time=results.traces.t;
source_response_z=results.single_source.z;
response_time=results.single_source.time;
source_start_time=results.single_source.source_start_time;
event_weight=results.single_source.event_weight;
save(output_file,'model','source','geometry','traces_z','trace_time', ...
    'source_response_z','response_time','source_start_time', ...
    'event_weight','-v7.3');
end
