function output_file=export_python_goldens(output_file)
%EXPORT_PYTHON_GOLDENS Export compact MATLAB reference arrays for Python/C++.
%
% This intentionally exports primitive numerical kernels rather than the
% multi-gigabyte end-to-end workspace.  The Python test suite consumes the
% real/imaginary datasets and checks each migrated DWM layer.

repository_root=fileparts(fileparts(mfilename('fullpath')));
matlab_core=fullfile(repository_root,'lx-passive seismic modeling by DWM');
active_core=fullfile(repository_root,'active seismic modeling by DWM');
addpath(repository_root,matlab_core,active_core);
if nargin<1 || strlength(string(output_file))==0
    output_file=fullfile(fileparts(mfilename('fullpath')), ...
        'golden_dwm_kernels.h5');
end
if isfile(output_file)
    delete(output_file);
end

model.vp=[380 760];
model.vs=[220 440];
model.dns=[2000 2000];
model.thk=[10 inf];
model.Damp_p=[0 0];
model.Damp_s=[0 0];
wavenumber=[0 0.01 0.05 0.1];
frequency=[0 5 12 18];

h5write_numeric(output_file,'/model/vp_mps',model.vp);
h5write_numeric(output_file,'/model/vs_mps',model.vs);
h5write_numeric(output_file,'/model/density_kgm3',model.dns);
h5write_numeric(output_file,'/model/thickness_m',model.thk);
h5write_numeric(output_file,'/grid/wavenumber_radpm',wavenumber);
h5write_numeric(output_file,'/grid/frequency_hz',frequency);

case_count=numel(wavenumber)*numel(frequency);
kg_sh=zeros(2,2,case_count);
kg_svp=zeros(4,4,case_count);
case_k=zeros(1,case_count);
case_f=zeros(1,case_count);
case_index=0;
for frequency_index=1:numel(frequency)
    for wavenumber_index=1:numel(wavenumber)
        case_index=case_index+1;
        case_k(case_index)=wavenumber(wavenumber_index);
        case_f(case_index)=frequency(frequency_index);
        omega=2*pi*frequency(frequency_index)-1i*2*pi*0.25;
        [kg_sh(:,:,case_index),kg_svp(:,:,case_index)]=GlobalStiffMat( ...
            model.vp,model.vs,model.Damp_p,model.Damp_s,model.dns, ...
            model.thk,wavenumber(wavenumber_index),omega);
    end
end
h5write_numeric(output_file,'/global_stiffness/wavenumber_radpm',case_k);
h5write_numeric(output_file,'/global_stiffness/frequency_hz',case_f);
h5write_complex(output_file,'/global_stiffness/kg_sh',kg_sh);
h5write_complex(output_file,'/global_stiffness/kg_svp',kg_svp);

% Deterministic finite-record NCF reference (no source labels).
dt=0.01;
receiver_x=0:2:8;
time=(0:799)'*dt;
mixed=zeros(numel(time),numel(receiver_x));
for receiver_index=1:numel(receiver_x)
    delay=receiver_x(receiver_index)/275;
    mixed(:,receiver_index)=sin(2*pi*6*(time-delay))+ ...
        0.35*cos(2*pi*9.5*(time-0.7*delay))+ ...
        0.05*sin(2*pi*(1.3+0.2*receiver_index)*time);
end
[ncf,lag_time,distance,pair_count]=compute_ncf_gather_from_traces( ...
    mixed,receiver_x,dt,'SegmentLength',4,'Overlap',0.5, ...
    'MaxLag',1,'Normalize','none','Branch','symmetric', ...
    'DistanceBin',2,'Algorithm','linear-fft');
h5write_numeric(output_file,'/processing_ncf/mixed_record',mixed);
h5write_numeric(output_file,'/processing_ncf/receiver_x_m',receiver_x);
h5write_numeric(output_file,'/processing_ncf/dt_s',dt);
h5write_numeric(output_file,'/processing_ncf/gather',ncf);
h5write_numeric(output_file,'/processing_ncf/lag_time_s',lag_time);
h5write_numeric(output_file,'/processing_ncf/distance_m',distance);
h5write_numeric(output_file,'/processing_ncf/pair_count',pair_count);

% Phase-shift reference with an exactly known non-dispersive velocity.
dt_dispersion=0.005;
time_dispersion=(0:999)'*dt_dispersion;
distance_dispersion=5:5:200;
true_velocity=280;
signal_frequency=[7 9 11 13];
wavefield=zeros(numel(time_dispersion),numel(distance_dispersion));
for signal_index=1:numel(signal_frequency)
    wavefield=wavefield+cos(2*pi*signal_frequency(signal_index)*( ...
        time_dispersion-distance_dispersion/true_velocity));
end
velocity=180:2:380;
[phase_energy,phase_frequency]=surface_wave_dispersion( ...
    wavefield,distance_dispersion,dt_dispersion,velocity,14, ...
    'Method','phase','FrequencyMin',6,'ZeroPadding',4, ...
    'DistanceTaperFraction',0.5,'Plot',false);
h5write_numeric(output_file,'/processing_dispersion/wavefield',wavefield);
h5write_numeric(output_file,'/processing_dispersion/distance_m',distance_dispersion);
h5write_numeric(output_file,'/processing_dispersion/dt_s',dt_dispersion);
h5write_numeric(output_file,'/processing_dispersion/velocity_mps',velocity);
h5write_numeric(output_file,'/processing_dispersion/frequency_hz',phase_frequency);
h5write_numeric(output_file,'/processing_dispersion/energy',phase_energy);

% Compact vertical-load f-k Green response for the migrated C++ solver.
green_nt=512;
green_dt=0.01;
green_ymax=100;
green_delay_s=0.5;
green_fmax=10;
green=Green_HV_delta_one_par(model,green_nt,green_dt,green_ymax, ...
    green_delay_s,green_fmax,'hv');
green_frequency=(0:size(green.VV,2)-1)/(green_nt*green_dt);
h5write_numeric(output_file,'/vertical_fk/nt',green_nt);
h5write_numeric(output_file,'/vertical_fk/dt_s',green_dt);
h5write_numeric(output_file,'/vertical_fk/ymax_m',green_ymax);
h5write_numeric(output_file,'/vertical_fk/delay_s',green_delay_s);
h5write_numeric(output_file,'/vertical_fk/maximum_frequency_hz',green_fmax);
h5write_numeric(output_file,'/vertical_fk/wavenumber_radpm',green.k);
h5write_numeric(output_file,'/vertical_fk/frequency_hz',green_frequency);
h5write_complex(output_file,'/vertical_fk/vr',green.VR);
h5write_complex(output_file,'/vertical_fk/vv',green.VV);
h5write_complex(output_file,'/horizontal_fk/hr',green.HR);
h5write_complex(output_file,'/horizontal_fk/ht',green.HT);
h5write_complex(output_file,'/horizontal_fk/hv',green.HV);
green_distance=[10 25 50 100];
[green_wavefield,green_time]=Green_HV_delta_two( ...
    green,green_nt,green_dt,green_distance,green_fmax,'hv');
h5write_numeric(output_file,'/vertical_receiver/distance_m',green_distance);
h5write_numeric(output_file,'/vertical_receiver/time_s',green_time);
h5write_numeric(output_file,'/vertical_receiver/radial',green_wavefield(:,:,4));
h5write_numeric(output_file,'/vertical_receiver/vertical',green_wavefield(:,:,5));
h5write_numeric(output_file,'/horizontal_receiver/radial',green_wavefield(:,:,1));
h5write_numeric(output_file,'/horizontal_receiver/tangential',green_wavefield(:,:,2));
h5write_numeric(output_file,'/horizontal_receiver/vertical',green_wavefield(:,:,3));

% Compact full active shot including Ricker convolution and delay removal.
active_source.fm=5;
active_source.dt=0.01;
active_source.offset=10:30:100;
active_source.maxfre=10;
[active_wavefield,active_time]=active_modeling_DWM( ...
    model,active_source,'hv');
h5write_numeric(output_file,'/active_shot/time_s',active_time);
h5write_numeric(output_file,'/active_shot/distance_m',active_source.offset);
h5write_numeric(output_file,'/active_shot/components',active_wavefield);
h5write_numeric(output_file,'/active_shot/dominant_frequency_hz',active_source.fm);
h5write_numeric(output_file,'/active_shot/dt_s',active_source.dt);
h5write_numeric(output_file,'/active_shot/maximum_frequency_hz',active_source.maxfre);

h5writeatt(output_file,'/','description', ...
    'MATLAB DWM primitive reference; omega includes -i*2*pi*0.25 EWM shift');
fprintf('Python/C++ golden data: %s\n',output_file);
end

function h5write_numeric(filename,dataset,value)
h5create(filename,dataset,size(value),'Datatype','double');
h5write(filename,dataset,double(value));
end

function h5write_complex(filename,dataset,value)
h5write_numeric(filename,[dataset,'_real'],real(value));
h5write_numeric(filename,[dataset,'_imag'],imag(value));
end
