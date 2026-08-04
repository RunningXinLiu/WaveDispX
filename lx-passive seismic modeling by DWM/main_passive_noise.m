%%  half-space model
tic
% Damp_p=1./(2*Qp); % damping
% Damp_s=1./(2*Qs);

% model.vs   = 440;   % Shear wave velocity [m/s]
% model.vp   =760;
% model.dns  = 2;    % Mass density of layers (kg/m^3)
% model.thk  = inf;      % depth of layers [m]
% model.Damp_s=0;  %  viscoealstic damping
% model.Damp_p=0;  % A zero represents elastic media.

%%  three-layer model
% model.vs   = [300 180 450];   % Shear wave velocity [m/s]
% model.vp   =[600 1700 2000];
% model.dns  = [2 2 2 ];    % Mass density of layers (kg/m^3)
% model.thk  = [20 20 inf];      % depth of layers [m]
% model.Damp_s=[0 0 0];  %  viscoealstic damping
% model.Damp_p=[0 0 0];  % A zero represents elastic media.

% %% two-layer model
% model.vs   = [220 440];   % Shear wave velocity [m/s]
% model.vp   =[380 760];
% model.dns  = [1 1 ]*2000;    % Mass density of layers (kg/m^3)
% model.thk    = [10 inf];      % depth of layers [m]
% Qs=[30 30];
% Qp=[30 30];
% model.Damp_p =[0 0] ;%1./(2*Qp); % damping
% model.Damp_s = [0 0];%1./(2*Qs);

%% two-layer model elastic
model.vs   = [220 440];   % Shear wave velocity [m/s]
model.vp   =[380 760];
model.dns  = [1 1 ]*2000;    % Mass density of layers (kg/m^3)
model.thk    = [10 inf];      % depth of layers [m]
model.Damp_p = [0 0];
model.Damp_s = [0 0];


%% time and source setting
source.number=20; 
source.recordlength=20; % record time(s)
source.dt=0.005; %  time step
source.maxfre=20;  %% max frequency to be calculated
source.fmin=3;  %% min. Dominant Frequency of Ricker wavelet
source.fmax=18;  %% max. Dominant Frequency of Ricker wavelet

%%%%%%%%%%%%  set point loads   %%%%%%%%%%%%%%%%%%%%%
flag='v';

if any(strcmp(flag,{'v','V'}))
    source.Fx=zeros(1,source.number); % point load in x-axis
    source.Fy=zeros(1,source.number); % point load in y-axis
    source.Fz=-rand(1,source.number);% point load in z-axis,z-axis is positive in the upward direction
elseif any(strcmp(flag,{'h','H'}))
    source.Fx=rand(1,source.number); % point load in x-axis
   % source.Fx=zeros(1,source.number); % point load in x-axis
    source.Fy=rand(1,source.number); % point load in y-axis
    source.Fz=zeros(1,source.number);% point load in z-axis,z-axis is positive in the upward direction
elseif any(strcmp(flag,{'hv','HV','Hv','hV'}))
    source.Fx=rand(1,source.number); % point load in x-axis
    source.Fy=rand(1,source.number); % point load in y-axis
    source.Fz=-rand(1,source.number);% point load in z-axis,z-axis is positive in the upward direction
end

%% Geometry setting
rng(42,'twister');
SR=S_R_geometry(source.number,'Distribution','annulus', ...
    'InnerRadius',300,'OuterRadius',500, ...
    'ReceiverHalfLength',100,'ReceiverSpacing',5);
ambient_geometry_plot(SR,1)
%% do calculating

[traces,TT]=PassiveFW(model,source,SR,flag);

toc

%% plot traces
scal = 1;
figure;
wigb_normalized(traces.z,scal,SR.rx,traces.t)
title('Z Component');
ylabel('Time (s)');
zlabel('station loc (m)');

%% do the cross-correlation





% %% save traces as sac files
% sacfolder = './sac';
% if ~exist(sacfolder, 'dir')
%     mkdir(sacfolder)
% end
% rec_num = length(SR.rx); % number of receivers
% origin_lat = 30;
% origin_lon = 100;
% for i = 1:rec_num
%     fprintf('Processing receiver %d of %d\n', i, rec_num);
%     % Call the subfunction to save the traces
%     save_sac_traces(SR, sacfolder, i, origin_lat, origin_lon, source.dt, traces);
% end
