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
model.vs   = [300 180 450];   % Shear wave velocity [m/s]
model.vp   =[600 1700 2000];
model.dns  = [2 2 2];    % Mass density of layers (kg/m^3)
model.thk  = [20 20 inf];      % depth of layers [m]
model.Damp_s=[0 0 0];  %  viscoealstic damping
model.Damp_p=[0 0 0];  % A zero represents elastic media.

%% two-layer model
% model.vs   = [220 440];   % Shear wave velocity [m/s]
% model.vp   =[380 760];
% model.dns  = [1 1 ]*2000;    % Mass density of layers (kg/m^3)
% model.thk    = [10 inf];      % depth of layers [m]
% Qs=[30 30];
% Qp=[30 30];
% % model.Damp_p = 1./(2*Qp); % damping
% % model.Damp_s = 1./(2*Qs);
% model.Damp_p =[0 0] ;%1./(2*Qp); % damping
% model.Damp_s = [0 0];%1./(2*Qs);

%% time and source setting
source.number=1; 
source.recordlength=5; % record time(s)
source.dt=0.005; %  time step
source.maxfre=100;  %% max frequency to be calculated
source.fmin=5;  %% min. Dominant Frequency of Ricker wavelet
source.fmax=30;  %% max. Dominant Frequency of Ricker wavelet

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
% change the source and reciever geometry inside this function
SR=S_R_geometry(source.number); 
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


% %% Save the traces into sac files
% sacfolder = './sac';
% if ~exist(sacfolder, 'dir')
%     mkdir(sacfolder)
% end
% 
% rec_num = length(SR.rx); % number of receivers
% origin_lat = 30;
% origin_lon = 100;
% 
% for i = 1:rec_num
%     sacfile = readsac(['/home/user/softwares/dastomo_matlab_linux/DASAutotomo1.2/src_app','/pulse.sac']);
%     fprintf('Processing receiver %d of %d\n', i, rec_num);
%     % Calculate receiver latitude and longitude using more accurate functions
%     [lat, lon] = reckon(origin_lat, origin_lon, km2deg(SR.ry(i) / 1000), azimuth(origin_lat, origin_lon, origin_lat, origin_lon + SR.rx(i) / 111320));
% 
%     % Save x component
%     filename_x = fullfile(sacfolder, sprintf('rec_%d_x.SAC', i));
%     sacfile.FILENAME = filename_x;
%     sacfile.B = 0;
%     sacfile.DELTA = source.dt;
%     sacfile.NPTS = length(traces.t);
%     sacfile.DIST = sqrt(SR.rx(i)^2 + SR.ry(i)^2);
%     sacfile.STLA = lat;
%     sacfile.STLO = lon;
%     sacfile.IFTYPE = 'ITIME';
%     sacfile.DATA1 = traces.x(:, i);
%     sacfile.LCALDA = 1;
%     sacfile.DEPMIN = min(sacfile.DATA1);
%     sacfile.DEPMAX = max(sacfile.DATA1);
%     writesac(sacfile);
%     sacfile.DATA1 = [];
% 
%     % Save y component
%     filename_y = fullfile(sacfolder, sprintf('rec_%d_y.SAC', i));
%     sacfile.FILENAME = filename_y;
%     sacfile.DATA1 = traces.y(:, i);
%     sacfile.DEPMIN = min(sacfile.DATA1);
%     sacfile.DEPMAX = max(sacfile.DATA1);
%     writesac(sacfile);
%     sacfile.DATA1 = [];
% 
%     % Save z component
%     filename_z = fullfile(sacfolder, sprintf('rec_%d_z.SAC', i));
%     sacfile.FILENAME = filename_z;
%     sacfile.DATA1 = traces.z(:, i);
%     sacfile.DEPMIN = min(sacfile.DATA1);
%     sacfile.DEPMAX = max(sacfile.DATA1);
%     writesac(sacfile);
%     sacfile.DATA1 = [];
% end