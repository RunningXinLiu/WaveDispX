clear;
tic
script_dir=fileparts(mfilename('fullpath'));
repository_root=fileparts(script_dir);
addpath(repository_root, script_dir);
%%  half-space model
% Damp_p=1./(2*Qp); % damping
% Damp_s=1./(2*Qs);

% model.vs   = 440;   % Shear wave velocity [m/s]
% model.vp   =760;
% model.dns  = 2;    % Mass density of layers (kg/m^3)
% model.thk  = inf;      % depth of layers [m]
% model.Damp_s=0;  %  viscoealstic damping
% model.Damp_p=0;  % A zero represents elastic media.

%% two-layer model
model.vs   = [220 440];   % Shear wave velocity [m/s]
model.vp   =[380 760];
model.dns  = [1 1 ]*2000;    % Mass density of layers (kg/m^3)
model.thk    = [10 inf];      % depth of layers [m]
Qs=[30 30];
Qp=[30 30];
model.Damp_p =[0 0] ;%1./(2*Qp); % damping
model.Damp_s = [0 0];%1./(2*Qs);

%%  three-layer model
% model.vs   = [300 180 450];   % Shear wave velocity [m/s]
% model.vp   =[600 1700 2000];
% model.dns  = [2 2 2 ];    % Mass density of layers (kg/m^3)
% model.thk  = [20 20 inf];      % depth of layers [m]
% model.Damp_s=[0 0 0];  %  viscoealstic damping
% model.Damp_p=[0 0 0];  % A zero represents elastic media.

%%  five-layer model   %  mi binbin
% vs = [270 367 125 453 540];   % Shear wave velocity [m/s]
% vp =[750 1400 550 1600 1800];
% dns = [1.86 1.91 1.96 2.02 2.09];    % Mass density of layers (kg/m^3)
% thk = [3 3 4 4 inf];      % depth of layers [m]
% Damp_s =[0 0 0 0 0];  %  viscoealstic damping
% Damp_p =[0 0 0 0 0];  % A zero represents elastic media.

source.fm=20;  %% source main frequency
source.dt=0.001;  %% time step
source.offset=1:200;  %% receivers position
source.maxfre=100;  %% max frequency to be calculated

flag='hv';
[ut,tt]=active_modeling_DWM(model,source,flag);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot shot gathers
flag2=5;
wigb_normalized(ut(:,:,flag2),2,source.offset,tt);
close(gcf); % keep batch runs from exhausting graphics resources

%% dispersion imaging

load(fullfile(script_dir,'Rayleigh_Love_DC.mat'),'vr','vl');
vv=100:500;
[p1, f1]=surface_wave_dispersion(ut(:,:,5),source.offset,source.dt,vv,50, ...
    'Method','hankel','Component','vertical','ZeroPadding',5,'Plot',false);

[p2, f2]=surface_wave_dispersion(ut(:,:,2),source.offset,source.dt,vv,50, ...
    'Method','hankel','Component','transverse','ZeroPadding',5,'Plot',false);

rayleigh_picks=pick_dispersion_ridge(p1,f1,vv, ...
    'FrequencyRange',[5 50],'VelocityRange',[150 450], ...
    'Smoothness',0.3,'MaxJump',25,'MinimumEnergy',0.05);
love_picks=pick_dispersion_ridge(p2,f2,vv, ...
    'FrequencyRange',[5 50],'VelocityRange',[200 460], ...
    'Smoothness',0.3,'MaxJump',25,'MinimumEnergy',0.05);

rayleigh_figure=plot_active_surface_wave_summary(model,source,tt,ut(:,:,5), ...
    p1,f1,vv,vr,rayleigh_picks,'Rayleigh');
love_figure=plot_active_surface_wave_summary(model,source,tt,ut(:,:,2), ...
    p2,f2,vv,vl,love_picks,'Love');

run_id=char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
output_folder=fullfile(script_dir,'output',['active_',run_id]);
mkdir(output_folder);
save(fullfile(output_folder,'active_twolayer_results.mat'), ...
    'model','source','ut','tt','p1','f1','p2','f2','vr','vl', ...
    'rayleigh_picks','love_picks','-v7.3');
writetable(rayleigh_picks,fullfile(output_folder,'picked_rayleigh.csv'));
writetable(love_picks,fullfile(output_folder,'picked_love.csv'));
exportgraphics(rayleigh_figure,fullfile(output_folder,'rayleigh_dispersion.png'), ...
    'Resolution',180);
exportgraphics(love_figure,fullfile(output_folder,'love_dispersion.png'), ...
    'Resolution',180);
fprintf('Active results: %s\n',output_folder);

toc
