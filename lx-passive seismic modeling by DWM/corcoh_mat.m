% This scripts calculate coherency and correlation
function evelist =  corcoh_mat(traces, SR ,Para)
% SR: source-receiver geometry, contains the information of source and receivers, see S_R_geometry.m
% SR.rx and SR.ry: receiver locations
% traces: traces of the receivers,
% Para: parameters
% outcohdir = Para.outcohdir;  % output directory of coherency
outcordir = Para.outcordir;  % output directory of correlation
% if ~exist(outcohdir,'dir')
%     mkdir(outcohdir);
% end
if ~exist(outcordir,'dir')
    mkdir(outcordir);
end

twin = Para.twin;  % time window (s)
tlap = Para.tlap;   % overlap signal (s)
dt = Para.dt;   % time sampling (s)
tflag = Para.tflag;  % 保留到互相关长度到一半，不要令其等于twin而要小于它，否则不适合计算ntmn和ntmx
fmn = Para.fmn; fmx = Para.fmx;  % freuqncy band of bandpass filtering
np1 = Para.np1;
np2 = Para.np2;
savecohflag = Para.savecohflag;  % savecohflag = 0, save both real and imag part, others not.
dfcoh = Para.dfcoh;  % (Hz), frequency interval of coherency to save, dfcoh = 0 indicates using raw df.
flag1 = Para.flag1;     % 1.Detrend. 2.Demean. 3.Prefiltering. 4.Time-norm. 5.Spec-norm.
flag2 = Para.flag2;  % 1.Cal coh? 2.Save coh of every hour? 3.Cal cor? 4.Save cor of every hour?
winfun = Para.winfun; r = Para.r; % windows function
srcdir = Para.srcdir;

nw = round(twin/dt);
fs = 1/dt;  f = (0:nw-1)*(fs/nw);  df = f(2) - f(1);
if(dfcoh==0)
    dfcoh = df;
end
ndfcoh = round(dfcoh/df);

nfmn = round(fmn/df)+1;  nfmx = round(fmx/df)+1;  nwf = nfmx - nfmn + 1;
ntmn = nw - round(tflag/dt);  ntmx = nw + round(tflag/dt);  nwt = ntmx - ntmn + 1;

if isempty(gcp('nocreate'))
    parpool('local');
end

parfor ie = 1:ne
    % Initialize parallel pool if not already started

    % parfor (ie=1:ne,8)
    % for ie=1:ne
    %% coherence dir is the same as the correlation dir
    cordirm = [outcordir,'/',evelist(ie).name];
    cohdirm= cordirm;
    disp(strcat('Making the one minute CCF of ',{' '},num2str(ie),'/',num2str(ne)));
    %     clear lat lon net stnm
    %     clear swin;
    %     lat = []; lon = []; net = []; stnm = [];
    swin = [];
    for ic=1:nc
        %         filelist = dir([inevedir,'/',evelist(ie).name,'/*',cmp{ic},'*.SAC']);
        %         ns = length(filelist); if(ns<=1); continue; end
        %     parfor (is=1:ns,10)
        ns = length(stnm); if(ns<=1); continue; end
        %         k = 1;
        for is=1:ns
            %             filename = [inevedir,'/',evelist(ie).name,'/',filelist(is).name];
            filename = [inevedir,'/',evelist(ie).name,'/',net{is},'_',stnm{is},'.',cmp{ic},'.SAC'];
            sacfile = readsac(filename);
            s = sacfile.DATA1;
            %             net{is} = sacfile.KNETWK;
            %             stnm{is} = sacfile.KSTNM;
            %             lat(is) = sacfile.STLA;
            %             lon(is) = sacfile.STLO;
            %             index = find(ismember(netref,net{is})&ismember(stnmref,stnm{is}));
            %             if isempty(index)
            %                 continue;
            %             end
            %             lat(k) = latref(index);
            %             lon(k) = lonref(index);
            s = signalcut(s,dt,twin,tlap,winfun,r);
            swin(:,:,is,ic) = antpre(s,dt,fmn,fmx,np1,np2,flag1,flag2); % do pre-process for every window
            %             k = k+1;
        end
        [nw,nn,n0,~] = size(swin);
        swin(:,:,n0+1:ns,ic) = zeros(nw,nn,ns-n0);  % auto-completion station
        % swin: data in one window, number of windows, number of stations,
        % number of correlation components (1 for RR for DAS data)
        %% demedian for DAS common mode noise
        % need to be modified!!! This median is merely the median of the
        % very 1 minute HDF5 file
        if flag_rmmedian == 1 % xin liu, 2024/02/06
            swin_median = median(swin,3);
            swin = swin - swin_median;
        end
    end
    % reorder signals based on station list in sta.loc
    %     fid = fopen(inloc,'r');
    %     AAA = textscan(fid,'%s%s%f%f%f');
    %     fclose(fid);
    %     stnmref = AAA{1}; netref = AAA{2};
    %     swintmp = swin; lattmp = lat; lontmp = lon; nettmp = net; stnmtmp = stnm;
    % %     swintmp = swin; lat = AAA{3}; lattmp = lat; lon = AAA{4}; lontmp = lon; nettmp = net; stnmtmp = stnm;
    %     k = 1;
    %     for i=1:length(stnmref)
    %         index = find(ismember(stnm,stnmref{i})&ismember(net,netref{i}));
    %         if isempty(index)
    %             continue;
    %         end
    %         swintmp(:,:,k,:) = swin(:,:,index,:);
    %         lattmp(k) = lat(index);
    %         lontmp(k) = lon(index);
    %         nettmp{k} = net{index};
    %         stnmtmp{k} = stnm{index};
    %         k = k+1;
    %     end
    %     swin = swintmp;
    %     lat = lattmp;
    %     lon = lontmp;
    %     net = nettmp;
    %     stnm = stnmtmp;
    %     clear swintmp lattmp lontmp nettmp stnmtmp;

    %
    for i=1:ns
        %         sacfile(i) = sachd();
        sacfile(i) = readsac([srcdir,'/pulse.sac']);
    end
    %     parfor (i=1:ns-1,5)
    for i=1:ns-1
        %     for i=1:ns
        %         i
        nmi = [net{i},'_',stnm{i}];
        for j=i+1:ns
            %         for j=i
            for ic1=1:nc
                for ic2=1:nc
                    [arclen,az] = distance(lat(i),lon(i),lat(j),lon(j));
                    dist = deg2km(arclen);
                    nmj = [net{j},'_',stnm{j}];

                    % calculate coherency
                    if flag2(1)&&flagnez(ic1,ic2)
                        %                         roua = real(coherency(swin(:,:,i,ic1),swin(:,:,j,ic2)));
                        %                         roua = coherency(swin(:,:,i,ic1),swin(:,:,j,ic2));
                        roua0 = coherency(swin(:,:,j,ic1),swin(:,:,i,ic2)); %
                        roua0 = sum(roua0,2)/nn;
                        roua = roua0(ntmn:ntmx);
                        %                         roua = roua(nfmn:ndfcoh:nfmx);  % this step should be attention! (2014/04/9)
                    end
                    % calculate correlation
                    if flag2(3)&&flagnez(ic1,ic2)
                        cor = 0;
                        for l=1:nn
                            %                             cor0 = xcorr(swin(:,l,i,ic1),swin(:,l,j,ic2));
                            cor0 = xcorr(swin(:,l,j,ic1),swin(:,l,i,ic2));
                            %                             cor0 = xcorr(swin(:,l,j,ic1),swin(:,l,i,ic2))./(norm(swin(:,l,j,ic1)*norm(swin(:,l,i,ic2))));
                            cor = cor + cor0(ntmn:ntmx);
                        end
                    end
                    %                     % save coherency of one hour result
                    %                     if flag2(1)&&flag2(2)&&flagnez(ic1,ic2)
                    %                         filename = [cohdirm,'/COH','_',nmi,'_',nmj,'.',cmp{ic1}(end),cmp{ic2}(end),'.SAC'];
                    %                         sacfile(i).FILENAME = filename;
                    %                         sacfile(i).B = f(nfmn);
                    %                         sacfile(i).DELTA = df*ndfcoh;
                    %                         sacfile(i).NPTS = length(roua);
                    %                         sacfile(i).DIST = dist;
                    %                         sacfile(i).EVLA = lat(i);
                    %                         sacfile(i).EVLO = lon(i);
                    %                         sacfile(i).STLA = lat(j);
                    %                         sacfile(i).STLO = lon(j);
                    %                         sacfile(i).LCALDA = 1;
                    %                         if(savecohflag==0)
                    %                             sacfile(i).IFTYPE = 'IRLIM';
                    %                             sacfile(i).DATA1 = real(roua);
                    %                             sacfile(i).DATA2 = imag(roua);
                    %                             writesac(sacfile(i));
                    %                             sacfile(i).DATA1 = []; sacfile(i).DATA2 = [];
                    %                         else
                    %                             sacfile(i).IFTYPE = 'ITIME';
                    %                             sacfile(i).DATA1 = real(roua);
                    %                             writesac(sacfile(i));
                    %                             sacfile(i).DATA1 = [];
                    %                         end
                    %                     end
                    % save coherency of one hour result
                    if flag2(1)&&flag2(2)&&flagnez(ic1,ic2)
                        filename = [cordirm,'/COR','_',nmi,'_',nmj,'.',cmp{ic1}(end),cmp{ic2}(end),'.SAC'];
                        sacfile(i).FILENAME = filename;
                        sacfile(i).B = -tflag;
                        sacfile(i).DELTA = dt;
                        sacfile(i).NPTS = length(roua);
                        sacfile(i).DIST = dist;
                        sacfile(i).EVLA = lat(i);
                        sacfile(i).EVLO = lon(i);
                        sacfile(i).STLA = lat(j);
                        sacfile(i).STLO = lon(j);
                        sacfile(i).IFTYPE = 'ITIME';
                        sacfile(i).DATA1 = roua;
                        sacfile(i).LCALDA = 1;
                        writesac(sacfile(i))
                        sacfile(i).DATA1 = [];
                    end
                    % save correlation of one hour result
                    if flag2(3)&&flag2(4)&&flagnez(ic1,ic2)
                        filename = [cordirm,'/COR','_',nmi,'_',nmj,'.',cmp{ic1}(end),cmp{ic2}(end),'.SAC'];
                        sacfile(i).FILENAME = filename;
                        sacfile(i).B = -tflag;
                        sacfile(i).DELTA = dt;
                        sacfile(i).NPTS = length(cor);
                        sacfile(i).DIST = dist;
                        sacfile(i).EVLA = lat(i);
                        sacfile(i).EVLO = lon(i);
                        sacfile(i).STLA = lat(j);
                        sacfile(i).STLO = lon(j);
                        sacfile(i).IFTYPE = 'ITIME';
                        sacfile(i).DATA1 = cor;
                        sacfile(i).LCALDA = 1;
                        writesac(sacfile(i))
                        sacfile(i).DATA1 = [];
                    end
                end
            end
        end
    end

    %     disp([evelist(ie).name,' complete.'])
    %     toc
end
% delete(gcp)


function temp = signalcut(s,dt,twin,tlap,winfun,r)
    % temp = signalcut(s,dt,twin,tlap,winfun,r)
    % cut signal into sub-signal
    % input:
    %       s           time signal
    %       dt          time sampling
    %       twin        windowed time signal length
    %       tlap        overlap between two neighbouring sub-signal
    %       winfun      window function to taper
    %       r           window function coefficients
    %
    % By Luo Song, 2013/10/25

    if (nargin <= 4); winfun = 'tukeywin'; r=0; end;
    if (nargin <= 3); tlap = 0; end;
    if (nargin <= 2); disp('Error input in signalcut.m !'); end;

    n = floor(twin/dt);
    nlap = floor(tlap/dt);
    k = n - nlap;
    N = length(s);

    w = window(winfun,n,r);
    i = 1;
    temp = [];
    while i*k+nlap<=N
        temp(:,i) = w.*s((i-1)*k+1:i*k+nlap);
        i = i + 1;
    end

function s = antpre(swin,dt,fmn,fmx,np1,np2,flag,flag2)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,nn] = size(swin);
    swin = swin';

    for i=1:nn

        s = swin(i,:);

        % detrend ------------------------------------------------------------%
        if flag(1)
            s = detrend(s);
        end

        % demean -------------------------------------------------------------%
        if flag(2)
            s = rmean(s);
        end

        % pre-filtering ------------------------------------------------------%
        if flag(3)
            s = bandpass_butter(s,fmn,fmx,dt);
        end

        % temporal normalization --------------------------------------------%
        % use this temporal normalization only for cross correlation (Xin, 2024.03)
        if flag(5) && flag2(3)
            %     if flag(4)
            s = rnabmn(s,np1);
        end

        % spectral normalization ---------------------------------------------%
        % use this spectral normalization only for cross correlation (Xin, 2024.03)
        %% however, cross-coherency with de-median and not doing this spectrum whitening
        % is causing some spurious vertical strips around zero-time lag. I
        % haven't figured out why yet.
        if flag(5) && flag2(3)
            %     if flag(5)
            s = whitening(s,np2);
        end

        % filtering ----------------------------------------------------------%
        if all(flag)
            s = fftshift(s);
            s = bandpass_butter(s,fmn,fmx,dt);
            s = ifftshift(s);
        end

        swin(i,:) = s;
    end
    s = swin';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function song = rmean(s)
    % remove mean value
    sm = mean(s);
    song = s - sm;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function song = rnabmn(s,np)
    % running-absolute-mean normalization, refer to Bensen (2007)
    % input:
    %       np ---- number of point involved in normalization
    % the fast method (2013/10/31) refer to Yang yingjie program
    om = smooth(abs(s),np);
    om = om';

    flag = find(om==0);
    song = s./om;
    song(flag) = s(flag);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function song = whitening(s,np)
    % spectral whitening use running-absolute-mean normalization
    % input:
    %       np ---- number of point involved in normalization
    sfft = fft(s);
    am = abs(sfft);
    am = smooth(am,np);
    w = 0.01*mean(am);
    am(am<=w) = w;
    sfft = sfft./am';  % (2013/11/1) refer to Yang yingjie program
    song = ifft(sfft,'symmetric');  % note: the 'symmetric' must be added
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function song = onebit(s)
    % % one-bit normalization, refer to Bensen (2007)
    % s(s>0) = 1;
    % s(s<0) = -1;
    % song = s;
    %
    %
    % function song = revent(s,dt,rt,c)
    % % remove event, refer to Bensen (2007)
    % % input:
    % %       rt ---- remove time length after a event been detected
    % %       c ---- coefficients, generally, c=2.5
    %
    % n = length(s);
    % A = abs(s);
    % rtn = round(rt/dt) + 1;
    % rms = sqrt(sum(s.^2)/n);
    % flag = find(A>c*rms,1);
    %
    % if isempty(flag)
    %     song = s;
    %     return
    % else
    %     while isempty(flag)==0
    %         if flag+rtn<=n
    %             s(flag:flag+rtn) = 0;
    %         else
    %             s(flag:n) = 0;
    %         end
    %         A = abs(s);
    %         flag = find(A>c*rms,1);
    %     end
    %     song = s;
    %     return
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % function song = clipping(s,c)
    % % the application of a clipping threshold equal to the c*rms amplitude of
    % % the signal
    % n = length(s);
    % rms = sqrt(sum(s.^2)/n);
    %
    % s(s>c*rms) = c*rms;
    % s(s<-c*rms) = -c*rms;
    % song = s;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




