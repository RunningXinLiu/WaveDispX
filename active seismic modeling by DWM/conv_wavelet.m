
function  ut2=conv_wavelet(t,ut,fm)
% t: time vector row
% fm: main frequency row
% ut: signal, one colum represents one trace
[m,n]=size(ut);
if length(fm)==1
   fm=ones(1,n)*fm; 
end
dt=t(2)-t(1);
t5=(0:m-1)*dt;
pz2=rickerkai(t5,fm);
nfft=2^nextpow2(2*m-1);
ut_linear=real(ifft(fft(ut,nfft,1).*fft(pz2,nfft,1),[],1));
ut2=ut_linear(1:m,:);




