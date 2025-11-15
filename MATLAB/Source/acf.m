function rk = acf(x,maxlag,barsize)

% ACF computation
R = xcorr(x,maxlag,'coeff');

% Truncating to positive lags
rk = R(maxlag+2:end);

if nargout==0
    figure,
    bar([1:maxlag],rk,barsize,'b'),hold on
    plot([1:maxlag],(1.96/sqrt(length(x))).*ones(maxlag,1),'r',...
        [1:maxlag],(-1.96/sqrt(length(x))).*ones(maxlag,1),'r')
    axis([0 maxlag+1 -1 1])
    xlabel('Lag'),ylabel('A.C.F. ( \rho_\kappa )'),zoom on,hold off
end