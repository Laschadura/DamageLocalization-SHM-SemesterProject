function [th,modal] = runARMA(Y,Ts,nmin,nmax)

z = iddata(Y,[],Ts);
N = length(z.OutputData);
Fs = 1/Ts; 

figure(300),
hold on
xlabel('Frequency (Hz)')
ylabel('AR/MA order')
axis([0 Fs/2 nmin nmax+1])

for n = nmin:nmax
    th = armax(z,[n n]);
    e = pe(th,z);
    d = n+n;      
    bic_nn(n) = log(var(e.OutputData))+(d*(log(N)/N));
    RSS_nn(n) = sum(e.OutputData.^2);
    
    mu = roots(th.a);
    lambda = log(mu)/Ts;
    wn = abs(lambda)/(2*pi);
    zeta = -100*real(lambda)./abs(lambda);
    
    for p = 1:n
        if ~isreal(lambda)
            plot([wn(p) wn(p)],[n n+(1-zeta(p)/100)]);
        end
    end
end

figure(301)
subplot(2,1,1),plot(nmin:nmax,RSS_nn(nmin:nmax),'-o')
title('RSS of model ARMA(n,n)')
subplot(2,1,2),plot(nmin:nmax,bic_nn(nmin:nmax),'-o')
title('BIC of model ARMA(n,n)')
zoom on

ni = input('Give AR/MA order: ');

th = armax(z,[ni ni]);
% Re-fit the Model or Construct a Sparse Model
threshold = 0.1;  % Define a threshold for coefficient magnitude
sparse_A = th.A;  % Copy the coefficients
th.A(abs(sparse_A) < threshold) = 0;    % Set small coefficients to zero
sparse_C = th.C;  % Copy the coefficients
th.C(abs(sparse_C) < threshold) = 0;    % Set small coefficients to zero

e = pe(th,z);
present(th),
mu = roots(th.a);
lambda = log(mu)/Ts;
wn = abs(lambda)/(2*pi);
zeta = -100*real(lambda)./abs(lambda);
modal = [wn zeta];
modal = sortrows(modal,1);

figure(302)
subplot(2,2,3),histfit(e.OutputData(100:end),45),title('residuals')
subplot(2,2,4),normplot(e.OutputData(100:end))
subplot(2,1,1),plot([zeros(99,1) ;e.OutputData(100:end)])

acf(e.OutputData,200,.2);