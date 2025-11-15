function [Frq,Phi]=FDD(Input,Fs,pp_range)

% Frequency Domain Decomposition (FDD) algorithm for modal analysis
% This code allows you to manually select the peaks by simply drawing a
% rectangle around the peaks.
% Original version (without damping ratio identification):
% Programmer: Mohammad Farshchin, Ph.D candidate at The UofM
% Email: Mohammad.Farshchin@gmail.com

% Modified version (add damping ratio identification):
% Programmer: Xudong JIAN, postdoc at Singapore-ETH Centre
% Email: xudong.jian@sec.ethz.ch

% Input: each column is one measurement
% Fs: sampling frequency
% pp_range: frequency range for peak picking
% Frq: identified frequencies
% Phi: identified mode shapes

% For detailed information about frequency and mode shape identification, see:
% Brincker R, Zhang LM, Andersen P. Modal identification from ambient responses using Frequency Domain Decomposition. In: Proceedings of the 18th International Modal Analysis Conf., USA: San Antonio, 2000.

% For detailed information about damping identification, see:
% R. Brincker, C. E. Ventura and P. Andersen. Damping estimation by frequency domain decomposition.

% -------------------------------------------------------------------------
% Initialization
% -------------------------------------------------------------------------
% Import time history data: Processed accelereation data must be
% arranged in a columnwise format (one column for each measurement channel)
% Note that the acceleration data must be preprocessed (detrend, filtered etc.).
% Read acceleration data from the excel file
Acc=Input;
% -------------------------------------------------------------------------
% Compute Power Spectral Density (PSD) matrix.
% CPSD function, with default settings, is used to compute the cross power
% spectral density matrix. More sophisticated methods can also be
% applied for higher accuracy.
nfft = 1024;
for I=1:size(Acc,2)
    for J=1:size(Acc,2)
        [PSD(I,J,:),F(I,J,:)]=cpsd(Acc(:,I),Acc(:,J),hamming(nfft/4),nfft/8,nfft,Fs);
    end
end
Frequencies(:,1)=F(1,1,:);
% -------------------------------------------------------------------------
% Perform Modal Analysis (Use the Identifier function, below)
[Frq,Phi] = Identifier(PSD,Frequencies,pp_range);
% -------------------------------------------------------------------------

end

%% ---------------------------- subfunctions ------------------------------
function [Frq,Phi] = Identifier(PSD,Frequencies,pp_range)

% Compute SVD of the PSD at each frequency
for I=1:size(PSD,3)
    [u,s,~] = svd(PSD(:,:,I));
    s1(I) = s(1);           % First singular values
    s2(I) = s(2,2);         % Second singular values
    u1(:,I) = u(:,1);         % Mode shape: first singular vectors
end

NumModes = length(pp_range(:,1));
locs = zeros(NumModes,1);
for i = 1:NumModes
    [~,I1] = min(abs(Frequencies-pp_range(i,1)));
    [~,I2] = min(abs(Frequencies-pp_range(i,2)));
    [~,locs(i)] = max(s1(I1:I2));
    locs(i) = locs(i)+I1-1;
end

% Identified modal frequencies, which are Frequencies related to selected peaks
Frq=Frequencies(locs);

% Identify mode shape for each selected peak
Phi = zeros(length(s),length(Frq));
for i=1:length(locs)
    Phi(:,i) = u1(:,locs(i));
end

end