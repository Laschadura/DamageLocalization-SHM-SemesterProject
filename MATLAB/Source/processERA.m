function [freq, phi, damp_rel] = processERA(output, ref, inptype, input, fs, Nfft, nch, ndof, order)
    % processERA: Computes Impulse Response and performs ERA analysis.
    %
    % Inputs:
    %   - output : System output signals
    %   - ref : Reference channel for cross-correlation
    %   - inptype : Input type ('imp', 'WN', or known input case)
    %   - input : Input signal (if known input case)
    %   - fs : Sampling frequency
    %   - Nfft : Number of FFT points
    %   - nch : Number of channels
    %   - ndof : Number of degrees of freedom
    %   - order : Model order for ERA
    %
    % Outputs:
    %   - freq : Identified frequencies
    %   - phi : Corresponding modal shapes
    %   - YY : Impulse response function
    %   - TT : Time vector for impulse response
    %   - freq_rel : Identified relative frequencies
    %   - damp_rel : Identified damping ratios
    %   - modal_shapes : Modal shapes
    %
    % Author: Eleni Chatzi

    switch inptype
        case 'imp'
            YY = output; % Already impulse response
            dt = 1/fs;
            
        case 'WN'
            maxlag = Nfft;
            for ii = 1:nch
                % Cross-correlation and CSD
                Rxy(:, ii) = xcorr(output(:, ii), output(:, ref), maxlag, 'coeff');
                [csdxy(:, ii), Fxy] = cpsd(output(:, ref), output(:, ii), [], [], Nfft, fs);

                % Mirroring with complex conjugate
                df = Fxy(2) - Fxy(1);
                CSD(:, ii) = [csdxy(:, ii); conj(flipud(csdxy(1:end-1, ii)))];
                dt = 1 / (df * Nfft);
                YY(:, ii) = real(ifft(CSD(:, ii), Nfft)) / dt;

                % Truncate to positive lags
                RY(:, ii) = Rxy(maxlag + 2:end, ii);
            end
            
        otherwise
            % Known input case
            % Normalize input and output
            norm_factor = max(abs(input));
            input_norm = input / norm_factor;
            output_norm = output / norm_factor;

            % Compute CSD and PSD
            [csdxx, Fxx] = cpsd(input_norm, input_norm, [], [], Nfft, fs);
            for ii = 1:nch
                [csdxy(:, ii), Fxy] = cpsd(input_norm, output_norm(:, ii), [], [], Nfft, fs);
            end

            % Transfer function H = Sxy / Sxx
            for ii = 1:nch
                Hxy(:, ii) = csdxy(:, ii) ./ csdxx;
            end

            % Mirroring with complex conjugate
            for ii = 1:nch
                Htemp(:, ii) = [Hxy(:, ii); conj(flipud(Hxy(1:end-1, ii)))];
            end

            % Impulse response function
            df = Fxy(2) - Fxy(1);
            dt = 1 / (df * Nfft);
            TT = 0:dt:(Nfft - 1) * dt;
            for ii = 1:nch
                YY(:, ii) = real(ifft(Htemp(:, ii), Nfft)) / dt;
            end
    end

    % Perform ERA
    [yx, freq_rel2, damp_rel, modal_shapes, freq_rel, ao, bo, co] = ...
        ERA(YY, dt, ref, 1, size(YY, 1), 2 * ndof, order);

    % Clear out spurious modes
    [C, ia] = unique(freq_rel2);
    freq = freq_rel2(setdiff(1:length(freq_rel2), ia));
    modal_shapes = modal_shapes(:, setdiff(1:length(freq_rel2), ia));

    % Sort frequencies and modal shapes
    [freq, ind] = sort(freq);
%     disp('Identified Frequencies (Hz):');
%     disp(freq);

    phi = real(modal_shapes(:, ind)); % Keep only real parts
end
