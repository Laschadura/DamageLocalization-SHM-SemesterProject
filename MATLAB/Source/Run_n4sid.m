function [freq_rel,modal_shapes,repeated_zeta] =  Run_n4sid(output,f,inptype,order,fs)
%=============================================================================================
% This is the main file for calling the N4SID function
dt = 1/fs;

switch inptype
    case 'imp'
        
     YY=output;      %This is already impulse response
     z = iddata(YY,[],dt);

    case 'WN'
        
     YY=output;      %n4sid now works as stochastic
     z = iddata(YY,[],dt);
    
    otherwise
    YY=output;   
    input=f;
    z = iddata(YY,input,dt);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Estimate the state-space model using SSI-Data with n4sid
sys = n4sid(z, order, 'Cov', 'none');
% sys = n4sid(z,order,'Ts',1/fs);
ao=sys.A; bo=sys.B; co=sys.C;

% Compute the eigenvalues (poles) and eigenvectors (mode shapes) of the A matrix
[Vectors, Values] = eig(ao);
Lambda = diag(Values);  % Eigenvalues in the Z-plane

% Convert eigenvalues to continuous time (frequency-domain)
s = log(Lambda) * fs;    % Eigenvalues in the continuous-time s-plane
zeta = -real(s) ./ abs(s);  % Damping factors
ind = find(zeta<=1);

fd = abs(imag(s)) / (2 * pi);   % Damped natural frequencies in Hz

fd = fd(ind);
Vectors = Vectors(:,ind);
zeta = zeta(ind);


% Find the repeated frequencies (within a small tolerance to account for numerical issues)
tolerance = 1e-6;  % Adjust this tolerance if necessary
[unique_frequencies, ~, indices] = uniquetol(fd, tolerance, 'ByRows', true);

% Count occurrences of each unique frequency
occurrences = histc(indices, unique(indices));

% Select only the frequencies that are repeated (occurrences > 1)
repeated_indices = find(occurrences > 1);  % Find the index of repeated frequencies
freq_rel = unique_frequencies(repeated_indices);

% Select the mode shapes corresponding to the repeated frequencies
repeated_mode_shapes = [];
repeated_zeta = [];

for i = 1:length(repeated_indices)
    freq_idx = repeated_indices(i);
    matching_modes = find(abs(fd - unique_frequencies(freq_idx)) < tolerance);
    repeated_mode_shapes = [repeated_mode_shapes, Vectors(:, matching_modes(1))];
    repeated_zeta = [repeated_zeta, zeta(matching_modes(1))];
end

modal_shapes=co*repeated_mode_shapes;