function [MOD_f_unq, MOD_u_unq, MOD_z_unq,h_combMAC] = getUniqueModes5(MOD_f,MOD_u,MOD_z, MAC_thr,tot_thr,freq_thr_min,freq_thr_max,mac_title,plt)
    %% Get unique modes for each dataset
% %%%%% INPUT
% MAC_thr=0.80; % MAC comparison threshold 
% tot_thr=0.80; % Combined MAC and freq comparison
% freq_thr=3; % Hig pass limit to discard modes with lower freq [Hz] 
% plt_unq=1; % 1 to plot the MAC plot of the unique modes
% %%%%%

    fr=MOD_f;
    
    % Exclude modes with frequencies below the threshold
    c3= find(fr<freq_thr_min|fr>freq_thr_max);
    mod=MOD_u;
    mod(c3,:,:)=[];
    fr(c3)=[];
    z=MOD_z;
    z(c3)=[];
    n_modes=length(fr);
    n_dir=size(mod,3);    

    % Get MAC matrix for this dataset
    phi1=MOD2phi2(mod);
    phi2=phi1;

    mac=MAC(phi1',phi2',mac_title,0);

    ii=1;
    while ii<=n_modes % for each identified mode
        freq_ref=fr(ii);
        c2 = find(mac(ii,:)>MAC_thr); % Find identified modes with similar Macs to mode ii
        c2(c2<ii)=[];

        n_modes_mac=length(c2); % Number of modes with similar Mac to mode ii

        % Calculate combined fit criterion
        for iii=1:n_modes_mac 
            %ft_tot(iii)=(1-abs(freq_ref-fr(c2(iii)))./freq_ref)*0.5+0.5*mac(ii,c2(iii)); % fit vector for all modes that have similar mac
            ft_tot(iii)=((1-min(1,abs(freq_ref-fr(c2(iii)))./freq_ref))*0.5+0.5*mac(ii,c2(iii)))^2;% fit vector for all modes that have similar mac
        end
        c3= find(ft_tot>tot_thr);
        clearvars ft_tot
        last_fit=c2(c3(end));
        n_fits= last_fit-c2(1)+1;

        % Get a vector with the indexes of unique modes
        if ii==1
            mod_unique=c2(1);
        else
            mod_unique=[mod_unique;c2(1)];
        end      
        ii=ii+n_fits;    
    end

    % Get data with unique modes
    MOD_u_unq = phi2MOD(phi1(mod_unique,:),n_dir);
    MOD_f_unq=fr(mod_unique);
    MOD_z_unq=z(mod_unique);

    if plt==1
       %mac=MAC(phi1',phi2',plt);
       fr1=fr;
       fr2=fr1;
       combMAC(phi1',phi2',fr1,fr2,0);
       phi1=MOD2phi2(MOD_u_unq);
       phi2=phi1;
       fr1=MOD_f_unq;
       fr2=fr1;
       %mac=MAC(phi1',phi2',plt);
       [~,h_combMAC]=combMAC(phi1',phi2',fr1,fr2,plt);
    else 
        h_combMAC=0;
    end
    


    
    
end

