function [mac,h]=MAC(phi1,phi2,mac_title,plt)
    % This function is to compute and plot Modal Assurance Criterion (MAC) matrix between identified mode shapes
    % rectangle around the peaks.
    
    % phi: matrix of the identified mode shapes modal displ x modes
    % mac: MAC matrix
    for I=1:size(phi1,2)
        for J=1:size(phi2,2)
            mac(I,J)=Mac(phi1(:,I),phi2(:,J));
            if I==J
                aa=1;
            end
        end
    end
    % plot mac matrix
    % figure
    % bar3(mac)
    % title('MAC')
    
    if plt==1
        h=figure;
        b1=bar3(mac);
        %set(gca,'YTickLabel',[0  1 2 3.1 3.2 4])
        colorbar
        caxis([0 1])
        ylabel(mac_title(1))
        xlabel(mac_title(2))
        if length(mac_title)<3
            mac_title(3)="MAC";
        end
        title(mac_title(3))
        % set(get(h,'Title') ,'String','TAC Index');
        for k = 1:length(b1)
            zdata = b1(k).ZData;
            b1(k).CData = zdata;
            b1(k).FaceColor = 'interp';
        end
        
    
        view(270,90)
    else
        h=0;
    
end
    
    
    end
    
    function mAc=Mac(Phi1,Phi2)
    % This function calculates mac between phi1 and phi2
    mAc= (abs(Phi1'*Phi2))^2/((Phi1'*Phi1)*(Phi2'*Phi2));
end