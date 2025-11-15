function csvwrite_with_headers(filename, data, headers)
    % Open the file in write mode
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file for writing.');
    end
    
    % Write the header
    fprintf(fid, '%s,', headers{1, 1:end-1});
    fprintf(fid, '%s\n', headers{1, end});
    
    % Write the data
    for i = 1:size(data, 1)
        fprintf(fid, '%f,', data(i, 1:end-1)); % write each data row
        fprintf(fid, '%f\n', data(i, end));    % write last column without a trailing comma
    end
    
    % Close the file
    fclose(fid);
end
