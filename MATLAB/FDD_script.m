
% This script processes accelerometer data using Frequency Domain Decomposition (FDD)
% to extract modal parameters from vibration tests on a masonry arch bridge.
%
% The script performs the following steps:
% 1. Loads accelerometer data from vibration tests.
% 2. Applies high-pass filtering to remove low-frequency noise.
% 3. Identifies impact events based on peak detection.
% 4. Computes cross power spectral density (PSD) and applies Singular Value Decomposition (SVD).
% 5. Detects peaks in the spectral response to identify potential modal frequencies.
% 6. Uses the FDD method to extract mode shapes and evaluates their consistency using the MAC criterion.
% 7. Saves results, including mode shapes, frequencies, and MAC values, as CSV files and visualizations.
%
% Author: Dr. M. Farshchin
% Altered by: Dr. Xudong Jian, Simon Scandella

%% Initialization for Batch Processing
clc;
clear;

% Add the current folder and all its subfolders to the MATLAB path
addpath(genpath(pwd));

% Define the base folder path containing the test folders
base_folder = "DATA/VibrationTestData";
output_base_folder = "Results";

% Sampling rate and cutoff frequency
fs = 200;
cutoff = 10;
nfft = 1024;

% Lower threshold for peak detection and minimum MAC threshold for filtering
peak_detection_threshold = 0.01; % Detect more peaks with a lower threshold
MAC_median_threshold = 0.95;    % Minimum median MAC threshold to consider a mode as true

% Get a list of all folders in the base folder (tests 1 to 25 excluding 23 and 24)
test_folders = dir(fullfile(base_folder, 'Test_*'));
test_folders = test_folders(~ismember({test_folders.name}, {'Test_23', 'Test_24'})); % Remove Test_23 and Test_24

% Loop through each test folder
for test_idx = 1:length(test_folders)
    test_folder_name = test_folders(test_idx).name;
    test_folder_path = fullfile(base_folder, test_folder_name);
    
    % Define the output folder for this test (create folder if it doesn't exist)
    output_folder = fullfile(output_base_folder, test_folder_name);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % Find all CSV files in the current test folder
    csv_files = dir(fullfile(test_folder_path, '*.csv'));
    
    % Loop over each CSV file in the test folder
    for file_idx = 1:length(csv_files)
        file_name = csv_files(file_idx).name;
        file_path = fullfile(test_folder_path, file_name);
        
        % Display which file is being processed
        fprintf('Processing file: %s\n', file_name);
        
        file_parts = split(file_name, '_');  % Split by underscores
        test_num = file_parts{1};            % '1' (the test number is not needed)
        excitation_type = strjoin(file_parts(3:end), '_');  % Everything between the underscores except the last part
        
        % If you want to also remove the ".csv" part, you can trim the file extension
        [~, excitation_type_without_extension, ~] = fileparts(excitation_type);
        
        % Now create the folder
        excitation_folder = fullfile(output_folder, excitation_type_without_extension);
        if ~exist(excitation_folder, 'dir')
            mkdir(excitation_folder);
        end

        
        % Load the data
        data = readmatrix(file_path);
        acc = data(:, 2:end); % Assume first column is time or indexing

        % Apply high-pass filter to the acceleration data
        acc_highpass = highpass(acc, cutoff, fs, ImpulseResponse="iir", Steepness=0.95);

        % Find impact indices for peaks in the filtered data
        [~, impact_index] = findpeaks(acc_highpass(:, 3), ...
            'MinPeakHeight', max(acc_highpass(:, 3)) / 3, ...
            'MinPeakDistance', 200);

        % Initialize storage to collect first singular values for each segment
        s1_all = [];

        % Singular spectrum analysis for each segment
        num_segments = min(9, length(impact_index) - 1);
        for impact_no = 1:num_segments
            acc_highpass_seg = acc_highpass(impact_index(impact_no) - 40 : impact_index(impact_no) + 560, :);
            
            % Calculate the cross power spectral density (PSD) matrix for each segment
            for I = 1:size(acc_highpass_seg, 2)
                for J = 1:size(acc_highpass_seg, 2)
                    [PSD(I, J, :), F(I, J, :)] = cpsd(acc_highpass_seg(:, I), acc_highpass_seg(:, J), ...
                        hamming(nfft / 4), nfft / 8, nfft, fs);
                end
            end
            Frequencies = F(1, 1, :);
            Frequencies = Frequencies(:);

            % Perform SVD on PSD matrix at each frequency and collect first singular values
            s1 = zeros(size(PSD, 3), 1);
            for I = 1:size(PSD, 3)
                [~, s, ~] = svd(PSD(:, :, I));
                s1(I) = s(1, 1);
            end
            s1_all = [s1_all, s1];
       end

        % Calculate the average of all segments for peak detection
        s1_avg = mean(s1_all, 2);
        [peak_values, peak_locs] = findpeaks(s1_avg, Frequencies, 'MinPeakHeight', max(s1_avg) * peak_detection_threshold);

        % Save spectral diagram for all segments (overlayed)
        figure;
        hold on;  % Keep adding to the plot
        for impact_no = 1:num_segments
            plot(Frequencies, abs(s1_all(:, impact_no)), 'LineWidth', 1);
        end
        hold off;
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        title('Spectral Diagram for All Segments');
        saveas(gcf, fullfile(excitation_folder, 'Spectral_Diagram_All_Segments.png'));
        close all;  % Close the figure after saving
        
        % Save the averaged spectral diagram
        figure;
        plot(Frequencies, s1_avg, 'LineWidth', 2, 'Color', 'r');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        title('Averaged Spectral Diagram');
        saveas(gcf, fullfile(excitation_folder, 'Averaged_Spectral_Diagram.png'));
        close all;  % Close the figure after saving

        % Data storage for CSV
        csv_data = [];  % Initialize
        
        % Analyze each peak to identify "true" modes
        mode_count = 0;
        for peak_idx = 1:length(peak_locs)
            pp_range = [peak_locs(peak_idx) - 1, peak_locs(peak_idx) + 1];
            Frq = cell(num_segments, 1);
            Phi = cell(num_segments, 1);
            MAC_values = zeros(num_segments, 1);
        
            % Run FDD and calculate MAC values
            for impact_no = 1:num_segments
                acc_highpass_seg = acc_highpass(impact_index(impact_no) - 40 : impact_index(impact_no) + 560, :);
                [Frq{impact_no}, Phi{impact_no}] = FDD(acc_highpass_seg, fs, pp_range);
        
                if impact_no > 1
                    MAC_values(impact_no) = MAC_FDD(real(Phi{impact_no}), real(Phi{1}));
                end
            end
        
            % Normalize and process mode shapes
            for impact_no = 1:num_segments
                if ~isempty(Phi{impact_no})
                    Phi{impact_no} = real(Phi{impact_no}) / max(abs(real(Phi{impact_no})));
                end
            end
        
            % Calculate average, median, and standard deviation of MAC values
            MAC_avg = mean(MAC_values);
            MAC_median = median(MAC_values);
            MAC_std = std(MAC_values);
        
            % Only consider true modes based on MAC median threshold
            if MAC_median >= MAC_median_threshold
                mode_count = mode_count + 1;
                fprintf('True Mode Found: Frequency = %.2f Hz, pp_range = [%.2f, %.2f] Hz, Median MAC = %.2f\n', ...
                    peak_locs(peak_idx), pp_range(1), pp_range(2), MAC_median);
        
                % Append mode information for CSV (store in numeric array)
                for impact_no = 1:num_segments
                    row_data = [peak_locs(peak_idx), mode_count, Phi{impact_no}', MAC_avg, MAC_median, MAC_std];
                    csv_data = [csv_data; row_data];
                end
        
                % Save MAC values plot
                figure;
                bar(MAC_values, 'FaceColor', 'b');
                hold on;
                bar(1, MAC_values(1), 'FaceColor', 'r');
                xlabel('Impact test no.');
                ylabel('MAC values');
                title(sprintf('MAC Values for Frequency %.2f Hz', peak_locs(peak_idx)));
                saveas(gcf, fullfile(excitation_folder, sprintf('MAC_Mode_%d_Freq_%.2fHz.png', mode_count, peak_locs(peak_idx))));
                hold off;
        
                % Save mode shapes plot
                figure;
                hold on;
                for impact_no = 1:num_segments
                    if dot(Phi{impact_no}, Phi{1}) < 0
                        Phi{impact_no} = -Phi{impact_no};
                    end
                    plot(Phi{impact_no}, '-o');
                end
                grid on;
                xlabel('Sensor channel');
                ylabel('Identified mode shapes');
                title(sprintf('Mode Shapes for Frequency %.2f Hz', peak_locs(peak_idx)));
                saveas(gcf, fullfile(excitation_folder, sprintf('MAC_Mode_%d_Freq_%.2fHz.png', mode_count, peak_locs(peak_idx))));
                hold off;
            end
        end
        
        % Save the CSV file with mode shapes, frequencies, and MAC values
        csv_filename = fullfile(excitation_folder, sprintf('Modes_%s_%s.csv', test_num, excitation_type)); % Dynamic file name
        csv_header = {'Frequency (Hz)', 'Mode Number', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', ...
                      'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8', 'Channel 9', 'Channel 10', 'Channel 11', 'Channel 12', ...
                      'Average MAC', 'Median MAC', 'MAC Std Dev'};
        
        % Ensure that csv_data is numeric and write to file
        csvwrite_with_headers(csv_filename, csv_data, csv_header);
        
        disp('Results saved in the Results folder.');
    end
end

