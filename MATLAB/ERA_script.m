% This MATLAB script processes vibration test data for modal analysis using the Eigensystem Realization Algorithm (ERA)
% and clusters stable modal frequencies. It automates data handling, modal parameter extraction, clustering, and visualization.
% 
% Key Steps:

% Initialization: Sets up paths, loads test data, and organizes outputs.
% Data Preprocessing: Reads CSV files, computes sampling rate, and isolates impulse response windows.
% Modal Parameter Extraction: Uses ERA to calculate frequencies, mode shapes, and damping ratios for various model orders.
% Clustering: Groups stable frequencies and mode shapes based on frequency tolerance and MAC (Modal Assurance Criterion).
% Results: Exports clustered modal parameters to CSV and visualizes impulse responses and mode shapes.

% Required Functions:

% isolateResponseWindows.m (impulse detection)
% processERA.m (modal parameter extraction using ERA)
% cluster_modes.m (frequency and mode shape clustering)
%
%Author: Prof. Dr. Eleni Chatzi
%altered by Simon Scandella




warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
%% Initialization for Batch Processing
clear; clc;

% Add the current folder and all its subfolders to the MATLAB path
addpath(genpath(pwd));

% Define the base folder path containing the test folders
base_folder = "DATA/VibrationTestData";
output_base_folder = "Results";

% Sampling rate
fs = 200;  % Replace with actual sampling rate if different

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
        
        % Parse the file name to extract the excitation type
        file_parts = split(file_name, '_');  % Split by underscores
        test_num = file_parts{2};            % Test number (if needed for debugging)
        excitation_type = strjoin(file_parts(3:end), '_');  % Join parts after the test number
        
        % Remove the ".csv" extension from the excitation type
        [~, excitation_type_without_extension, ~] = fileparts(excitation_type);
        
        % Create the folder for this excitation type
        excitation_folder = fullfile(output_folder, excitation_type_without_extension);
        if ~exist(excitation_folder, 'dir')
            mkdir(excitation_folder);
        end

        
        %% Load the Data
        Data = readtable(file_path);
        Acceleration = Data{:, 2:end};

        % Convert to datetime objects
        timeStamps = Data{:, 1};
        timeStamps = datetime(timeStamps, 'InputFormat', 'yyyy/MM/dd HH:mm:ss.SSS');

        % Compute the differences between successive timestamps
        timeDifferences = diff(timeStamps); % Returns duration array

        % Convert to seconds
        dt = mean(seconds(timeDifferences));
        fs = 1 / dt;    % Sampling rate

        % Select the y and z deck sensors
        cols_to_exclude = 1:3:size(Acceleration, 2);
        y = Acceleration(:, setdiff(1:size(Acceleration, 2), cols_to_exclude));
        responseColumn = y(:, 2);

        % Detect impulse windows
        thresholdMultiplier = 5; % Adjust based on the signal's characteristics
        impulseIndices = isolateResponseWindows(responseColumn, thresholdMultiplier, fs);

        % Plot the signal
        figure;
        time = 0:dt:(size(y, 1) - 1) * dt;
        plot(time, responseColumn, 'b'); % Plot the signal
        hold on;
        for i = 1:length(impulseIndices)
            xline(time(impulseIndices(i)), 'r--', 'LineWidth', 1.5); % Red dashed line
        end
        savefig(fullfile(excitation_folder, 'Impulse_Response.fig'));
        close;

        impulseIndices = impulseIndices(1:2:end);

        % Define the range of model orders to analyze
        orders = 40:4:60;
        num_orders = length(orders);
        num_windows = length(impulseIndices) - 1;

        % Initialize storage for all windows and orders
        frequencies = cell(num_windows, num_orders);
        mode_shapes = cell(num_windows, num_orders);
        zeta = cell(num_windows, num_orders);

        for i = 1:num_windows  % Loop over windows
            startIdx = impulseIndices(i);
            endIdx = impulseIndices(i + 1) - 1;

            Acc = y(startIdx:endIdx, :) - mean(y(startIdx:endIdx, :));
            [b, a] = butter(4, [5, 65] / (fs / 2), 'bandpass');
            Acc_f = filtfilt(b, a, Acc);

            factor = 1;
            Acc_f = downsample(Acc_f, factor);
            Fs_d = fs / factor;

            for k = 1:num_orders
                [freq_rel, phi, zeta_rel] = processERA(Acc_f, 2, 'imp', [], Fs_d, 2^(nextpow2(size(Acc_f, 1)) - 1), size(Acc, 2), 4, orders(k));
                frequencies{i, k} = freq_rel;
                mode_shapes{i, k} = real(phi);
                zeta{i, k} = zeta_rel;
            end
        end

        %% Clustering Stable Frequencies
        frequency_tolerance = 0.05125;  
        mac_threshold = 0.9;

        [clustered_frequencies, clustered_mode_shapes] = cluster_modes(frequencies, mode_shapes, zeta, frequency_tolerance, mac_threshold);

        lower_bound_frequency = 20; 
        valid_clusters = cellfun(@(x) length(x) >= 4 && all(x >= lower_bound_frequency), clustered_frequencies);
        filtered_frequencies = clustered_frequencies(valid_clusters);
        filtered_mode_shapes = clustered_mode_shapes(valid_clusters);

        global_reference = filtered_mode_shapes{1}(:, 1);
        global_reference = global_reference / max(abs(global_reference));  % Normalize the reference mode shape
        
        for cluster_idx = 1:length(filtered_mode_shapes)
            mode_shapes_cluster = filtered_mode_shapes{cluster_idx};
            for mode_idx = 1:size(mode_shapes_cluster, 2)
                % Normalize mode shape to max absolute value = 1
                mode_shapes_cluster(:, mode_idx) = mode_shapes_cluster(:, mode_idx) / max(abs(mode_shapes_cluster(:, mode_idx)));
                
                % Align with global reference
                alignment_score = dot(mode_shapes_cluster(:, mode_idx), global_reference);
                if alignment_score < 0
                    mode_shapes_cluster(:, mode_idx) = -mode_shapes_cluster(:, mode_idx);
                end
            end
            filtered_mode_shapes{cluster_idx} = mode_shapes_cluster;
        end

        
        % Export Results to CSV
        output_data = [];
        for cluster_idx = 1:length(filtered_frequencies)
            freq_cluster = filtered_frequencies{cluster_idx};
            mode_shapes_cluster = filtered_mode_shapes{cluster_idx};
            for mode_idx = 1:length(freq_cluster)
                frequency = freq_cluster(mode_idx);
                
                % Ensure mode shape is normalized before exporting
                normalized_mode_shape = mode_shapes_cluster(:, mode_idx) / max(abs(mode_shapes_cluster(:, mode_idx)));
                row = [frequency, cluster_idx, normalized_mode_shape.'];
                output_data = [output_data; row];
            end
        end
        
        headers = [{'Frequency', 'Mode Number'}, strcat('Channel ', string(1:size(mode_shapes_cluster, 1)))];
        output_table = array2table(output_data, 'VariableNames', headers);
        csv_path = fullfile(excitation_folder, sprintf('Test_%s_%s.csv', test_num, excitation_type_without_extension));
        writetable(output_table, csv_path);


        %% Save Clusters Figure
        figure('Name', 'All Valid Clusters');
        num_clusters = length(filtered_frequencies);
        rows = ceil(sqrt(num_clusters));
        cols = ceil(num_clusters / rows);

        for cluster_idx = 1:num_clusters
            subplot(rows, cols, cluster_idx);
            freq_cluster = filtered_frequencies{cluster_idx};
            mode_shapes_cluster = filtered_mode_shapes{cluster_idx};
            hold on;
            for mode_idx = 1:size(mode_shapes_cluster, 2)
                plot(1:size(mode_shapes_cluster, 1), mode_shapes_cluster(:, mode_idx), '-o', ...
                    'DisplayName', ['Frequency: ', num2str(freq_cluster(mode_idx), '%.3f'), ' Hz']);
            end
            title(['Cluster ', num2str(cluster_idx)]);
            xlabel('Channel');
            ylabel('Amplitude');
            legend('Location', 'best');
            hold off;
        end
        savefig(fullfile(excitation_folder, 'All_Valid_Clusters.fig'));
        close;
    end
end
