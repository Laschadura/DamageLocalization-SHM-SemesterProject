%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script: Find_structural_modes.m
% Purpose: 
% Process modal parameters from vibration tests into structural modes.
%   1. Cluster mode shapes using MAC similarity.
%   2. Refine clusters using frequency range from the smallest test number.
%   3. Ensure global sign unification of mode shapes.
%   4. Save clustered results and plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% **1. Initialization**
clear; clc; close all;

% Suppress specific warnings
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');

% Add the 'Sources' folder to the MATLAB path
addpath('Sources');

%% **2. Define Input Parameters**
input_folder = 'Results ERA';            % Path to input data folder
output_folder = 'Clustered Results';    % Path to save output results

mac_threshold = 0.98;                    % Threshold for MAC-based clustering
frequency_tolerance_above = 1;           % Frequency tolerance above reference (Hz)
frequency_tolerance_below = 15;          % Frequency tolerance below reference (Hz)

%% **3. Initialize Data Structures**
all_clusters = struct('test_number', {}, 'excitation_type', {}, ...
    'mode_shapes', {}, 'frequencies', {}, 'mode_number', {});
metadata_index = 1;

%% **4. Load and Process Data**
fprintf('Loading data from tests...\n');

% Retrieve test folders
test_folders = dir(fullfile(input_folder, 'Test_*'));

for test_idx = 1:length(test_folders)
    test_folder = fullfile(input_folder, test_folders(test_idx).name);
    test_name = test_folders(test_idx).name;
    tokens = regexp(test_name, 'Test[_ ]*(\d+)', 'tokens');
    if isempty(tokens), continue; end
    test_number = str2double(tokens{1}{1});
    
    excitation_folders = dir(test_folder);
    excitation_folders = excitation_folders([excitation_folders.isdir] & ~ismember({excitation_folders.name}, {'.', '..'}));
    
    for excitation_idx = 1:length(excitation_folders)
        excitation_type = excitation_folders(excitation_idx).name;
        excitation_folder = fullfile(test_folder, excitation_type);
        csv_files = dir(fullfile(excitation_folder, '*.csv'));
        
        for file_idx = 1:length(csv_files)
            file_path = fullfile(excitation_folder, csv_files(file_idx).name);
            fprintf('Loading file: %s\n', file_path);
            
            % Load the CSV file
            try
                data = readtable(file_path, 'VariableNamingRule', 'preserve');
            catch
                warning('Failed to read file: %s. Skipping.', file_path);
                continue;
            end
            
            % Check for required columns
            if ~ismember('Frequency', data.Properties.VariableNames) || ...
               ~ismember('Mode Number', data.Properties.VariableNames)
                warning('Required columns missing in file: %s. Skipping.', file_path);
                continue;
            end
            
            frequencies = data.Frequency;
            mode_numbers = data.("Mode Number");
            mode_shape_columns = setdiff(data.Properties.VariableNames, {'Frequency', 'Mode Number'});
            mode_shapes = data{:, mode_shape_columns};
            
            % Normalize mode shapes
            mode_shapes(isnan(mode_shapes) | isinf(mode_shapes)) = 0;
            mode_shapes = mode_shapes ./ max(abs(mode_shapes), [], 2);
            
            % Group by mode number
            unique_modes = unique(mode_numbers);
            for mode = unique_modes'
                mode_indices = find(mode_numbers == mode);
                
                % Store mode group data
                all_clusters(metadata_index).test_number = test_number;
                all_clusters(metadata_index).excitation_type = excitation_type;
                all_clusters(metadata_index).mode_shapes = mode_shapes(mode_indices, :)';
                all_clusters(metadata_index).frequencies = frequencies(mode_indices);
                all_clusters(metadata_index).mode_number = mode;
                metadata_index = metadata_index + 1;
            end
        end
    end
end

fprintf('Data loading complete: %d initial clusters created.\n', length(all_clusters));

%% **5. MAC-Based Clustering**
fprintf('Clustering across tests using MAC similarity...\n');

cluster_indices = zeros(length(all_clusters), 1);
current_cluster_id = 1;

for i = 1:length(all_clusters)
    if cluster_indices(i) > 0
        continue; % Already assigned
    end
    
    % Start a new cluster
    cluster_indices(i) = current_cluster_id;
    cluster_queue = i; % Queue of clusters to process
    
    while ~isempty(cluster_queue)
        current_idx = cluster_queue(1);
        cluster_queue(1) = [];
        
        for j = 1:length(all_clusters)
            if cluster_indices(j) > 0 || i == j
                continue; % Skip already assigned or self
            end
            
            % Compute median MAC
            modes_i = all_clusters(current_idx).mode_shapes;
            modes_j = all_clusters(j).mode_shapes;
            modes_i_norm = modes_i ./ vecnorm(modes_i);
            modes_j_norm = modes_j ./ vecnorm(modes_j);
            mac_matrix = abs(modes_i_norm' * modes_j_norm).^2;
            mac_matrix(~isfinite(mac_matrix)) = 0;
            median_mac = median(mac_matrix(:));
            
            if median_mac >= mac_threshold
                cluster_indices(j) = current_cluster_id;
                cluster_queue = [cluster_queue, j]; % Add to processing queue
            end
        end
    end
    
    current_cluster_id = current_cluster_id + 1;
end

fprintf('MAC-based clustering complete.\n');

%% **6. Frequency-Based Refinement**
fprintf('Refining clusters based on frequency range...\n');

unique_clusters = unique(cluster_indices);
for cluster_id = unique_clusters'
    if cluster_id == 0, continue; end % Skip noise clusters
    
    % Get modes in the current cluster
    cluster_modes = find(cluster_indices == cluster_id);
    
    % Find the test with the smallest test number in the cluster
    [~, ref_test_idx] = min([all_clusters(cluster_modes).test_number]);
    ref_test_idx = cluster_modes(ref_test_idx);
    
    % Get the reference frequency from the smallest test number
    ref_frequency = all_clusters(ref_test_idx).frequencies(1);
    
    % Define frequency bounds
    lower_bound = ref_frequency - frequency_tolerance_below;
    upper_bound = ref_frequency + frequency_tolerance_above;
    
    for idx = cluster_modes'
        % Check if all frequencies in the mode group fall within the bounds
        mode_frequencies = all_clusters(idx).frequencies;
        if any(mode_frequencies < lower_bound | mode_frequencies > upper_bound)
            cluster_indices(idx) = 0; % Reassign to noise
        end
    end
end

fprintf('Frequency-based refinement complete.\n');

%% **7. Global Sign Unification**
fprintf('Performing global sign unification...\n');

unique_clusters = unique(cluster_indices);
for cluster_id = unique_clusters'
    if cluster_id == 0, continue; end % Skip noise clusters
    
    % Get modes in the current cluster
    cluster_modes = find(cluster_indices == cluster_id);
    
    % Choose a global reference shape (first shape in the cluster)
    ref_mode_shape = all_clusters(cluster_modes(1)).mode_shapes(:, 1);
    ref_mode_shape = ref_mode_shape / max(abs(ref_mode_shape)); % Normalize
    
    for idx = cluster_modes'
        mode_shapes = all_clusters(idx).mode_shapes;
        
        % Align each shape in the mode group
        for shape_idx = 1:size(mode_shapes, 2)
            current_shape = mode_shapes(:, shape_idx) / max(abs(mode_shapes(:, shape_idx))); % Normalize
            if dot(ref_mode_shape, current_shape) < 0
                mode_shapes(:, shape_idx) = -current_shape; % Flip sign
            else
                mode_shapes(:, shape_idx) = current_shape; % Ensure normalization
            end
        end
        
        % Update mode shapes with unified signs
        all_clusters(idx).mode_shapes = mode_shapes;
    end
end

fprintf('Global sign unification complete.\n');

%% **8. Save Results**

fprintf('Saving results...\n');

% Create output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

unique_tests = unique([all_clusters.test_number]);
for test_number = unique_tests
    test_clusters = find([all_clusters.test_number] == test_number);
    unique_excitations = unique({all_clusters(test_clusters).excitation_type});
    
    for excitation = unique_excitations
        excitation_clusters = test_clusters(strcmp({all_clusters(test_clusters).excitation_type}, excitation));
        all_frequencies = [];
        all_mode_shapes = [];
        all_cluster_ids = [];

        for idx = excitation_clusters
            frequencies = all_clusters(idx).frequencies;
            mode_shapes = all_clusters(idx).mode_shapes;
            cluster_id = cluster_indices(idx);

            all_frequencies = [all_frequencies; frequencies];
            all_mode_shapes = [all_mode_shapes; mode_shapes'];
            all_cluster_ids = [all_cluster_ids; repmat(cluster_id, size(frequencies))];
        end

        num_channels = size(all_mode_shapes, 2);
        mode_shape_columns = strcat('Channel', string(1:num_channels));
        output_table = array2table([all_frequencies, all_cluster_ids, all_mode_shapes], ...
            'VariableNames', [{'Frequency', 'ClusterID'}, mode_shape_columns]);

        excitation_folder = fullfile(output_folder, sprintf('Test_%d', test_number), excitation{1});
        if ~exist(excitation_folder, 'dir')
            mkdir(excitation_folder);
        end

        output_file_name = sprintf('Test_%d_%s.csv', test_number, excitation{1});
        writetable(output_table, fullfile(excitation_folder, output_file_name));
    end
end

fprintf('Results saved successfully.\n');

%% **9. Summary of Structural and Noise Data**

total_data_points = length(cluster_indices);
structural_points = sum(cluster_indices > 0);
noise_points = sum(cluster_indices == 0);

structural_percentage = (structural_points / total_data_points) * 100;
noise_percentage = (noise_points / total_data_points) * 100;

fprintf('\nSummary:\n');
fprintf('  Total Data Points: %d\n', total_data_points);
fprintf('  Structural Modes (Cluster ID > 0): %d (%.2f%%)\n', structural_points, structural_percentage);
fprintf('  Noise (Cluster ID = 0): %d (%.2f%%)\n', noise_points, noise_percentage);

%% **10. Plot Results**

fprintf('Generating plots for structural clusters...\n');
plot_folder = fullfile(output_folder, 'Cluster_Plots');
if ~exist(plot_folder, 'dir')
    mkdir(plot_folder);
end

unique_clusters = unique(cluster_indices);
for cluster_id = unique_clusters'
    if cluster_id == 0, continue; end % Skip noise clusters

    % Get modes in the cluster
    cluster_modes = find(cluster_indices == cluster_id);
    
    % Ensure at least two unique test numbers
    unique_tests_in_cluster = unique([all_clusters(cluster_modes).test_number]);
    if length(unique_tests_in_cluster) < 2
        continue; % Skip clusters with less than two unique tests
    end

    % Prepare data for sorted plotting
    plot_data = [];
    for idx = cluster_modes'
        test_number = all_clusters(idx).test_number;
        frequencies = all_clusters(idx).frequencies;
        mode_shapes = all_clusters(idx).mode_shapes;

        % Select a representative mode shape and its frequency
        representative_frequency = frequencies(1);
        representative_mode_shape = mode_shapes(:, 1);

        % Normalize and ensure sign consistency
        representative_mode_shape = representative_mode_shape / max(abs(representative_mode_shape));

        % Store in plot data
        plot_data = [plot_data; struct('test_number', test_number, ...
                                       'frequency', representative_frequency, ...
                                       'mode_shape', representative_mode_shape)];
    end

    % Sort plot data by test number
    [~, sorted_idx] = sort([plot_data.test_number]);
    plot_data = plot_data(sorted_idx);

    % Create plot
    figure('Name', sprintf('Structural Cluster %d', cluster_id));
    hold on;

    % Plot sorted data
    for i = 1:length(plot_data)
        data = plot_data(i);
        plot(data.mode_shape, '-o', 'DisplayName', ...
             sprintf('Test %d (%.2f Hz)', data.test_number, data.frequency));
    end

    % Finalize plot
    title(sprintf('Structural Cluster %d', cluster_id));
    xlabel('Channel');
    ylabel('Normalized Amplitude');
    legend('show', 'Location', 'best');
    grid on;
    hold off;

    % Save plot
    saveas(gcf, fullfile(plot_folder, sprintf('Cluster_%d.fig', cluster_id)));
    close;
end

fprintf('Plots saved successfully.\n');

