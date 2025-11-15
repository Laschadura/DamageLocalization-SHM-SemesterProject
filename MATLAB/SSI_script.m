% This script processes preprocessed accelerometer data to extract modal parameters 
% using Stochastic Subspace Identification (SSICOV).
%
% The script performs the following steps:
% 1. Loads preprocessed acceleration data from CSV files.
% 2. Applies SSICOV to estimate modal parameters (natural frequencies, damping ratios, mode shapes).
% 3. Saves computed modal parameters and mode shapes as CSV files.
% 4. Generates stabilization diagrams and mode shape visualizations (2D and 3D).
%
% Computational efficiency: The script supports parallel processing for improved performance.
% 
% Author: Original script by Prof. Dr. Cheynet
% Altered by: Dr. Charikleia Stoura, Simon Scandella

% Clear workspace
clear; close all; clc;

% Base path (assuming script is run from the project folder)
base_path = pwd;

% Relative paths for data and output folders
folder_preprocessed = fullfile(base_path, 'DATA', 'PreprocessedData');
output_modal_folder = fullfile(base_path, 'DATA', 'ModalParameters');

plot_folder = fullfile(output_modal_folder, 'Plots');
if ~exist(plot_folder, 'dir')
    mkdir(plot_folder);
end


% Get a list of all CSV files in the PreprocessedData folder
file_list = dir(fullfile(folder_preprocessed, '*.csv')); % Match all .csv files

% Start parallel pool if not already started
%if isempty(gcp('nocreate'))
%    parpool;
%end

% Use parfor for parallel processing
for file_idx = 1:length(file_list)
    try
        % Start total timer for each file
        total_tic = tic;

        % Load the current file
        current_file = file_list(file_idx).name; % Load current file name outside parfor
        acceleration_file_name = fullfile(folder_preprocessed, current_file);

        % Display message
        disp(['Starting SSICOV for ', current_file, '...']);

        % Load the acceleration data from the CSV file
        opts = detectImportOptions(acceleration_file_name, 'VariableNamingRule', 'preserve');
        data = readtable(acceleration_file_name, opts);

        % Extract acceleration data (excluding time) and time step (dt)
        rz = data{:, 2:end}; % Exclude the first column (time)
        dt = 1 / 200; % Sampling period in seconds

        % Transpose the acc. data to match the [sensors x time steps] format
        rz = rz';

        % Perform SSICOV with optimized parameters
        [fn, zeta, phi, paraPlot] = SSICOV(rz, dt, ...
            'Ts', 2, ... % Adjust time segment length (smaller values for faster computation)
            'Nmin', 2, 'Nmax', 50,'eps_freq',1e-1, 'eps_cluster', 0.2); % Lower the model order range for speed

        % Add a mode identifier
        mode_numbers = (1:length(fn))'; % Mode identifiers (1, 2, 3, ...)

        % Save the modal parameters (natural frequencies and damping ratios) to a CSV file
        output_file_name_modal = sprintf('Modal_Parameters_%s', erase(current_file, 'Filtered_Acc_Data_'));
        modal_parameters_table = table(mode_numbers, fn', zeta', ...
            'VariableNames', {'Mode', 'Natural_Frequencies_Hz', 'Damping_Ratios_percent'});
        writetable(modal_parameters_table, fullfile(output_modal_folder, output_file_name_modal));
        disp(['Modal parameters saved to ', fullfile(output_modal_folder, output_file_name_modal)]);

        % Save the mode shapes to a separate CSV file
        output_file_name_modes = sprintf('Mode_Shapes_%s', erase(current_file, 'Filtered_Acc_Data_'));
        mode_shapes_table = array2table(phi, 'VariableNames', ...
            arrayfun(@(x) sprintf('Mode_%d', x), 1:size(phi, 2), 'UniformOutput', false));
        writetable(mode_shapes_table, fullfile(output_modal_folder, output_file_name_modes));
        disp(['Mode shapes saved to ', fullfile(output_modal_folder, output_file_name_modes)]);

        % Stop total timer and display elapsed time for the current file
        total_elapsed_time = toc(total_tic);
        disp(['Elapsed time for ', current_file, ': ', num2str(total_elapsed_time), ' seconds']);

        [h] = plotStabDiag(paraPlot.fn,rz(3,:),200,paraPlot.status,paraPlot.Nmin,paraPlot.Nmax);

        stab_diag_name = sprintf('%s_Stabilization_Diagram', erase(current_file, '.csv'));
        saveas(h, fullfile(plot_folder, [stab_diag_name, '.png']));
        close(h); % Close the stabilization figure after saving

        % visualization of identified modes
        posSensors_x = [5.885/2 5.885/2 5.885/2-3/4 5.885/2+3/4];
        posSensors_y = [0 2.015 2.015 2.015];
        posSensors_z = zeros(size(posSensors_x)); % z-coordinates (assuming sensors are in the x-y plane)

        mode_shapes_x = phi(:,[1,4,7,10])';  % Mode shapes in x-direction
        mode_shapes_y = phi(:,[2,5,8,11])';  % Mode shapes in y-direction
        mode_shapes_z = phi(:,[3,6,9,12])';  % Mode shapes in z-direction

        % Check data dimensions
        num_sensors = length(posSensors_x);
        [num_sensors_modeshape, num_modes] = size(mode_shapes_x);

        % Create a figure for plotting
        scale_factor = 1.5;

        % Define sensor labels
        sensor_labels = arrayfun(@(x) sprintf('S%d', x), 1:num_sensors, 'UniformOutput', false);

        % Plot mode shapes for each direction
        for mode_idx = 1:num_modes
            
            % Extract mode shape for current mode in each direction
            current_mode_shape_x = mode_shapes_x(:, mode_idx);
            current_mode_shape_y = mode_shapes_y(:, mode_idx);
            current_mode_shape_z = mode_shapes_z(:, mode_idx);

            % Compute displaced positions for each sensor based on the mode shape
            displaced_x = posSensors_x + scale_factor * current_mode_shape_x';
            displaced_y = posSensors_y + scale_factor * current_mode_shape_y';
            displaced_z = posSensors_z + scale_factor * current_mode_shape_z';

            figure(100+mode_idx);
            % Plot x-y plane
            subplot(2, 1, 1);  
            scatter(posSensors_x, posSensors_y, 100, 'filled',...
                'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', ...
             'DisplayName', 'Initial Position Sensors');
            hold on;
            for i = 1:num_sensors
                text(posSensors_x(i), posSensors_y(i), sensor_labels{i}, ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'black');
            end
            scatter(displaced_x, displaced_y, 100, mode_shapes_x(:, mode_idx), 'filled',...
                'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', ...
             'DisplayName', 'Deformed Position Sensors');
            for i = 1:num_sensors
                text(displaced_x(i), displaced_y(i), sensor_labels{i}, ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'black');
            end
            title(['Mode ', num2str(mode_idx), ' - X-Y Plane']);
            xlabel('X Coordinate');
            xlim([0 5.885])
            ylabel('Y Coordinate');
            grid on;
            legend;
           
            % Plot x-z plane
            subplot(2, 1, 2);  % Row 2: y-direction
            scatter(posSensors_x, posSensors_z, 100, 'filled',...
                'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', ...
             'DisplayName', 'Initial Position Sensors');
            hold on;
            for i = 1:num_sensors
                text(posSensors_x(i), posSensors_z(i), sensor_labels{i}, ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'black');
            end
            scatter(displaced_x, displaced_z, 100, mode_shapes_y(:, mode_idx), 'filled',...
                'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', ...
             'DisplayName', 'Deformed Position Sensors');
            for i = 1:num_sensors
                text(displaced_x(i), displaced_z(i), sensor_labels{i}, ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'black');
            end
            title(['Mode ', num2str(mode_idx), ' - X-Z Plane']);
            xlabel('X Coordinate');
            xlim([0 5.885])
            ylabel('Z Coordinate');
            grid on;
            legend;
            
            % Save the 2D plot after both subplots (X-Y and X-Z) are created
            mode_2d_name = sprintf('%s_Mode_%d_2D.png', erase(current_file, '.csv'), mode_idx);
            saveas(gcf, fullfile(plot_folder, mode_2d_name));
            close(gcf); % Close the 2D plot figure after saving


            % plot in 3D
            figure(200+mode_idx);
            % Plot the baseline (initial sensor positions)
            plot3(posSensors_x, posSensors_y, posSensors_z, 'k--', ...
                'LineWidth', 1.5); % Dashed line for initial position
            hold on;
            % Plot sensor locations
            scatter3(posSensors_x, posSensors_y, posSensors_z, ...
                100, 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
            for i = 1:num_sensors
                text(posSensors_x(i), posSensors_y(i), posSensors_z(i), sensor_labels{i}, ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'black');
            end
            scatter3(displaced_x, displaced_y, displaced_z, ...
             100, 'filled', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', ...
             'DisplayName', 'Deformed Position Sensors');
            for i = 1:num_sensors
                text(displaced_x(i), displaced_y(i), displaced_z(i), sensor_labels{i}, ...
                    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'black');
            end
            % Title and axis labels
            title(['Mode Shape ', num2str(mode_idx)]);
            xlabel('X Coordinate');
            xlim([0 5.885])
            ylabel('Y Coordinate');
            zlabel('Z Coordinate');
            axis equal;
            grid on;
            legend;

            % Save the 3D plot after creating the 3D mode shape visualization
            mode_3d_name = sprintf('%s_Mode_%d_3D.png', erase(current_file, '.csv'), mode_idx);
            saveas(gcf, fullfile(plot_folder, mode_3d_name));
            close(gcf); % Close the 3D plot figure after saving

        end
    catch ME
        % If an error occurs, display the message
        disp(['Error processing ', current_file, ': ', ME.message]);
    end
end

disp('Processing complete for all datasets.');