function plot_stabilization_and_modesOSMOS(orders, frequencies, clustered_frequencies, clustered_mode_shapes)
% PLOT_STABILIZATION_AND_MODESOSMOS Plots the stabilization chart with scatter points for each order
% and overlays mode shapes for each frequency bin in subplots, showing clustered frequencies with unique colors.
%
% INPUTS:
%   orders                - Vector of SSI orders corresponding to each experiment.
%   frequencies           - 2D cell array of frequencies, organized as frequencies{window, order}.
%   clustered_frequencies - Cell array containing clustered frequencies after clustering.
%   clustered_mode_shapes - Cell array containing mode shapes corresponding to clustered frequencies.

    % --- Plot the Stabilization Chart ---
    figure;
    hold on;

    % Scatter plot of individual frequencies for each order across all windows
    [num_windows, num_orders] = size(frequencies);
    for i = 1:num_windows
        for k = 1:num_orders
            if ~isempty(frequencies{i, k})
                scatter(frequencies{i, k}, orders(k) * ones(size(frequencies{i, k})), 'filled');
            end
        end
    end
    
    grid on;

    % Assign unique colors for each cluster
    cmap = lines(length(clustered_frequencies));  % Use distinct colors (e.g., 'lines' colormap)
    
    % Plot clustered frequencies with different colors
    for cluster_id = 1:length(clustered_frequencies)
        cluster_freqs = clustered_frequencies{cluster_id};  % Frequencies in this cluster
        
        % Plot each frequency in the cluster as a vertical line across all orders
        for f = 1:length(cluster_freqs)
            xline(cluster_freqs(f), 'Color', cmap(cluster_id, :), 'LineWidth', 1.5);
        end
    end
    
    % Customize stabilization chart appearance
    xlabel('Frequency (Hz)');
    ylabel('SSI Order');
    title('Stabilization Plot with Clustered Frequencies (Colored)');
    grid on;
    hold off;
    
    % --- Plot Mode Shapes in Subplots ---
    figure;
    num_clusters = length(clustered_frequencies);  % Number of clusters
    
    % Calculate number of rows and columns for subplots based on the number of clusters
    num_cols = ceil(sqrt(num_clusters));
    num_rows = ceil(num_clusters / num_cols);
    
    for cluster_id = 1:num_clusters
        % Get mode shapes and frequencies for the current cluster
        mode_shapes = clustered_mode_shapes{cluster_id};
        cluster_freqs = clustered_frequencies{cluster_id};
        
        % Plot in subplots
        subplot(num_rows, num_cols, cluster_id);
        hold on;
        
        % Plot each mode shape for the current frequency bin
        for m = 1:size(mode_shapes, 2)  % Iterate over the mode shapes (columns)
            mode_shape = mode_shapes(:, m);
            
            % Normalize the mode shape
            [mval, ind] = max(abs(mode_shape));
            mode_shape = mode_shape * sign(mode_shape(ind)) / mval;
            
            % Plot the normalized mode shape
            plot(0:length(mode_shape)-1, mode_shape, 'DisplayName', ...
                 ['Frequency: ' num2str(cluster_freqs(m)) ' Hz'], 'Color', cmap(cluster_id, :));
            grid on;
        end
        
        % Add legend and title to each subplot
        legend('show');
        title(['Cluster ' num2str(cluster_id) ' Mode Shapes']);
        xlabel('Length along beam');
        ylabel('Mode shape amplitude');
        hold off;
    end
end
