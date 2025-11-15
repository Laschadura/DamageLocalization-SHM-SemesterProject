function [clustered_frequencies, clustered_mode_shapes] = cluster_modes(frequencies, mode_shapes, zeta, frequency_tolerance, mac_threshold)
% CLUSTER_MODES Clusters frequencies and mode shapes based on proximity and MAC criterion.

% INPUTS:
%   frequencies         - 2D Cell array of frequencies, organized as frequencies{window, order}.
%   mode_shapes         - 2D Cell array of mode shapes, organized as mode_shapes{window, order}.
%   zeta                - 2D Cell array of damping ratios, organized as zeta{window, order}.
%   frequency_tolerance - Tolerance for clustering nearby frequencies (e.g., 0.1 Hz).
%   mac_threshold       - Threshold for Modal Assurance Criterion (MAC) to consider
%                         two mode shapes similar (e.g., 0.95).

% OUTPUTS:
%   clustered_frequencies - Cell array containing clustered frequencies.
%   clustered_mode_shapes - Cell array containing clustered mode shapes.

% Initialize lists to collect all frequencies and mode shapes across windows and orders
all_frequencies = [];
all_mode_shapes = [];

% Collect data from all windows and orders
[num_windows, num_orders] = size(frequencies);
for i = 1:num_windows
    for j = 1:num_orders
        % Extract data for this window and order
        current_frequencies = frequencies{i, j};
        current_mode_shapes = mode_shapes{i, j};

        % Append data to global arrays
        if ~isempty(current_frequencies) % Only process non-empty cells
            all_frequencies = [all_frequencies; current_frequencies(:)];
            all_mode_shapes = cat(2, all_mode_shapes, current_mode_shapes);
        end
    end
end

% Initialize arrays to store the valid clusters based on MAC values
clustered_frequencies = {};
clustered_mode_shapes = {};

% Initialize a logical array to track unclustered frequencies
unclustered_indices = true(length(all_frequencies), 1);

% Step 1: Mutual Frequency Tolerance Clustering
while any(unclustered_indices)
    % Start a new cluster with the first unclustered frequency
    current_cluster_indices = find(unclustered_indices, 1);
    cluster_frequencies = all_frequencies(current_cluster_indices);
    cluster_mode_shapes = all_mode_shapes(:, current_cluster_indices);
    
    % Iteratively add frequencies that are mutually within the tolerance
    for idx = find(unclustered_indices)'
        % Check mutual frequency tolerance
        if all(abs(all_frequencies(idx) - cluster_frequencies) ./ ...
               min(all_frequencies(idx), cluster_frequencies) <= frequency_tolerance)
            cluster_frequencies = [cluster_frequencies; all_frequencies(idx)];
            cluster_mode_shapes = [cluster_mode_shapes, all_mode_shapes(:, idx)];
        end
    end
    
    % Update unclustered indices
    unclustered_indices(ismember(all_frequencies, cluster_frequencies)) = false;

    % Step 2: Filter by MAC Criterion
    num_modes = length(cluster_frequencies);
    adjacency_matrix = zeros(num_modes);

    % Build the adjacency matrix based on MAC criteria
    for i = 1:num_modes
        for j = i+1:num_modes
            % Compute MAC between mode shapes
            mac_value = (abs(cluster_mode_shapes(:, i)' * cluster_mode_shapes(:, j)))^2 / ...
                        ((cluster_mode_shapes(:, i)' * cluster_mode_shapes(:, i)) * ...
                         (cluster_mode_shapes(:, j)' * cluster_mode_shapes(:, j)));
            % If the MAC criterion is met, create an edge
            if mac_value >= mac_threshold
                adjacency_matrix(i, j) = 1;
                adjacency_matrix(j, i) = 1;
            end
        end
    end

    % Find connected components in the graph (clusters of compatible modes)
    graph_clusters = conncomp(graph(adjacency_matrix));

    % Aggregate frequencies and mode shapes for each connected component
    unique_clusters = unique(graph_clusters);
    for k = 1:length(unique_clusters)
        component_indices = find(graph_clusters == unique_clusters(k));
        
        % Only add the cluster if it has more than one frequency
        if length(component_indices) > 1
            clustered_frequencies{end+1} = cluster_frequencies(component_indices);
            clustered_mode_shapes{end+1} = cluster_mode_shapes(:, component_indices);
        end
    end
end

end
