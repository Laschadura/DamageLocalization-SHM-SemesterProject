function [cluster_indices, num_clusters] = cluster_within_test(mode_shapes, frequencies, mac_threshold, frequency_tolerance)
% CLUSTER_WITHIN_TEST Clusters mode shapes within a single test based on MAC similarity and frequency proximity.
%
% INPUTS:
%   mode_shapes         - 2D array where each column is a mode shape vector.
%   frequencies         - Vector of frequencies corresponding to each mode shape.
%   mac_threshold       - MAC threshold for clustering within the same test.
%   frequency_tolerance - Relative frequency tolerance within the same test.
%
% OUTPUTS:
%   cluster_indices     - Array indicating the cluster index for each mode shape.
%   num_clusters        - Total number of clusters formed.

    % Number of modes
    num_modes = size(mode_shapes, 2);

    % Handle cases with less than two modes
    if num_modes < 2
        cluster_indices = ones(num_modes, 1);
        num_clusters = num_modes;
        fprintf('Only one mode present. Assigning to a single cluster.\n');
        return;
    end

    % Initialize adjacency matrix
    adjacency = zeros(num_modes);

    % Compute pairwise MAC and frequency differences
    fprintf('Computing pairwise MAC and frequency differences...\n');
    for i = 1:num_modes
        for j = i+1:num_modes
            % Compute relative frequency difference
            freq_rel_diff = abs(frequencies(i) - frequencies(j)) / min(frequencies(i), frequencies(j));

            % Compute MAC
            numerator = (abs(mode_shapes(:, i)' * mode_shapes(:, j)))^2;
            denominator = (mode_shapes(:, i)' * mode_shapes(:, i)) * (mode_shapes(:, j)' * mode_shapes(:, j));
            if denominator == 0
                MAC = 0;
            else
                MAC = numerator / denominator;
            end

            % Debugging: Print MAC and frequency difference for first 10 comparisons
            if i <= 2 && j <= 10
                fprintf('Comparing Mode %d and Mode %d: MAC = %.4f, Freq Rel Diff = %.4f\n', i, j, MAC, freq_rel_diff);
            end

            % Check if both frequency and MAC criteria are met
            if freq_rel_diff <= frequency_tolerance && MAC >= mac_threshold
                adjacency(i, j) = 1;
                adjacency(j, i) = 1;
            end
        end
    end

    % Create graph and find connected components
    G = graph(adjacency);
    clusters = conncomp(G);
    cluster_indices = clusters';
    num_clusters = max(clusters);

    % Debugging: Print number of clusters formed
    fprintf('Number of clusters formed: %d\n', num_clusters);

    % Optional: Print cluster sizes
    if num_clusters > 1
        cluster_sizes = histcounts(cluster_indices, 1:(num_clusters+1));
        for c = 1:num_clusters
            fprintf('  Cluster %d: %d modes\n', c, cluster_sizes(c));
        end
    end
end
