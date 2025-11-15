function [structural_clusters, cluster_indices] = cluster_structural_modes(mode_shapes, frequencies, test_numbers, mac_within_threshold, mac_across_threshold, frequency_tolerance_within, frequency_tolerance_across)
% CLUSTER_STRUCTURAL_MODES Clusters mode shapes based on MAC similarity,
% using hierarchical clustering with complete linkage, considering frequency differences
% within the same test and across different tests.

% INPUTS:
%   mode_shapes              - 2D array where each column is a mode shape vector.
%   frequencies              - Vector of frequencies corresponding to each mode shape.
%   test_numbers             - Vector of test numbers corresponding to each mode shape.
%   mac_within_threshold     - MAC threshold for clustering within the same test.
%   mac_across_threshold     - MAC threshold for clustering across different tests.
%   frequency_tolerance_within - Relative frequency tolerance within the same test.
%   frequency_tolerance_across - Relative frequency tolerance across different tests.

% OUTPUTS:
%   structural_clusters      - Cell array where each cell contains the mode shapes in a cluster.
%   cluster_indices          - Array indicating the cluster index for each mode shape.

% Number of modes
num_modes = size(mode_shapes, 2);

% Compute the pairwise MAC matrix
MAC = zeros(num_modes);
for i = 1:num_modes
    for j = i:num_modes
        mac_value = (abs(mode_shapes(:, i)' * mode_shapes(:, j)))^2 / ...
                    ((mode_shapes(:, i)' * mode_shapes(:, i)) * (mode_shapes(:, j)' * mode_shapes(:, j)));
        MAC(i, j) = mac_value;
        MAC(j, i) = mac_value;
    end
end

% Convert MAC to dissimilarity
dissimilarity = 1 - MAC;

% Apply frequency criteria to dissimilarity matrix
for i = 1:num_modes
    for j = i+1:num_modes
        freq_rel_diff = abs(frequencies(i) - frequencies(j)) / min(frequencies(i), frequencies(j));
        if test_numbers(i) == test_numbers(j)
            % Same test
            if freq_rel_diff > frequency_tolerance_within
                dissimilarity(i, j) = 1; % Set maximum dissimilarity
                dissimilarity(j, i) = 1;
            end
            if MAC(i, j) < mac_within_threshold
                dissimilarity(i, j) = 1; % Set maximum dissimilarity
                dissimilarity(j, i) = 1;
            end
        else
            % Different tests
            if freq_rel_diff > frequency_tolerance_across
                dissimilarity(i, j) = 1; % Set maximum dissimilarity
                dissimilarity(j, i) = 1;
            end
            if MAC(i, j) < mac_across_threshold
                dissimilarity(i, j) = 1; % Set maximum dissimilarity
                dissimilarity(j, i) = 1;
            end
        end
    end
end

% Flatten the upper triangle of the dissimilarity matrix for clustering
dissimilarity_vector = squareform(dissimilarity);

% Perform hierarchical clustering using complete linkage
Z = linkage(dissimilarity_vector, 'complete');

% Determine clusters by cutting the dendrogram at dissimilarity = 1 - threshold
% Since we have set dissimilarities to 1 where criteria are not met,
% we can use a threshold slightly less than 1 to exclude those.
cluster_indices = cluster(Z, 'cutoff', 0.999, 'Criterion', 'distance');

% Group mode shapes into clusters
unique_clusters = unique(cluster_indices);
structural_clusters = cell(length(unique_clusters), 1);
for k = 1:length(unique_clusters)
    structural_clusters{k} = mode_shapes(:, cluster_indices == unique_clusters(k));
end

end
