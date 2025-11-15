function [cluster_indices, num_clusters] = cluster_across_tests(all_clusters, mac_initial_threshold, mac_min_threshold, linear_decline_slope)
    %% **1. Initialization**
    num_clusters_input = length(all_clusters);
    if num_clusters_input < 2
        cluster_indices = ones(num_clusters_input, 1);
        num_clusters = num_clusters_input;
        fprintf('Only one cluster present. Assigning all to a single cluster.\n');
        return;
    end

    % Initialize variables
    cluster_indices = (1:num_clusters_input)'; % Each cluster starts as its own
    test_numbers = [all_clusters.test_number];
    test_number_range = max(test_numbers) - min(test_numbers);
    similarity_matrix = zeros(num_clusters_input); % Store pairwise median MAC

    %% **2. Compute Pairwise Similarities**
    fprintf('Computing pairwise median MAC similarities...\n');
    for i = 1:num_clusters_input
        for j = i+1:num_clusters_input
            % Compute test number difference
            test_diff = abs(test_numbers(i) - test_numbers(j));
            normalized_test_diff = test_diff / test_number_range;

            % Compute adaptive MAC threshold
            current_threshold = max(mac_initial_threshold - linear_decline_slope * normalized_test_diff, mac_min_threshold);

            % Compute representative mode shapes for both clusters
            rep_i = median(all_clusters(i).mode_shapes, 2);
            rep_j = median(all_clusters(j).mode_shapes, 2);

            % Normalize representatives
            rep_i = rep_i / norm(rep_i);
            rep_j = rep_j / norm(rep_j);

            % Compute MAC similarity
            mac_value = abs(rep_i' * rep_j)^2;

            % Assign similarity if above threshold
            if mac_value >= current_threshold
                similarity_matrix(i, j) = mac_value;
                similarity_matrix(j, i) = mac_value;
            end
        end
    end

    %% **3. Hierarchical Clustering**
    fprintf('Performing hierarchical clustering...\n');

    % Convert similarity matrix to dissimilarity matrix
    dissimilarity_matrix = 1 - similarity_matrix;
    dissimilarity_matrix(isnan(dissimilarity_matrix)) = 1; % Handle NaNs

    % Compute hierarchical clustering
    Z = linkage(squareform(dissimilarity_matrix), 'average'); % Use average linkage

    % Dynamically determine the cutoff threshold
    dynamic_cutoff = 1 - mac_min_threshold; % Adjust based on minimum threshold
    cluster_indices = cluster(Z, 'cutoff', dynamic_cutoff, 'Criterion', 'distance');

    %% **4. Cluster Quality Check**
    fprintf('Performing cluster quality checks...\n');
    unique_clusters = unique(cluster_indices);
    for cluster_id = unique_clusters'
        cluster_members = find(cluster_indices == cluster_id);
        if numel(cluster_members) < 2
            continue; % Skip single-member clusters
        end

        % Compute intra-cluster similarity
        intra_cluster_sims = [];
        for i = 1:length(cluster_members)
            for j = i+1:length(cluster_members)
                member_i = cluster_members(i);
                member_j = cluster_members(j);

                % Compute MAC for pairwise shapes
                modes_i = all_clusters(member_i).mode_shapes;
                modes_j = all_clusters(member_j).mode_shapes;

                % Representative mode shapes
                rep_i = median(modes_i, 2);
                rep_j = median(modes_j, 2);

                % Normalize
                rep_i = rep_i / norm(rep_i);
                rep_j = rep_j / norm(rep_j);

                % MAC value
                mac_value = abs(rep_i' * rep_j)^2;
                intra_cluster_sims = [intra_cluster_sims; mac_value];
            end
        end

        % Variance check
        if var(intra_cluster_sims) > 0.05 % Example threshold for variability
            fprintf('Cluster %d has high intra-cluster variability. Marking for further review.\n', cluster_id);
        end
    end

    %% **5. Output Results**
    num_clusters = max(cluster_indices);
    fprintf('Clustering across tests completed. Final number of clusters: %d\n', num_clusters);
end
