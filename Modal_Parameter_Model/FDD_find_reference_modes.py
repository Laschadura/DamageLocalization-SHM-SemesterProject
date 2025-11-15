import os
import re
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

def load_data_grouped_by_mode(root_dir="Results FDD", labels_path="Damage_Labels_20.csv"):
    """
    Collects mode shapes and associated features from nested directories,
    ensuring no data leakage by associating samples with `test_id` and `excitation`.

    Args:
        root_dir (str): Root directory containing test folders.
        labels_path (str): Path to the CSV file containing damage labels.

    Returns:
        data (list): List of dictionaries, each representing a single sample with associated features.
        labels_df (pd.DataFrame): DataFrame containing damage labels.
    """
    data = []

    # Iterate through all CSV files in nested directories
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv.csv"), recursive=True)

    if not csv_files:
        print(f"No CSV files found in directory: {root_dir}")
        return [], None

    print(f"Found {len(csv_files)} CSV files.")
    
    for csv_file in csv_files[:5]:  # Display details for first 5 files only
        print(f"Processing file: {csv_file}")

    for csv_file in csv_files:
        # Attempt to extract test_id and excitation
        match = re.search(r"Test_(\d+)_([A-Za-z_]+)\.csv\.csv", os.path.basename(csv_file))
        if not match:
            # Fallback to extract excitation and assign synthetic test_id
            match = re.search(r"Modes_Test_([A-Za-z_]+)\.csv\.csv", os.path.basename(csv_file))
            if match:
                excitation = match.group(1)
                test_id = int(re.search(r"Test_(\d+)", csv_file).group(1)) if re.search(r"Test_(\d+)", csv_file) else None
            else:
                print(f"Warning: Unable to parse test_id or excitation from file name {csv_file}")
                continue
        else:
            test_id = int(match.group(1))
            excitation = match.group(2)

        try:
            csv_data = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        # Ensure 'Mode Number' column exists
        if 'Mode Number' not in csv_data.columns:
            print(f"Warning: 'Mode Number' column not found in {csv_file}. Skipping.")
            continue

        # Group by Mode Number
        grouped = csv_data.groupby('Mode Number')
        for mode_number, group in grouped:
            frequency = group['Frequency (Hz)'].iloc[0]
            avg_mac = group['Average MAC'].iloc[0]
            median_mac = group['Median MAC'].iloc[0]
            mac_std_dev = group['MAC Std Dev'].iloc[0]

            # Process each mode shape row as an independent sample
            for idx, row in group.iterrows():
                mode_shape = row.iloc[2:14].values  # Extract sensor channels as a vector
                data.append({
                    'test_id': test_id,
                    'excitation': excitation,
                    'mode_number': mode_number,  # Use mode number from CSV
                    'segment_number': idx + 1,
                    'frequency': frequency,
                    'sample_avg_mac': avg_mac,
                    'sample_median_mac': median_mac,
                    'sample_mac_std_dev': mac_std_dev,
                    'sample_mode_shape': mode_shape
                })

    if not data:
        print("No mode shapes extracted. Please check the input data files.")
        return [], None

    try:
        labels_df = pd.read_csv(labels_path).set_index('Test Number')
        print(f"Labels loaded successfully. Labels shape: {labels_df.shape}")
    except Exception as e:
        print(f"Error reading labels file {labels_path}: {e}")
        labels_df = None

    print(f"Extracted {len(data)} mode shapes in total.")
    if len(data) > 0:
        print(f"Sample data entry: {data[0]}")  # Print a single sample entry

    return data, labels_df

def filter_clusters_by_test_occurrence(clustered_data, min_test_count):
    """
    Filters clusters to keep only those that appear in a minimum number of unique test IDs.

    Args:
        clustered_data (list): List of clustered mode shapes with associated test IDs.
        min_test_count (int): Minimum number of unique test IDs required for a cluster to be retained.

    Returns:
        filtered_clusters (dict): Dictionary of filtered clusters with test IDs and excitations.
    """
    from collections import defaultdict

    # Group mode shapes by cluster
    cluster_tests = defaultdict(set)

    for entry in clustered_data:
        cluster_label = entry['cluster_label']
        test_id = entry['test_id']
        excitation = entry['excitation']
        cluster_tests[cluster_label].add(f"Test_{test_id}_{excitation}")

    # Filter clusters based on the number of unique test IDs
    filtered_clusters = {
        cluster_label: tests
        for cluster_label, tests in cluster_tests.items()
        if len(set(test.split('_')[1] for test in tests)) >= min_test_count
    }

    print(f"Retained {len(filtered_clusters)} clusters after filtering with min_test_count={min_test_count}")
    return filtered_clusters


def analyze_clusters(clustered_data, early_tests, later_tests, min_test_count):
    """
    Analyzes clusters to find references that appear in early tests and repeat in later tests.
    Filters based on the total unique test numbers (early + later).

    Args:
        clustered_data (list): List of clustered mode shapes with associated test IDs.
        early_tests (list): List of early test IDs to focus on.
        later_tests (list): List of later test IDs to check for repetition.
        min_test_count (int): Minimum total unique test numbers required for a cluster to be retained.

    Returns:
        dict: Relevant clusters with early reference shapes and later tests.
    """
    from collections import defaultdict

    # Dictionary to track early and later tests for each cluster
    cluster_tests = defaultdict(lambda: {"early_tests": set(), "later_tests": set(), "reference_shapes": []})

    for entry in clustered_data:
        cluster_label = entry['cluster_label']
        test_id = entry['test_id']
        excitation = entry['excitation']
        mode_number = entry['mode_number']  # Use mode number directly from data

        # Populate early and later test information
        if test_id in early_tests:
            cluster_tests[cluster_label]["early_tests"].add(test_id)
            cluster_tests[cluster_label]["reference_shapes"].append({
                "test_id": test_id,
                "excitation": excitation,
                "mode_number": mode_number
            })

        if test_id in later_tests:
            cluster_tests[cluster_label]["later_tests"].add(test_id)

    # Filter clusters based on total unique test numbers (early + later)
    filtered_clusters = {
        cluster_label: details
        for cluster_label, details in cluster_tests.items()
        if (len(details["later_tests"]) + (1 if details["early_tests"] else 0)) >= min_test_count
    }

    return filtered_clusters


def calculate_mac_matrix(Phi, reference_modes):
    """
    Calculate MAC (Modal Assurance Criterion) matrix between mode shapes and reference modes.
    """
    numerator = np.abs(Phi @ reference_modes.T) ** 2
    denom_modes = np.sum(Phi ** 2, axis=1, keepdims=True)
    denom_refs = np.sum(reference_modes ** 2, axis=1, keepdims=True).T
    return numerator / (denom_modes @ denom_refs)


def cluster_mode_shapes_across_tests(mode_shapes_data, mac_threshold=0.8):
    """
    Clusters mode shapes across all tests based on MAC similarity.

    Args:
        mode_shapes_data (list): List of dictionaries containing mode shapes and their properties.
        mac_threshold (float): Minimum MAC similarity to consider a mode shape part of a cluster.

    Returns:
        list: Mode shapes with assigned cluster labels and MAC values.
    """
    # Debugging: Check input structure
    if not isinstance(mode_shapes_data, list) or len(mode_shapes_data) == 0:
        print("Error: mode_shapes_data is not a list or is empty.")
        return []
    if not all(isinstance(entry, dict) and 'sample_mode_shape' in entry for entry in mode_shapes_data):
        print("Error: mode_shapes_data does not contain expected dictionaries with 'sample_mode_shape'.")
        return []

    # Extract mode shapes as a matrix
    Phi = np.array([entry['sample_mode_shape'] for entry in mode_shapes_data])

    # Debug: Check extracted matrix
    print(f"Extracted Phi matrix with shape: {Phi.shape}")

    # Calculate MAC matrix between all mode shapes
    mac_matrix = calculate_mac_matrix(Phi, Phi)

    # Initialize cluster assignments
    cluster_labels = -1 * np.ones(len(mode_shapes_data), dtype=int)  # Default: -1 (Noise)
    cluster_id = 0

    # Perform clustering based on MAC
    for i in range(len(mac_matrix)):
        if cluster_labels[i] == -1:  # If not yet assigned to a cluster
            # Find all similar mode shapes
            similar_indices = np.where(mac_matrix[i] >= mac_threshold)[0]
            cluster_labels[similar_indices] = cluster_id
            cluster_id += 1

    # Assign cluster labels back to mode_shapes_data
    clustered_data = []
    for idx, entry in enumerate(mode_shapes_data):
        clustered_data.append({
            **entry,
            'cluster_label': f"Cluster_{cluster_labels[idx]}",
            'mac_values': mac_matrix[idx].tolist()
        })

    return clustered_data


def summarize_clusters(clustered_data):
    """
    Summarizes the clusters, showing which tests and excitations belong to each cluster.
    """
    cluster_summary = {}
    for entry in clustered_data:
        cluster = entry['cluster_label']
        test_excitation = f"Test_{entry['test_id']}_{entry['excitation']}"

        if cluster not in cluster_summary:
            cluster_summary[cluster] = set()

        cluster_summary[cluster].add(test_excitation)

    # Convert sets to sorted lists for better readability
    for cluster in cluster_summary:
        cluster_summary[cluster] = sorted(cluster_summary[cluster])

    return cluster_summary

def save_reference_modes(clustered_data, output_file="reference_modes.csv", healthy_tests=None, min_test_count=10):
    """
    Saves the reference mode shapes to a CSV file, ensuring one reference shape per cluster.
    Only includes clusters with a unique test appearance count >= min_test_count.

    Args:
        clustered_data (list): List of clustered mode shapes with associated metadata.
        output_file (str): Output file path for saving the reference modes.
        healthy_tests (list): List of test IDs considered as healthy states. If provided, reference shapes
                              will be selected only from these tests.
        min_test_count (int): Minimum number of unique test IDs required for a cluster to be included.

    Returns:
        None
    """
    from collections import defaultdict

    # Group mode shapes by cluster
    cluster_shapes = defaultdict(list)
    for entry in clustered_data:
        cluster_shapes[entry['cluster_label']].append(entry)

    reference_modes = []
    for cluster_label, shapes in cluster_shapes.items():
        # Count unique test appearances
        unique_tests = {shape['test_id'] for shape in shapes}

        # Filter out clusters with unique test appearances < min_test_count
        if len(unique_tests) < min_test_count:
            continue

        # Filter shapes to only include those from healthy tests if specified
        if healthy_tests:
            shapes = [shape for shape in shapes if shape['test_id'] in healthy_tests]

        if not shapes:
            print(f"No shapes available for cluster {cluster_label} in the specified healthy tests.")
            continue

        # Extract mode shapes and calculate pairwise MAC
        mode_shapes = np.array([shape['sample_mode_shape'] for shape in shapes])
        mac_matrix = calculate_mac_matrix(mode_shapes, mode_shapes)

        # Calculate average MAC score for each mode shape
        avg_mac_scores = np.mean(mac_matrix, axis=1)

        # Identify the shape with the highest average MAC score
        best_shape_index = np.argmax(avg_mac_scores)
        best_shape = shapes[best_shape_index]

        # Add the reference mode to the output list
        reference_modes.append({
            "Cluster": cluster_label,
            "Test Number": best_shape['test_id'],
            "Excitation": best_shape['excitation'],
            "Mode Number": best_shape['mode_number'],
            "Appearance Count": len(unique_tests),  # Include unique test appearance count
            "Mode Shape": ", ".join(map(str, best_shape['sample_mode_shape']))  # Convert shape to string for CSV
        })

    # Save to CSV
    df = pd.DataFrame(reference_modes)
    df.to_csv(output_file, index=False)
    print(f"Reference modes saved to {output_file}")



if __name__ == "__main__":
    # Load mode shapes data
    mode_shapes_data, labels_df = load_data_grouped_by_mode(labels_path="Damage_Labels_20.csv")

    if not mode_shapes_data:
        print("No mode shapes data loaded.")
        exit()

    # Pass validated data to clustering
    clustered_data = cluster_mode_shapes_across_tests(mode_shapes_data, mac_threshold=0.7)

    # Centralized min_test_count setting
    min_test_count = 10  

    # Analyze clusters for early and later test references
    early_tests = [1, 2, 3]
    later_tests = list(range(4, 26))
    relevant_clusters = analyze_clusters(clustered_data, early_tests, later_tests, min_test_count)

    # Improved print format for Reference Clusters Summary
    print("\nReference Clusters Summary with Compact Early Reference Shapes:")
    for cluster, details in relevant_clusters.items():
        print(f"{cluster}:")
        print("  Early Reference Shapes:")

        # Compact the early reference shapes by grouping
        grouped_shapes = {}
        for ref in details["reference_shapes"]:
            key = (ref['test_id'], ref['excitation'])
            if key not in grouped_shapes:
                grouped_shapes[key] = set()
            grouped_shapes[key].add(ref['mode_number'])

        # Print the grouped summary
        for (test_id, excitation), modes in grouped_shapes.items():
            modes_str = ", ".join(map(str, sorted(modes)))
            print(f"    Test {test_id}, {excitation}, Modes: {modes_str}")

        later_test_list = sorted(details["later_tests"])
        print(f"  Later Tests: {', '.join(map(str, later_test_list))}")

        save_reference_modes(
        clustered_data,
        output_file="reference_modes.csv",
        healthy_tests=[1, 2, 3],  # Optional: Include only these tests
        min_test_count=10  # Only include clusters appearing in at least 10 unique tests
        )
