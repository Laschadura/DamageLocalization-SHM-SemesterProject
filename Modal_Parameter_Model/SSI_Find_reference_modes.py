import pandas as pd
import numpy as np
import glob
import os


# script to select most representative mode shapes for the SSI pipeline.
# It loads Selection.csv, computes MAC scores and exports final_refrence_modes.csv


# Load selection table with the modes to be used as reference
selection_df = pd.read_csv("Selection.csv")

def load_mode_shape(test_num, excitation, modes, mode_shapes_dir="ModeShapes"):
    """
    Load specific mode shapes based on test number and excitation, 
    and retrieve rows based on mode numbers as listed in the selection.

    Parameters:
    - test_num: int, test number.
    - excitation: str, excitation type.
    - modes: list of int, mode numbers to retrieve (based on row position).
    - mode_shapes_dir: str, directory where mode shapes are stored.
    
    Returns:
    - mode_shapes_list: list of mode shapes (each shape is a tuple with file, mode number, and mode shape data).
    """
    # Define file pattern and find all matching files (without augmentation)
    file_pattern = os.path.join(mode_shapes_dir, f"Mode_Shapes_Test_{test_num}_{excitation}.csv")
    mode_shape_files = glob.glob(file_pattern)

    # Filter out augmented files (containing "aug" in the filename)
    non_augmented_files = [f for f in mode_shape_files if "aug" not in f]

    if not non_augmented_files:
        print(f"No non-augmented mode shape file found for Test {test_num}, Excitation {excitation}")
        return None

    # Load the first non-augmented file (assumes there's only one valid file per test/excitation)
    mode_shape_file = non_augmented_files[0]
    mode_shapes_data = pd.read_csv(mode_shape_file)

    # Extract specific rows based on the selection; adjust by 1 due to header row
    mode_shapes_list = []
    for mode in modes:
        row_index = mode  # No need to subtract 1 since we’re considering it as the row position after header
        mode_shape = mode_shapes_data.iloc[row_index - 1, :].values  # Extract as numpy array
        mode_shapes_list.append((mode_shape_file, mode, mode_shape))  # Store file name, mode number, and shape

    return mode_shapes_list

# Dictionary to hold mode shapes for each type, now including Mode E
shape_types = {'Mode A': [], 'Mode B': [], 'Mode C': [], 'Mode D': [], 'Mode E': []}

# Iterate through the selection dataframe
for _, row in selection_df.iterrows():
    test_num = int(row['Test'])
    excitation = row['Escitation']
    
    # For each mode type, load the respective modes
    for mode_type in ['Mode A', 'Mode B', 'Mode C', 'Mode D', 'Mode E']:
        modes = row.get(mode_type)
        if pd.notna(modes):
            # Read the modes from Selection.csv
            mode_numbers = [int(float(m.strip())) for m in str(modes).split(';')]

            # Load the mode shapes for this test, excitation, and mode list
            mode_shapes = load_mode_shape(test_num, excitation, mode_numbers)
            
            # Store the mode shapes in the corresponding list for the shape type
            if mode_shapes:
                shape_types[mode_type].extend(mode_shapes)

# MAC calculation function
def calculate_mac(phi1, phi2):
    numerator = np.abs(np.dot(phi1.T, phi2)) ** 2
    denominator = (np.dot(phi1.T, phi1) * np.dot(phi2.T, phi2))
    return numerator / denominator if denominator != 0 else 0

# To store the most representative mode shape for each type
representative_modes = {}

for mode_type, mode_shapes in shape_types.items():
    if not mode_shapes:
        continue  # Skip if no mode shapes are available for this type
    
    # Convert list of shapes to numpy array for easier matrix operations
    mode_shapes_array = np.array([shape[2] for shape in mode_shapes])  # Only mode shape data
    mac_scores = np.zeros((len(mode_shapes), len(mode_shapes)))

    # Compute MAC scores between each pair and store info
    for i, (_, mode_i, phi1) in enumerate(mode_shapes):
        for j, (_, mode_j, phi2) in enumerate(mode_shapes):
            mac_scores[i, j] = calculate_mac(phi1, phi2)

    # Calculate median MAC for each mode shape and find the representative
    median_mac_scores = np.median(mac_scores, axis=1)
    best_index = np.argmax(median_mac_scores)
    selected_file, selected_mode, best_mode_shape = mode_shapes[best_index][:3]

    # Store the best representative mode shape
    representative_modes[mode_type] = best_mode_shape

    # Print details of selected mode
    print(f"Selected representative for {mode_type}:")
    print(f"  File: {selected_file}")
    print(f"  Mode: {selected_mode}")
    print(f"  Median MAC Score: {median_mac_scores[best_index]}")
    print("  MAC Scores:")
    for i, (_, mode, _) in enumerate(mode_shapes):
        print(f"    MAC with Mode {mode}: {mac_scores[best_index, i]}")

# Example to save representative modes to a CSV
reference_df = pd.DataFrame(representative_modes)
reference_df.to_csv("final_reference_modes.csv", index=False)
print("Saved representative modes to final_reference_modes.csv")
