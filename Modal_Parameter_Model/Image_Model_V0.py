import os
import re
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers  # type: ignore
from tensorflow.keras.layers import Input, Dropout, Dense, Reshape, Conv2DTranspose  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam  # type: ignore
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import random
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from concurrent.futures import ThreadPoolExecutor

# >>> Additional imports for metrics <<<
from sklearn.metrics import classification_report, confusion_matrix

#Final version for submission. 04.12.2025


# ----------------------------------------
# 1) Utility: Derivatives
# ----------------------------------------
def compute_derivatives(mode_shape: np.ndarray) -> tuple:
    first_derivative = np.gradient(mode_shape)
    second_derivative = np.gradient(first_derivative)
    return first_derivative, second_derivative


# ----------------------------------------
# 2) Gaussian Smoothing for Label Masks
#    ensuring max remains 1 where original mask was 1
# ----------------------------------------
def smooth_mask_preserve_ones(binary_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Applies a Gaussian filter to 'soften' the edges of a binary mask (0/1),
    but re-scales so that any pixel originally 1 stays at 1 in the result,
    creating a smooth transition near edges.
    
    Args:
        binary_mask: (H, W, 1) array with values in {0, 1}
        sigma: Gaussian sigma for smoothing

    Returns:
        smooth_mask: float mask in [0, 1] with softened edges
                     and center pixels = 1 if they were originally 1.
    """
    if binary_mask.ndim == 3 and binary_mask.shape[-1] == 1:
        binary_mask = binary_mask[..., 0]  # Drop the channel dim for gaussian_filter
    elif binary_mask.ndim == 2:
        pass  # OK
    else:
        raise ValueError(f"Expected (H,W,1) or (H,W); got shape {binary_mask.shape}")

    smoothed = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma, mode='constant')

    # Identify originally-1 pixels
    original_ones = (binary_mask > 0.5)
    if np.any(original_ones):
        local_max = smoothed[original_ones].max()
        if local_max > 0:
            # Scale so that max in region of originally 1 is back to 1
            smoothed /= local_max

    smoothed = np.clip(smoothed, 0.0, 1.0)
    smoothed = smoothed[..., None]
    return smoothed


# ----------------------------------------
# 3) Preload Label Images
# ----------------------------------------
def preload_label_images(labels_root_dir: str, perspective_map: dict, image_shape: tuple = (256, 256)) -> dict:
    preloaded_images = defaultdict(dict)
    test_subdirs = glob.glob(os.path.join(labels_root_dir, 'Test_*'))

    def load_image(test_dir: str, test_number: int, perspective: str, label_prefix: str):
        label_image_filename = f"{label_prefix}_T{test_number}.png"
        label_image_path = os.path.join(test_dir, label_image_filename)
        if os.path.exists(label_image_path):
            img = cv2.imread(label_image_path, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                target_h, target_w = image_shape

                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                pad_w = target_w - new_w
                pad_h = target_h - new_h
                top, bottom = pad_h // 2, pad_h - (pad_h // 2)
                left, right = pad_w // 2, pad_w - (pad_w // 2)
                img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

                hsv = cv2.cvtColor(img_padded, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 70, 50])
                upper_red2 = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)

                red_mask = red_mask.astype(np.float32) / 255.0
                binary_mask = red_mask.reshape(target_h, target_w, 1)
                return binary_mask

        print(f"WARNING: Image not found for Test {test_number}, Perspective {perspective}. Using zeros.")
        return np.zeros((image_shape[0], image_shape[1], 1), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for test_dir in test_subdirs:
            match = re.search(r"Test[_]?(\d+)", os.path.basename(test_dir))
            if not match:
                print(f"WARNING: Unable to parse test_number from directory name {test_dir}. Skipping.")
                continue
            test_number = int(match.group(1))
            fut_A = executor.submit(load_image, test_dir, test_number, 'A', perspective_map['A'])
            fut_B = executor.submit(load_image, test_dir, test_number, 'B', perspective_map['B'])
            fut_C = executor.submit(load_image, test_dir, test_number, 'C', perspective_map['C'])

            img_A = fut_A.result()
            img_B = fut_B.result()
            img_C = fut_C.result()

            composite_image = np.concatenate([img_A, img_B, img_C], axis=1)
            preloaded_images[test_number]['composite'] = composite_image

    return preloaded_images


# ----------------------------------------
# 4) Load CSV & Construct Features
# ----------------------------------------
def load_data_as_tests_all_samples(
    root_dir: str = "Clustered Results",
    preloaded_images: dict = None,
    perspective_map: dict = None,
    image_shape: tuple = (256, 256),
    smoothing_sigma: float = 2.0
) -> tuple:
    """
    Loads CSV files for each test, builds feature vectors, and applies
    Gaussian smoothing to the mask before storing it.
    """
    if preloaded_images is None or perspective_map is None:
        raise ValueError("preloaded_images and perspective_map must be provided.")

    tests_features = defaultdict(list)
    tests_heatmaps = {}
    test_ids = []

    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)

    # Determine the maximum test number for normalization
    test_numbers = []
    for csv_file in csv_files:
        match = re.search(r"Test[_]?(\d+)", os.path.basename(csv_file))
        if not match:
            continue
        test_number = int(match.group(1))
        if test_number not in [23, 24]:
            test_numbers.append(test_number)
    max_test_number = max(test_numbers) if test_numbers else 1  # Avoid division by zero

    for csv_file in csv_files:
        match = re.search(r"Test[_]?(\d+)", os.path.basename(csv_file))
        if not match:
            continue
        test_number = int(match.group(1))

        if test_number in [23, 24]:
            print(f"INFO: Skipping Test {test_number} as it is missing.")
            continue

        try:
            csv_data = pd.read_csv(csv_file)
        except Exception as e:
            print(f"ERROR: Error reading {csv_file}: {e}")
            continue

        required_columns = ['Frequency', 'ClusterID'] + [f"Channel{i}" for i in range(1, 9)]
        if not all(col in csv_data.columns for col in required_columns):
            continue

        for _, row in csv_data.iterrows():
            frequency = row['Frequency']
            cluster_id = row['ClusterID']
            mode_shape = row[[f"Channel{i}" for i in range(1, 9)]].values

            first_derivative, second_derivative = compute_derivatives(mode_shape)

            normalized_test_number = test_number / max_test_number

            feature_vector = np.concatenate((
                [frequency],
                mode_shape,
                first_derivative,
                second_derivative,
                [cluster_id],
                [normalized_test_number]
            ))
            tests_features[test_number].append(feature_vector)
            test_ids.append(test_number)

    # Apply smoothing
    for test_number in tests_features.keys():
        raw_mask = preloaded_images[test_number]['composite']
        smoothed_mask = smooth_mask_preserve_ones(raw_mask, sigma=smoothing_sigma)
        tests_heatmaps[test_number] = smoothed_mask

    unique_tests = sorted(tests_features.keys())
    print(f"INFO: Loaded data for {len(unique_tests)} tests.")
    return tests_features, tests_heatmaps, unique_tests, max_test_number


# ----------------------------------------
# 5) Chronological Splits
# ----------------------------------------

def chronological_splits(unique_test_ids: list, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    num_tests = len(unique_test_ids)
    num_train = int(train_ratio * num_tests)
    num_val = int(val_ratio * num_tests)
    num_test = num_tests - num_train - num_val

    train_ids = unique_test_ids[:num_train]
    val_ids = unique_test_ids[num_train:num_train + num_val]
    test_ids = unique_test_ids[num_train + num_val:]

    split_labels = {}
    for t in train_ids:
        split_labels[t] = 'train'
    for t in val_ids:
        split_labels[t] = 'val'
    for t in test_ids:
        split_labels[t] = 'test'

    print(f"Assigned splits chronologically: {num_train} train, {num_val} val, {num_test} test.")
    print("Train Test IDs:", train_ids)
    print("Validation Test IDs:", val_ids)
    print("Test Test IDs:", test_ids)

    return split_labels


# ----------------------------------------
# 6) Plot Training History
# ----------------------------------------
def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss over Epochs')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    if 'classification_output_accuracy' in history.history:
        ax[1].plot(history.history['classification_output_accuracy'], label='Train Acc')
        if 'val_classification_output_accuracy' in history.history:
            ax[1].plot(history.history['val_classification_output_accuracy'], label='Val Acc')
        ax[1].set_title('Classification Accuracy over Epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

    plt.suptitle("Training & Validation Metrics")
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# 7) Prepare Sequences (Pad)
# ----------------------------------------
def prepare_sequences_all_samples(
    tests_features: dict,
    tests_heatmaps: dict,
    unique_test_ids: list,
    split_labels: dict,
    max_sequence_length: int = None
) -> tuple:

    sequences = []
    sequence_labels = []
    sequence_splits = []
    test_sequence_ids = []

    if max_sequence_length is None:
        max_sequence_length = max(len(features) for features in tests_features.values())
        print(f"INFO: Maximum sequence length set to {max_sequence_length} based on the longest test.")

    for test_id in unique_test_ids:
        features = tests_features[test_id]
        label_heatmap = tests_heatmaps[test_id]


        padded_sequence = pad_sequences(
            [features],
            maxlen=max_sequence_length,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )[0]

        sequences.append(padded_sequence)
        sequence_labels.append(label_heatmap)
        sequence_splits.append(split_labels[test_id])
        test_sequence_ids.append(test_id)

        
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    sequence_splits = np.array(sequence_splits)
    test_sequence_ids = np.array(test_sequence_ids)

    print(f"INFO: Prepared {len(sequences)} sequences with padding.")
    print(f"Sequence Labels shape: {sequence_labels.shape}")
    print(f"Sequence Splits distribution: {dict(zip(*np.unique(sequence_splits, return_counts=True)))}")

    return sequences, sequence_labels, sequence_splits, test_sequence_ids


# ----------------------------------------
# 8) CNN + Attention Model
# ----------------------------------------
def cnn_attention_model(input_shape, heatmap_shape, config):
    """
    CNN + Attention model for set aggregation:
    - 1D convolutions over the sample dimension
    - attention weighting
    - classification + heatmap outputs
    """
    l2_reg = regularizers.l2(1e-5)
    dropout_rate = 0.4

    max_samples, feature_dim = input_shape
    inputs = layers.Input(shape=(max_samples, feature_dim), name='input_layer')

    # Convolutional feature extraction over samples
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2_reg)(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Attention mechanism
    attention_scores = layers.Dense(1, activation=None, kernel_regularizer=l2_reg)(x)  # (batch, max_samples, 1)
    attention_scores = tf.nn.softmax(attention_scores, axis=1)  # softmax over sample dimension
    set_representation = tf.reduce_sum(x * attention_scores, axis=1)  # (batch, 64)

    # Classification head
    dense_class = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(set_representation)
    dense_class = layers.Dropout(dropout_rate)(dense_class)
    classification_output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l2_reg, name='classification_output')(dense_class)

    # Decoder for heatmap
    latent = layers.Dense(16 * 16 * 256, activation='relu', kernel_regularizer=l2_reg)(set_representation)
    reshape = layers.Reshape((16, 16, 256))(latent)

    deconv1 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2_reg)(reshape)
    deconv2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2_reg)(deconv1)
    deconv3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2_reg)(deconv2)
    deconv4 = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2_reg)(deconv3)
    deconv5 = layers.Conv2DTranspose(16, (3, 3), strides=(1, 3), padding='same', activation='relu', kernel_regularizer=l2_reg)(deconv4)

    heatmap_output = layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', activation='sigmoid', kernel_regularizer=l2_reg, name='heatmap_output')(deconv5)

    model = Model(inputs=inputs, outputs=[classification_output, heatmap_output])
    return model


# ----------------------------------------
# 9) Custom Losses & Metrics
# ----------------------------------------
def combined_heatmap_loss(y_true, y_pred, alpha=0.5):
    """
    Combined Weighted BCE + Dice loss.
    
    Args:
        y_true: Ground-truth mask, shape (batch, H, W, 1).
        y_pred: Model predictions, shape (batch, H, W, 1).
        alpha: Weight for the weighted BCE portion. 
               The Dice portion is weighted (1 - alpha).

    Returns:
        A scalar loss value.
    """
    # 1) Weighted BCE (same as your custom_heatmap_loss)
    weight_map = tf.cast(y_true > 0, tf.float32) * 2.0 + 1.0
    epsilon = 1e-7
    bce = - (y_true * tf.math.log(y_pred + epsilon) 
             + (1 - y_true) * tf.math.log(1 - y_pred + epsilon))
    weighted_bce = bce * weight_map
    bce_loss = tf.reduce_mean(weighted_bce)

    # 2) Dice loss = 1 - dice_coefficient
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=[1, 2]) 
                                           + tf.reduce_sum(y_pred, axis=[1, 2]) + smooth)
    dice = tf.reduce_mean(dice)  # average dice across batch
    dice_loss = 1.0 - dice

    # 3) Combine
    combined = alpha * bce_loss + (1.0 - alpha) * dice_loss
    return combined


def dice_coefficient_per_channel(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) + smooth)
    return tf.reduce_mean(dice)


# ----------------------------------------
# 10) Visualization
# ----------------------------------------
def visualize_predictions(
    test_ids: list,
    predictions: np.ndarray,
    true_heatmaps: np.ndarray,
    preloaded_images: dict,
    image_shape: tuple = (256, 256),
    sigma: float = 2.0,
    labels_root_dir: str = "Labels"
):
    """
    Visualize predictions vs. ground truth on a composite image.
    This is done after training for final analysis.
    """
    perspective_map = {
        'A': 'Arch_Intrados',
        'B': 'North_Spandrel_Wall',
        'C': 'South_Spandrel_Wall'
    }

    def compute_weight_map_np(true_heatmap, sigma=5.0, normalize=True):
        if true_heatmap.ndim == 2:
            true_heatmap = true_heatmap[..., None]
        heatmap = np.zeros_like(true_heatmap, dtype=np.float32)
        for c in range(true_heatmap.shape[-1]):
            ch = true_heatmap[..., c]
            gh = gaussian_filter(ch, sigma=sigma)
            if normalize and gh.max() > 0:
                gh /= gh.max()
            heatmap[..., c] = gh
        return heatmap

    for idx, test_id in enumerate(test_ids):
        pred = predictions[idx]   # (256,768,1)
        true = true_heatmaps[idx] # (256,768,1)


        enhanced_true = compute_weight_map_np(true, sigma=sigma)
        enhanced_pred = compute_weight_map_np(pred, sigma=sigma)

        true_color = plt.cm.coolwarm(enhanced_true[:, :, 0])[:, :, :3]
        pred_color = plt.cm.coolwarm(enhanced_pred[:, :, 0])[:, :, :3]

        def load_and_resize(perspective):
            label_prefix = perspective_map[perspective]
            label_image_filename = f"{label_prefix}_T{test_id}.png"
            label_image_path = os.path.join(labels_root_dir, f"Test_{test_id}", label_image_filename)
            if os.path.exists(label_image_path):
                img = cv2.imread(label_image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                target_h, target_w = image_shape
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pad_w = target_w - new_w
                pad_h = target_h - new_h
                top, bottom = pad_h // 2, pad_h - (pad_h // 2)
                left, right = pad_w // 2, pad_w - (pad_w // 2)
                img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                return img_padded / 255.0
            else:
                return np.zeros((256, 256, 3), dtype=np.float32)

        bg_A = load_and_resize('A')
        bg_B = load_and_resize('B')
        bg_C = load_and_resize('C')

        background_img = np.concatenate([bg_A, bg_B, bg_C], axis=1)

        overlay_true = np.clip(0.7 * background_img + 0.3 * true_color, 0, 1)
        overlay_pred = np.clip(0.7 * background_img + 0.3 * pred_color, 0, 1)

        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Test {test_id} - Composite Heatmap Visualization', fontsize=20)

        plt.subplot(1, 2, 1)
        plt.imshow(overlay_true)
        plt.title('True Composite Heatmap')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay_pred)
        plt.title('Predicted Composite Heatmap')
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_name = f'visualization_Test_{test_id}.png'
        plt.savefig(save_name)
        plt.close()
        print(f"INFO: Saved visualization for Test {test_id} as {save_name}")


# ----------------------------------------
# 11) Main Execution
# ----------------------------------------
def main():
    start_time = time.time()

    ORIGINAL_IMAGE_HEIGHT = 2479
    ORIGINAL_IMAGE_WIDTH = 3508
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    root_dir = "Clustered Results"
    labels_dir = "Labels"
    max_sequence_length = None

    perspective_map = {
        'A': 'Arch_Intrados',
        'B': 'North_Spandrel_Wall',
        'C': 'South_Spandrel_Wall'
    }

    config = {
        'learning_rate': 0.00005,
        'epochs': 150,
        'batch_size': 1,
        'patience': 15
    }

    # Hardcode classification labels: 1 => "new damage", 0 => "no new damage"
    classification_labels = {
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 0,
        6: 1,
        7: 1,
        8: 0,
        9: 1,
        10: 0,
        11: 1,
        12: 1,
        13: 1,
        14: 0,
        15: 1,
        16: 1,
        17: 1,
        18: 1,
        19: 1,
        20: 1,
        21: 1,
        22: 1,
        25: 1,
    }

    # 1) Preload label images
    print("Preloading all label images...")
    preloaded_images = preload_label_images(
        labels_root_dir=labels_dir,
        perspective_map=perspective_map,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    print("Completed preloading label images.")

    # 2) Load data and apply smoothing
    print("\nLoading data as tests (with smoothing)...")
    tests_features, tests_heatmaps, unique_test_ids, max_test_number = load_data_as_tests_all_samples(
        root_dir=root_dir,
        preloaded_images=preloaded_images,
        perspective_map=perspective_map,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        smoothing_sigma=2.0  # Adjust sigma as desired
    )

    if len(unique_test_ids) == 0:
        print("ERROR: No tests loaded. Exiting.")
        exit(1)

    print(f"Number of Tests Loaded: {len(unique_test_ids)}")
    sample_count_per_test = {test_id: len(features) for test_id, features in tests_features.items()}
    print(f"Sample counts per test: {sample_count_per_test}")

    # 3) Assign splits
    print("\nAssigning splits to each Test ID chronologically...")
    split_labels = chronological_splits(unique_test_ids, train_ratio=0.7, val_ratio=0.15)

    # 4) Prepare sequences
    print("\nPreparing sequences with all samples...")
    X_sequences, y_sequence_heatmaps, sequence_splits, test_sequence_ids = prepare_sequences_all_samples(
        tests_features,
        tests_heatmaps,
        unique_test_ids,
        split_labels,
        max_sequence_length=max_sequence_length
    )
    if max_sequence_length is None:
        max_sequence_length = X_sequences.shape[1]

 

    # Split into train/val/test
    X_train = X_sequences[sequence_splits == 'train']
    y_train_heatmap = y_sequence_heatmaps[sequence_splits == 'train']
    train_test_ids = test_sequence_ids[sequence_splits == 'train']


    X_val = X_sequences[sequence_splits == 'val']
    y_val_heatmap = y_sequence_heatmaps[sequence_splits == 'val']
    val_test_ids = test_sequence_ids[sequence_splits == 'val']

    X_test = X_sequences[sequence_splits == 'test']
    y_test_heatmap = y_sequence_heatmaps[sequence_splits == 'test']
    test_test_ids = test_sequence_ids[sequence_splits == 'test']

    print(f"Training set size: {X_train.shape[0]} sequences")
    print(f"Validation set size: {X_val.shape[0]} sequences")
    print(f"Testing set size: {X_test.shape[0]} sequences")

    # 5) Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    n_train_samples, seq_length, feature_dim = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, feature_dim)).reshape(n_train_samples, seq_length, feature_dim)
    n_val_samples = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(-1, feature_dim)).reshape(n_val_samples, seq_length, feature_dim)
    n_test_samples = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, feature_dim)).reshape(n_test_samples, seq_length, feature_dim)

    # 6) Compute classification labels using the HARDCODED dictionary
    print("\nSetting classification labels based on 'new damage' dictionary...")
    def get_new_damage_label(test_id: int) -> int:
        return classification_labels.get(test_id, 0)

    y_train_class = np.array([get_new_damage_label(tid) for tid in train_test_ids], dtype=np.float32).reshape(-1, 1)
    y_val_class   = np.array([get_new_damage_label(tid) for tid in val_test_ids], dtype=np.float32).reshape(-1, 1)
    y_test_class  = np.array([get_new_damage_label(tid) for tid in test_test_ids], dtype=np.float32).reshape(-1, 1)

    print("Train Classification Labels:", y_train_class.squeeze())
    print("Val   Classification Labels:", y_val_class.squeeze())
    print("Test  Classification Labels:", y_test_class.squeeze())

    # 7) Prepare heatmap labels for training
    y_train_heatmap_with_weights = y_train_heatmap
    y_val_heatmap_with_weights = y_val_heatmap
    y_test_heatmap_with_weights = y_test_heatmap

    # 8) Build Model
    print("\nBuilding the CNN + Attention model...")
    model = cnn_attention_model(input_shape=(seq_length, feature_dim), heatmap_shape=(256, 256, 1), config=config)
    model.summary()

    print("Compiling the model...")
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss={
            'classification_output': 'binary_crossentropy',
            'heatmap_output':  lambda y_true, y_pred: combined_heatmap_loss(y_true, y_pred, alpha=0.5)
        },
        loss_weights={
            'classification_output': 1.0,
            'heatmap_output': 1.0
        },
        metrics={
            'classification_output': ['accuracy'],
            'heatmap_output': [dice_coefficient_per_channel]
        }
    )

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, min_lr=0.0000075, verbose=1)
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath='best_deepsets_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        lr_reducer
    ]

    # 9) Train
    print("\nTraining the model...")
    history = model.fit(
        X_train_scaled,
        {
            'classification_output': y_train_class,
            'heatmap_output': y_train_heatmap_with_weights
        },
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(X_val_scaled, {
            'classification_output': y_val_class,
            'heatmap_output': y_val_heatmap_with_weights
        }),
        callbacks=callbacks_list,
        verbose=1
    )

    plot_training_history(history)

    #########################
    # PRINT TRAIN/VAL SUMMARY
    #########################
    def print_summary_of_metrics(history_obj):
        """Print final train & validation metrics from the Keras History object."""
        final_epoch_idx = len(history_obj.history['loss']) - 1  # Index of last epoch
        
        # --- Training metrics at final epoch ---
        train_loss = history_obj.history['loss'][final_epoch_idx]
        train_clf_loss = history_obj.history['classification_output_loss'][final_epoch_idx]
        train_clf_acc = history_obj.history['classification_output_accuracy'][final_epoch_idx]
        train_dice = history_obj.history['heatmap_output_dice_coefficient_per_channel'][final_epoch_idx]
        
        # --- Validation metrics (if present) ---
        if 'val_loss' in history_obj.history:
            val_loss = history_obj.history['val_loss'][final_epoch_idx]
            val_clf_loss = history_obj.history['val_classification_output_loss'][final_epoch_idx]
            val_clf_acc = history_obj.history['val_classification_output_accuracy'][final_epoch_idx]
            val_dice = history_obj.history['val_heatmap_output_dice_coefficient_per_channel'][final_epoch_idx]
        else:
            val_loss = val_clf_loss = val_clf_acc = val_dice = None

        print("\n=== Final Training Metrics ===")
        print(f"Overall Loss:             {train_loss:.4f}")
        print(f"Classification Loss:      {train_clf_loss:.4f}")
        print(f"Classification Accuracy:  {train_clf_acc:.4f}")
        print(f"Heatmap Dice Coefficient: {train_dice:.4f}")

        if val_loss is not None:
            print("\n=== Final Validation Metrics ===")
            print(f"Overall Loss:             {val_loss:.4f}")
            print(f"Classification Loss:      {val_clf_loss:.4f}")
            print(f"Classification Accuracy:  {val_clf_acc:.4f}")
            print(f"Heatmap Dice Coefficient: {val_dice:.4f}")
        else:
            print("\n(No validation metrics were recorded.)")

    # Call the summary printing function
    print_summary_of_metrics(history)

    # 10) Evaluate on TEST
    print("\nEvaluating the model on the TEST set only...")
    evaluation = model.evaluate(
        X_test_scaled,
        {
            'classification_output': y_test_class,
            'heatmap_output': y_test_heatmap_with_weights
        },
        verbose=1
    )
    # Indices in 'evaluation' correspond to:
    #   0 => total loss
    #   1 => classification_output_loss
    #   2 => heatmap_output_loss
    #   3 => classification_output_accuracy
    #   4 => heatmap_output_dice_coefficient_per_channel
    test_loss = evaluation[0]
    test_clf_loss = evaluation[1]
    test_heatmap_loss = evaluation[2]
    test_clf_acc = evaluation[3]
    test_heatmap_dice = evaluation[4]

    print("\n=== Test Metrics ===")
    print(f"Overall Test Loss:             {test_loss:.4f}")
    print(f"Test Classification Loss:      {test_clf_loss:.4f}")
    print(f"Test Classification Accuracy:  {test_clf_acc:.4f}")
    print(f"Test Heatmap Loss:             {test_heatmap_loss:.4f}")
    print(f"Test Heatmap Dice Coefficient: {test_heatmap_dice:.4f}")

    ##########################################
    # ADDITIONAL CLASSIFICATION METRICS (Precision, Recall, F1)
    ##########################################
    print("\nGenerating predictions for classification branch on TEST set...")
    y_pred_all = model.predict(X_test_scaled)
    y_pred_class_probs = y_pred_all[0]
    y_pred_class = (y_pred_class_probs > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    # Compute precision, recall, F1, and support
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_class, y_pred_class, average='binary', zero_division=0
    )

    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test_class, y_pred_class, zero_division=0))

    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test_class, y_pred_class))

    # Summarize these additional metrics:
    print("\n=== Additional Classification Metrics (Binary) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 11) Generate predictions for heatmap
    print("\nGenerating predictions for heatmap on TEST SET ONLY...")
    predictions_test = y_pred_all[1]  # second output => heatmap

    # Keep your existing visualization
    visualize_predictions(
        test_ids=test_test_ids,
        predictions=predictions_test,
        true_heatmaps=y_test_heatmap,
        preloaded_images=preloaded_images,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        sigma=2.0,
        labels_root_dir=labels_dir
    )

    # 12) Done
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal Runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL: An unhandled exception occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
