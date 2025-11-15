import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers  # type: ignore
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import tensorflow.keras.backend as K  # type: ignore

#Submission Version


######################################
# Configuration
######################################
DATA_DIR = "Data"
LABELS_DIR = "Labels"
IMAGE_SHAPE = (256, 768)  # Output heatmap size (height, width)
SKIP_TESTS = [23, 24]     # Tests to skip
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
LEARNING_RATE = 1e-4
EPOCHS = 280
BATCH_SIZE = 16
PATIENCE = 15

AUGMENT_TEST_ID = 25    # Test number to augment
N_AUGMENT = 5           # How many augmented samples per original sample
NOISE_LEVEL = 0.05      # Noise level (5%)


# Mapping for perspective images Labels
perspective_map = {
    'A': 'Arch_Intrados',
    'B': 'North_Spandrel_Wall',
    'C': 'South_Spandrel_Wall'
}

######################################
# Step 1: Load, Combine Data and Augment Data
######################################

def load_accelerometer_data(data_dir=DATA_DIR, skip_tests=SKIP_TESTS):
    EXPECTED_ROWS = 12000
    test_dirs = [d for d in glob.glob(os.path.join(data_dir, "Test_*")) if os.path.isdir(d)] #itarate over test folders
    tests_data = {}
    for td in test_dirs:
        match = re.search(r"Test[_]?(\d+)", os.path.basename(td))
        if not match:
            continue
        test_number = int(match.group(1))
        if test_number in skip_tests:
            print(f"INFO: Skipping Test {test_number}.")
            continue

        csv_files = sorted(glob.glob(os.path.join(td, f"Test_{test_number}_*.csv")))
        if len(csv_files) == 0:
            print(f"WARNING: No CSV files found for Test {test_number} in directory {td}.")
            continue

        samples = []
        for cf in csv_files:
            try:
                df = pd.read_csv(cf)
                accel_cols = [c for c in df.columns if "Accel [G]" in c]
                if not accel_cols:
                    print(f"WARNING: No acceleration columns found in file {cf}.")
                    continue
                data_arr = df[accel_cols].values.astype(np.float32)  # (12000, 12)
                # Truncate if necessary
                if data_arr.shape[0] > EXPECTED_ROWS:
                    print(f"Truncating file {cf} from {data_arr.shape} to ({EXPECTED_ROWS}, {data_arr.shape[1]}).")
                    data_arr = data_arr[:EXPECTED_ROWS, :]
                samples.append(data_arr)  # Each sample is (12000, 12)
            except Exception as e:
                print(f"ERROR: Failed to read {cf}. Error: {e}")
                continue

        if len(samples) == 0:
            print(f"WARNING: No valid acceleration data found for Test {test_number}.")
            continue

        tests_data[test_number] = samples  # List of samples, each (12000, 12)
        print(f"INFO: Loaded Test {test_number} with {len(samples)} samples, each with shape {samples[0].shape}.")
    return tests_data

def add_noise_to_data(sample, noise_level=NOISE_LEVEL):
    """
    Add Gaussian noise to the given sample. The data has been normalized, 
    so we can directly add noise scaled by noise_level.
    """
    noise = np.random.randn(*sample.shape) * noise_level
    return sample + noise

def create_augmented_data(tests_data, tests_heatmaps, augment_test_id, n_augment=N_AUGMENT):
    """
    Create augmented data for the specified test_id and return them as separate lists.
    """
    if augment_test_id not in tests_data:
        print(f"WARNING: Test {augment_test_id} not found. No augmentation performed.")
        return [], []
    
    original_samples = tests_data[augment_test_id]  # List of (12000, 12)
    heatmap = tests_heatmaps[augment_test_id]       # (256,768,1)

    augmented_X = []
    augmented_Y = []

    for sample in original_samples:
        # Each sample is (12000, 12)
        for _ in range(n_augment):
            aug_sample = add_noise_to_data(sample, noise_level=NOISE_LEVEL)
            augmented_X.append(aug_sample)
            augmented_Y.append(heatmap)

    return augmented_X, augmented_Y

######################################
# Compute Global Mean and Std for Normalization; apply high pass filter
######################################
def compute_global_mean_std(tests_data):
    # Stack all tests data along time to get global stats
    # tests_data[test_id] shape: list of (12000, 12)
    all_data = np.concatenate([samples for samples in tests_data.values()], axis=0)  # (num_samples, 12000, 12)
    mean = np.mean(all_data, axis=(0,1), keepdims=True)  # (1,1,12)
    std = np.std(all_data, axis=(0,1), keepdims=True)    # (1,1,12)
    print(f"INFO: Computed global mean and std for normalization.")
    return mean, std

def normalize_tests_data(tests_data, mean, std):
    # Normalize each channel
    # tests_data[test_id]: (12,60000)
    for k in tests_data.keys():
        tests_data[k] = (tests_data[k] - mean) / (std + 1e-8)
    print(f"INFO: Normalized all test data.")
    return tests_data

def apply_high_pass_filter(data, cutoff=10.0, fs=200.0, order=5):
    """
    Apply a high-pass Butterworth filter to the data.

    Parameters:
    - data: numpy array, shape (channels, time_steps)
    - cutoff: float, cutoff frequency in Hz
    - fs: float, sampling frequency in Hz
    - order: int, order of the filter

    Returns:
    - filtered_data: numpy array, same shape as data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)  # Changed to 'high'
    # Apply the filter to each channel
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        try:
            filtered_data[i] = filtfilt(b, a, data[i])
        except Exception as e:
            print(f"ERROR: Filtering failed for channel {i}. Error: {e}")
            filtered_data[i] = data[i]  # Fallback to unfiltered data
    return filtered_data

def filter_tests_data(tests_data, cutoff=10.0, fs=200.0, order=5):
    """
    Apply a high-pass filter to every sample of every test in tests_data.
    
    Parameters:
    - tests_data: dict of {test_id: [array_of_shape_(12000, 12), ...]}
    - cutoff: cutoff frequency for the high-pass filter
    - fs: sampling frequency (Hz)
    - order: order of the Butterworth filter

    Returns:
    - filtered_data: dict with the same structure, but filtered
    """
    filtered_data = {}
    for test_id, samples in tests_data.items():
        filtered_samples = []
        for sample in samples:
            # sample shape: (12000, 12)
            # apply_high_pass_filter expects (channels, time_steps), so we transpose
            # then transpose back
            filtered_sample = apply_high_pass_filter(sample.T, cutoff=cutoff, fs=fs, order=order).T
            filtered_samples.append(filtered_sample)
        filtered_data[test_id] = filtered_samples
    return filtered_data

######################################
# Step 3: Label Preprocessing
######################################

def load_perspective_image(test_id, perspective, image_shape=IMAGE_SHAPE):
    """ load an image file corresponding to a specific test case (test_id) 
        and perspective, process it, and return a padded and resized version"""
    
    label_prefix = perspective_map[perspective]
    fname = os.path.join(LABELS_DIR, f"Test_{test_id}", f"{label_prefix}_T{test_id}.png")
    if not os.path.exists(fname):
        print(f"WARNING: Label file {fname} does not exist.")
        return None
    img = cv2.imread(fname)
    if img is None:
        print(f"WARNING: Failed to read image {fname}.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    target_h, target_w = (image_shape[0], image_shape[1]//3)  # Each perspective occupies 1/3 width
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_padded

def load_and_combine_labels(labels_dir=LABELS_DIR, test_id=1, image_shape=IMAGE_SHAPE):

    #compute binary mask which detects red regions (damage regions)
    def red_mask_from_image(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2).astype(np.float32) / 255.0
        red_mask = red_mask[..., None]  # (256,768,1)
        return red_mask

    # Load and resize each perspective
    img_A = load_perspective_image(test_id, 'A', image_shape)
    img_B = load_perspective_image(test_id, 'B', image_shape)
    img_C = load_perspective_image(test_id, 'C', image_shape)

    # If any perspective is missing, replace with black image
    if img_A is None:
        img_A = np.zeros((image_shape[0], image_shape[1]//3, 3), dtype=np.float32)
    if img_B is None:
        img_B = np.zeros((image_shape[0], image_shape[1]//3, 3), dtype=np.float32)
    if img_C is None:
        img_C = np.zeros((image_shape[0], image_shape[1]//3, 3), dtype=np.float32)

    # Concatenate perspectives horizontally to form one image containing all perspectives (256,768,3)
    combined_image = np.concatenate([img_A, img_B, img_C], axis=1)  # (256,768,3)

    # Create red mask from combined image
    mask = red_mask_from_image(combined_image)  # (256,768,1)

    return mask  # (256,768,1)

def load_labels_for_all_tests(tests_data, labels_dir=LABELS_DIR):
    """
    Load all image Labels for all tests and compute gaussian heatmaps
    """
    tests_heatmaps = {}
    for test_id in tests_data.keys():
        combined_mask = load_and_combine_labels(labels_dir, test_id, IMAGE_SHAPE)
        heatmap = create_gaussian_heatmap(combined_mask, sigma=5.0)  # (256,768,1)
        tests_heatmaps[test_id] = heatmap
        print(f"INFO: Processed labels for Test {test_id}.")
    return tests_heatmaps

def create_gaussian_heatmap(binary_mask, sigma=2.0):
    """
    computes gaussian heatmap from binary masks (red masks)
    """
    # Apply Gaussian filter
    heatmap = gaussian_filter(binary_mask, sigma=sigma)
    # Normalize to [0,1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    print(f"INFO: Created Gaussian heatmap with shape {heatmap.shape}.")
    return heatmap

######################################
# Data Splitting
######################################

def chronological_splits(unique_test_ids, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    This function splits a list of unique test IDs into training, validation, and testing sets in a chronological manner.
    The earliest test IDs are allocated to the training set, followed by the validation set, and the latest IDs to the testing set.

    Parameters:
    unique_test_ids (list): A sorted list of unique test IDs in chronological order.
    train_ratio (float): The proportion of test IDs to allocate to the training set.
    val_ratio (float): The proportion of test IDs to allocate to the validation set.

    Returns:
    split_labels (dict): A dictionary mapping each test ID to its split ('train', 'val', or 'test').
    """
    num_tests = len(unique_test_ids)
    num_train = int(train_ratio * num_tests)
    num_val = int(val_ratio * num_tests)
    train_ids = unique_test_ids[:num_train]
    val_ids = unique_test_ids[num_train:num_train+num_val]
    test_ids = unique_test_ids[num_train+num_val:]
    split_labels = {}
    for t in train_ids:
        split_labels[t] = 'train'
    for t in val_ids:
        split_labels[t] = 'val'
    for t in test_ids:
        split_labels[t] = 'test'
    print(f"INFO: Data split into {len(train_ids)} training, {len(val_ids)} validation, and {len(test_ids)} testing samples.")
    return split_labels

######################################
# Create Datasets
######################################

def create_datasets(tests_data, tests_heatmaps, split_labels, max_test_number):
    """
    This function generates training, validation, and testing datasets from input 
    test data and corresponding heatmaps. It also allows augmentation of training 
    data for a specific test ID.

    Parameters:
    tests_data (dict): A dictionary where keys are test IDs and values are lists of sample data for each test.
    tests_heatmaps (dict): A dictionary where keys are test IDs and values are corresponding heatmaps.
    split_labels (dict): A mapping of test IDs to their respective splits ('train', 'val', or 'test').
    max_test_number (int): The maximum test ID number, used to normalize test numbers as a feature.

    Returns:
    X_train, Y_train (np.ndarray): Features and labels for the training set.
    X_val, Y_val (np.ndarray): Features and labels for the validation set.
    X_test, Y_test (np.ndarray): Features and labels for the testing set.
    test_ids_final (np.ndarray): Test IDs included in the testing set.
    """
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []
    test_ids_final = []

    # Create the normal (non-augmented) datasets first
    for test_id, samples in tests_data.items():
        split = split_labels[test_id]
        heatmap = tests_heatmaps[test_id]  # (256,768,1)
        normalized_test_number = test_id / max_test_number  # Normalize test number between 0 and 1

        for sample in samples:
            # Add test number as a feature
            test_number_feature = np.full((sample.shape[0], 1), normalized_test_number, dtype=np.float32)
            X_sample_with_test_num = np.concatenate((sample, test_number_feature), axis=1)  # (12000,13)
            Y_sample = heatmap

            if split == 'train':
                X_train.append(X_sample_with_test_num)
                Y_train.append(Y_sample)
            elif split == 'val':
                X_val.append(X_sample_with_test_num)
                Y_val.append(Y_sample)
            elif split == 'test':
                X_test.append(X_sample_with_test_num)
                Y_test.append(Y_sample)
                test_ids_final.append(test_id)

    # Generate augmented data for the chosen test (AUGMENT_TEST_ID)
    aug_X, aug_Y = create_augmented_data(tests_data, tests_heatmaps, AUGMENT_TEST_ID, N_AUGMENT)

    # Add augmented samples to the training set
    if AUGMENT_TEST_ID in tests_data and len(aug_X) > 0:
        normalized_test_number = AUGMENT_TEST_ID / max_test_number
        for i in range(len(aug_X)):
            sample = aug_X[i]
            test_number_feature = np.full((sample.shape[0], 1), normalized_test_number, dtype=np.float32)
            aug_sample_with_test_num = np.concatenate((sample, test_number_feature), axis=1)
            X_train.append(aug_sample_with_test_num)
            Y_train.append(aug_Y[i])
        print(f"INFO: Added {len(aug_X)} augmented samples for Test {AUGMENT_TEST_ID} to the training set.")

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    test_ids_final = np.array(test_ids_final)

    print(f"INFO: Created datasets - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}.")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_ids_final

######################################
# Model Architecture and Loss (Encoder-Decoder CNN)
######################################
def CNN_model(input_timesteps=12000, input_features=13):
    """
    1D-2D CNN model for feature extraction and damage localization using accelerometer data.
    Processes time series data, encodes it with 1D convolutions, and decodes it into a 2D heatmap.
    Inspiration for this architecture is from Dang. et. al.

    Architecture:
    - Input: Time series of shape (input_timesteps, input_features).
    - Encoder: Two 1D convolutional blocks with max pooling to extract features.
    - Bottleneck: Dense layers for feature compression.
    - Decoder: Reshapes and upscales features using transposed convolutions to generate a 2D heatmap.
    - Output: A (256, 768, 1) heatmap with sigmoid activation, indicating damage probabilities.

    Parameters:
    - input_timesteps (int): Number of time steps in input (default: 12000).
    - input_features (int): Features per time step (default: 12).

    Returns:
    - model: Keras Model object.
    """
    
    # Encoder
    input_layer = layers.Input(shape=(input_timesteps, input_features))
    
    # First Conv Block
    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)  # (3000, 64)
    
    # Second Conv Block
    x = layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)  # (750, 128)
    
    # Flatten and Bottleneck Dense Layers
    x = layers.Flatten()(x)  # (750 * 128,) = (96000,)
    x = layers.Dense(units=1024, activation='relu')(x)  # (1024)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=512, activation='relu')(x)  # (512)
    
    # Decoder
    # Project to a lower-dimensional space suitable for upsampling
    x = layers.Dense(units=32*32, activation='relu')(x)  # (1024)
    x = layers.Reshape(target_shape=(32, 32, 1))(x)  # (32, 32, 1)
    
    # Upsampling Blocks
    # First Upsampling Block
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # (64, 64, 64)
    x = layers.BatchNormalization()(x)
    
    # Second Upsampling Block
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # (128, 128, 32)
    x = layers.BatchNormalization()(x)
    
    # Third Upsampling Block
    x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # (256, 256, 16)
    x = layers.BatchNormalization()(x)
    
    # Final Upsampling Block to reach desired width (768)
    # Since height is already 256, adjust width from 256 to 768 (3x)
    x = layers.Conv2DTranspose(filters=8, kernel_size=(3,3), strides=(1,3), padding='same', activation='relu')(x)  # (256, 768, 8)
    x = layers.BatchNormalization()(x)
    
    # Output Layer
    output = layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(x)  # (256,768,1)
    
    # Define Model
    model = models.Model(inputs=input_layer, outputs=output)
    return model

def dice_loss(y_true, y_pred):
    """
    Computes the Dice Loss between the true and predicted masks.
    
    Parameters:
    - y_true: Ground truth masks, shape (batch_size, height, width, 1)
    - y_pred: Predicted masks, shape (batch_size, height, width, 1)
    
    Returns:
    - Dice loss value
    """
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error Loss
    
    Parameters:
    - y_true: Ground truth heatmaps, shape (batch_size, height, width, 1)
    - y_pred: Predicted heatmaps, shape (batch_size, height, width, 1)
    
    Returns:
    - mse: Mean Squared Error loss value
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def iou_loss(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU) Loss
    
    Parameters:
    - y_true: Ground truth masks, shape (batch_size, height, width, 1)
    - y_pred: Predicted masks, shape (batch_size, height, width, 1)
    - smooth: Smoothing factor to avoid division by zero
    
    Returns:
    - iou: IoU loss value
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - K.mean(iou)

def combined_loss(y_true, y_pred):
    """
    Combined Loss Function with Adjustable Weights for Each Component.
    
    Parameters:
    - y_true: Ground truth heatmaps, shape (batch_size, height, width, 1)
    - y_pred: Predicted heatmaps, shape (batch_size, height, width, 1)
    
    Returns:
    - total_loss: Weighted sum of individual loss components
    """
    # Compute individual losses
    mse = mse_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    
    # Define weights for each loss component
    lambda_mse = 0.8   # Weight for MSE Loss
    lambda_dice = 1.2   # Weight for Dice Loss
    lambda_iou = 0.1  # Weight for IoU Loss
    
    # Combine the losses with their respective weights
    total_loss = (lambda_mse * mse) + (lambda_dice * dice) + (lambda_iou * iou)
    
    return total_loss

##############################################
# Visualization Function and Performance Plots
##############################################

def load_perspective_background(test_id, image_shape=(256,768)):
    """
    Load perspective images (A, B, C) for a test ID and combine them into a composite background.
    
    Parameters:
    - test_id (int): ID of the test.
    - image_shape (tuple): Shape of the combined image (default: (256, 768)).
    
    Returns:
    - background (ndarray): Combined background image of shape (256, 768, 3).
    """
    # Load each perspective image (A,B,C) and form a composite background
    def load_and_resize_perspective(perspective):
        label_prefix = perspective_map[perspective]
        fname = os.path.join(LABELS_DIR, f"Test_{test_id}", f"{label_prefix}_T{test_id}.png")
        if not os.path.exists(fname):
            print(f"WARNING: Background image file {fname} does not exist. Using black image.")
            return np.zeros((image_shape[0], image_shape[1]//3, 3), dtype=np.float32)
        img = cv2.imread(fname)
        if img is None:
            print(f"WARNING: Failed to read background image {fname}. Using black image.")
            return np.zeros((image_shape[0], image_shape[1]//3, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        target_h, target_w = (image_shape[0], image_shape[1]//3)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w*scale), int(h*scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top, bottom = pad_h//2, pad_h-(pad_h//2)
        left, right = pad_w//2, pad_w-(pad_w//2)
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=[0,0,0])
        return img_padded.astype(np.float32)/255.0

    bg_A = load_and_resize_perspective('A')  # (256,256,3)
    bg_B = load_and_resize_perspective('B')  # (256,256,3)
    bg_C = load_and_resize_perspective('C')  # (256,256,3)
    background = np.concatenate([bg_A, bg_B, bg_C], axis=1) # (256,768,3)
    return background

def visualize_predictions(test_ids, predictions, true_heatmaps, image_shape=(256,768)):
    """
    Visualizes averaged predictions and true heatmaps for each test number.

    Parameters:
    - test_ids: numpy array of test numbers corresponding to each prediction
    - predictions: numpy array, shape (num_samples, 256,768,1)
    - true_heatmaps: numpy array, shape (num_samples, 256,768,1)
    - image_shape: tuple, (height, width)
    """
    unique_test_ids = np.unique(test_ids)
    for test_id in unique_test_ids:
        # Get all samples for this test_id
        idxs = np.where(test_ids == test_id)[0]
        preds = predictions[idxs]  # (num_samples_per_test, 256,768,1) -> we have 5 predictions per Test ID which we average.
        trues = true_heatmaps[idxs]  # (num_samples_per_test, 256,768,1) -> 1 true GT mask per Test ID
        
        # Average the predictions and true heatmaps
        pred_avg = np.mean(preds, axis=0)  # (256,768,1)
        true_avg = trues[0]  # (256,768,1)
    
        # Normalize
        if true_avg.max() > 0:
            true_normalized = true_avg / true_avg.max()
        else:
            true_normalized = true_avg
        if pred_avg.max() > 0:
            pred_normalized = pred_avg / pred_avg.max()
        else:
            pred_normalized = pred_avg
    
        # Convert to color maps
        true_color = plt.cm.coolwarm(true_normalized.squeeze())[:, :, :3]  # (256,768,3)
        pred_color = plt.cm.coolwarm(pred_normalized.squeeze())[:, :, :3]
    
        # Load composite background
        background_img = load_perspective_background(test_id, image_shape)
    
        # Overlay heatmaps
        overlay_true = np.clip(0.7*background_img + 0.3*true_color, 0, 1)
        overlay_pred = np.clip(0.7*background_img + 0.3*pred_color, 0, 1)
    
        # Plot side by side
        plt.figure(figsize=(18,6))
        plt.suptitle(f'Test {test_id} - Composite Heatmap Visualization', fontsize=20)
    
        plt.subplot(1,2,1)
        plt.imshow(overlay_true)
        plt.title('True Composite Heatmap')
        plt.axis('off')
    
        plt.subplot(1,2,2)
        plt.imshow(overlay_pred)
        plt.title('Predicted Composite Heatmap')
        plt.axis('off')
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        print(f"INFO: Visualized predictions for Test {test_id}.")

def plot_metrics(history):
    """
    Plots training and validation metrics over epochs.
    
    Parameters:
    - history: History object returned by model.fit()
    """
    metrics = ['loss', 'mse', 'dice_coefficient', 'iou', 'mae', 'binary_accuracy', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=f'Training {metric.capitalize()}', color='blue')
        plt.plot(history.history.get(f'val_{metric}', []), label=f'Validation {metric.capitalize()}', color='orange')
        plt.title(f'Training and Validation {metric.capitalize()} Over Epochs', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


######################################
# Performance Metrics
######################################

# Dice Coefficient
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

# Intersection over Union (IoU)
def iou(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

# Binary Accuracy
def binary_accuracy(y_true, y_pred):
    thresholded_pred = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, thresholded_pred), tf.float32))

# Precision
def precision(y_true, y_pred):
    smooth = 1e-6
    true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    return (true_positives + smooth) / (predicted_positives + smooth)

# Recall
def recall(y_true, y_pred):
    smooth = 1e-6
    true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
    possible_positives = tf.reduce_sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

def display_final_metrics(history):
    print("\nINFO: Final Epoch Metrics:")
    for metric in history.history:
        # Skip if it's a validation metric name, 
        # we'll handle val_ inside the same loop.
        if metric.startswith('val_'):
            continue

        # train_metric is the final training value for 'metric'
        train_metric_list = history.history[metric]
        train_metric = train_metric_list[-1]  # last epoch’s value

        # find the corresponding validation metric list, if it exists
        val_metric_list = history.history.get(f"val_{metric}", None)
        if val_metric_list is not None:
            # final validation epoch’s value
            val_metric = val_metric_list[-1]
            print(f"{metric.replace('_', ' ').capitalize()}: "
                  f"Train = {train_metric:.4f}, Val = {val_metric:.4f}")
        else:
            print(f"{metric.replace('_', ' ').capitalize()}: {train_metric:.4f}")

######################################
# Main Execution
######################################

def main():
    # Step 1: Load data
    print("INFO: Starting data loading...")
    tests_data = load_accelerometer_data(DATA_DIR, SKIP_TESTS)
    if len(tests_data) == 0:
        print("ERROR: No data loaded. Exiting.")
        return
    
    # Step 2: Chronological split of test IDs
    unique_ids = sorted(tests_data.keys())
    print("INFO: Splitting data into training, validation, and testing sets...")
    split_labels = chronological_splits(unique_ids)
    
    # Step 3: High-pass filter all raw data BEFORE normalization
    print("INFO: Applying high-pass filter to all test data (train/val/test)...")
    tests_data = filter_tests_data(tests_data, cutoff=10.0, fs=200.0, order=5)
    
    # Step 4: Compute mean/std from TRAIN portion only
    train_ids = [t_id for t_id in unique_ids if split_labels[t_id] == 'train']
    tests_data_train_only = {t_id: tests_data[t_id] for t_id in train_ids}
    print("INFO: Computing global mean/std from filtered TRAIN set only...")
    mean, std = compute_global_mean_std(tests_data_train_only) 
    # -> shape (1,1,12), as per your function

    # Step 5: Normalize all data (train/val/test) using train stats
    print("INFO: Normalizing all test data with train-derived mean/std...")
    tests_data = normalize_tests_data(tests_data, mean, std)
    
    # Step 6: Load labels and create heatmaps
    print("INFO: Loading labels and creating Gaussian heatmaps...")
    tests_heatmaps = load_labels_for_all_tests(tests_data, LABELS_DIR)
    
    # Step 7: Compute max test number for test ID normalization
    if unique_ids:
        max_test_number = max(unique_ids)
    else:
        max_test_number = 1  # fallback to avoid div-by-zero
    
    # Step 8: Create train/val/test datasets 
    print("INFO: Creating final train/val/test datasets...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, test_ids_final = create_datasets(
        tests_data, tests_heatmaps, split_labels, max_test_number
    )
    
    # Step 9: Verify shapes
    print(f"INFO: Train set shape: {X_train.shape}, "
          f"Validation set shape: {X_val.shape}, "
          f"Test set shape: {X_test.shape}.")
    
    # Step 10: Build the model
    print("INFO: Building the 1D CNN model...")
    model = CNN_model(input_timesteps=X_train.shape[1], input_features=X_train.shape[2])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=combined_loss,
        metrics=[
            'mse',
            dice_coefficient,
            iou,
            'mae',
            binary_accuracy,
            precision,
            recall
        ]
    )
    model.summary()
    
    # Step 11: Callbacks
    print("INFO: Setting up callbacks...")
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        # Uncomment the following line to enable ModelCheckpoint
        # callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    ]
    
    # Step 12: Train
    print("INFO: Starting model training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs
    )
    
    # Step 13: Plot training and validation metrics
    print("INFO: Plotting training and validation metrics...")
    plot_metrics(history)
    
    # Step 14: Evaluate on test set
    print("INFO: Evaluating the model on the test set...")
    eval_result = model.evaluate(X_test, Y_test)
    print(f"INFO: Test Loss: {eval_result[0]:.4f}, "
          f"Test MSE: {eval_result[1]:.4f}, "
          f"Test Dice Coefficient: {eval_result[2]:.4f}, "
          f"Test IoU: {eval_result[3]:.4f}, "
          f"Test MAE: {eval_result[4]:.4f}, "
          f"Test Binary Accuracy: {eval_result[5]:.4f}, "
          f"Test Precision: {eval_result[6]:.4f}, "
          f"Test Recall: {eval_result[7]:.4f}")
    
    # Step 15: Display training and validation metrics
    display_final_metrics(history)
    
    # Step 16: Predict and visualize
    print("INFO: Generating predictions on the test set...")
    Y_pred = model.predict(X_test)

    print("INFO: Visualizing predictions...")
    visualize_predictions(test_ids_final, Y_pred, Y_test, image_shape=IMAGE_SHAPE)


if __name__ == "__main__":
    main()
