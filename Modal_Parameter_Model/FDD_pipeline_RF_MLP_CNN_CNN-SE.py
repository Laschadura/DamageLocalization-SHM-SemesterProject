import os
import re
import glob
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from sklearn.model_selection import train_test_split, GroupShuffleSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    balanced_accuracy_score
)
from skmultilearn.model_selection import IterativeStratification
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers  # type: ignore
import tensorflow.keras.backend as K  # type: ignore
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for clarity

# Uses FDD dataset. Has simple CNN, CNN with SE Block, MLP and RF models. I could submit this.

# constants
MAC_THRESHOLD = 0.7

# Need this to make ready for submission

def compute_class_weights_dict(y):
    # y is a binary matrix of shape (num_samples, num_classes)
    import numpy as np
    class_totals = np.sum(y, axis=0)
    class_weight = {}
    total_samples = y.shape[0]
    for i in range(len(class_totals)):
        class_weight[i] = total_samples / (len(class_totals) * class_totals[i] + 1e-6)
    return class_weight

def compute_sample_weights(y, class_weight):
    # y is a binary matrix of shape (num_samples, num_classes)
    import numpy as np
    sample_weights = np.sum(y * np.array(list(class_weight.values())), axis=1)
    return sample_weights


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
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"No CSV files found in directory: {root_dir}")
        return [], None

    print(f"Found {len(csv_files)} CSV files.")

    for csv_file in csv_files:
        # Attempt to extract test_id and excitation
        match = re.search(r"Test_(\d+)_([A-Za-z_]+)\.csv", os.path.basename(csv_file))
        if not match:
            # Fallback to extract excitation and assign synthetic test_id
            match = re.search(r"Modes_Test_([A-Za-z_]+)\.csv", os.path.basename(csv_file))
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

        # Group by Mode Number
        if 'Mode Number' not in csv_data.columns:
            continue

        grouped = csv_data.groupby('Mode Number')
        for mode_number, group in grouped:  # Corrected loop to capture mode_number
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
                    'mode_number': mode_number,  # Use mode_number from the loop
                    'segment_number': idx + 1,
                    'frequency': frequency,
                    'sample_avg_mac': avg_mac,
                    'sample_median_mac': median_mac,
                    'sample_mac_std_dev': mac_std_dev,
                    'sample_mode_shape': mode_shape
                })

    try:
        labels_df = pd.read_csv(labels_path)
        labels_df.index = labels_df['Test Number'].astype(int)  # Ensure index is integer type
        labels_df = labels_df.drop(columns=['Test Number'])  # Drop the 'Test Number' column if it's now the index
        print(f"Labels loaded successfully. Labels shape: {labels_df.shape}")
    except Exception as e:
        print(f"Error reading labels file {labels_path}: {e}")
        labels_df = None

    print(f"Extracted {len(data)} mode shapes in total.")
    if len(data) > 0:
        print(f"Sample data entry: {data[0]}")  # Print a single sample entry

    return data, labels_df


def encode_categorical_features(X_train, X_test):
    """
    Encodes non-numeric features like 'excitation' and 'closest_cluster' using one-hot encoding.
    Fits the encoder on X_train and transforms both X_train and X_test.

    Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        X_test (pd.DataFrame): Testing feature DataFrame.

    Returns:
        Tuple: (X_train_encoded, X_test_encoded)
    """
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['excitation', 'closest_cluster']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_features])

    encoded_train = encoder.transform(X_train[categorical_features])
    encoded_test = encoder.transform(X_test[categorical_features])

    encoded_columns = encoder.get_feature_names_out(categorical_features)

    # Create DataFrames for encoded features
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_columns, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_columns, index=X_test.index)

    # Drop the original categorical columns
    X_train = X_train.drop(categorical_features, axis=1)
    X_test = X_test.drop(categorical_features, axis=1)

    # Concatenate the encoded features with the rest of the data
    X_train = pd.concat([X_train, encoded_train_df], axis=1)
    X_test = pd.concat([X_test, encoded_test_df], axis=1)

    return X_train, X_test


def extract_features(mode_shapes_data, reference_modes):
    """
    Extracts features including MAC statistics, closest cluster, excitation, and mode shapes.
    Does not encode categorical features yet.
    
    Args:
        mode_shapes_data (list): Mode shapes with associated features.
        reference_modes (pd.DataFrame): DataFrame containing reference modes (from `reference_modes.csv`).
    
    Returns:
        pd.DataFrame: Feature matrix for ML models.
    """
    features = []
    
    # Extract reference mode shapes and cluster labels
    import ast  # Safer than using eval
    reference_shapes = np.array(reference_modes['Mode Shape'].apply(lambda x: np.array(ast.literal_eval(x))).tolist())
    reference_clusters = reference_modes['Cluster'].values
    
    for entry in mode_shapes_data:
        mode_shape = entry['sample_mode_shape']
        frequency = entry['frequency']
        sample_avg_mac = entry['sample_avg_mac']
        sample_median_mac = entry['sample_median_mac']
        sample_mac_std_dev = entry['sample_mac_std_dev']
        excitation = entry['excitation']  # Include excitation

        # Calculate MAC scores with reference modes
        mac_scores = calculate_mac_matrix(np.array([mode_shape]), reference_shapes).flatten()
        max_mac = np.max(mac_scores)
        avg_mac = np.mean(mac_scores)
        std_mac = np.std(mac_scores)

        # Identify the closest cluster
        closest_cluster_index = np.argmax(mac_scores)
        closest_cluster = reference_clusters[closest_cluster_index]

        # Expand mode_shape into individual features
        mode_shape_features = {f"channel_{i+1}": val for i, val in enumerate(mode_shape)}

        # Add features for this segment
        feature_dict = {
            'test_id': entry['test_id'],
            'excitation': excitation,
            'mode_number': entry['mode_number'],
            'frequency': frequency,
            'sample_avg_mac': sample_avg_mac,
            'sample_median_mac': sample_median_mac,
            'sample_mac_std_dev': sample_mac_std_dev,
            'max_mac_to_reference': max_mac,
            'avg_mac_to_reference': avg_mac,
            'std_mac_to_reference': std_mac,
            'closest_cluster': closest_cluster
        }

        # Update feature_dict with mode_shape features
        feature_dict.update(mode_shape_features)

        features.append(feature_dict)

    # Create DataFrame without encoding categorical features yet
    feature_df = pd.DataFrame(features)

    return feature_df


def calculate_mac_matrix(Phi, reference_modes):
    """
    Calculate MAC (Modal Assurance Criterion) matrix between mode shapes and reference modes.
    
    Args:
        Phi (np.ndarray): Array of mode shapes (n_modes, n_features).
        reference_modes (np.ndarray): Array of reference mode shapes (n_references, n_features).
    
    Returns:
        np.ndarray: MAC matrix of shape (n_modes, n_references).
    """
    # Numerator of MAC: squared absolute dot products
    numerator = np.abs(Phi @ reference_modes.T) ** 2
    
    # Denominator of MAC: product of squared norms
    denom_modes = np.sum(Phi ** 2, axis=1, keepdims=True)
    denom_refs = np.sum(reference_modes ** 2, axis=1, keepdims=True).T
    mac_matrix = numerator / (denom_modes @ denom_refs)
    return mac_matrix


def data_split(feature_df, labels_df, test_size=0.3):
    """
    Splits data into training and testing sets based on groups without stratification.
    
    Args:
        feature_df (pd.DataFrame): Feature matrix with multiple samples per `test_id`.
        labels_df (pd.DataFrame): Labels DataFrame indexed by `test_id`.
        test_size (float): Proportion of data to allocate to the test set.
    
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    # Create groups to prevent data leakage
    groups = feature_df[['test_id', 'excitation', 'mode_number']].apply(
        lambda x: f"{x['test_id']}_{x['excitation']}_{x['mode_number']}", axis=1
    )
    
    # Map labels to each sample based on `test_id`
    try:
        y = labels_df.loc[feature_df['test_id']].values  # Shape: (3402, num_classes)
    except KeyError as e:
        missing_ids = feature_df['test_id'][~feature_df['test_id'].isin(labels_df.index)].unique()
        raise KeyError(f"Missing labels for test_id(s): {missing_ids}") from e
    
    # Initialize GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    
    # Perform the split
    train_idx, test_idx = next(splitter.split(feature_df, y, groups=groups))
    
    # Split the features
    X_train, X_test = feature_df.iloc[train_idx].reset_index(drop=True), feature_df.iloc[test_idx].reset_index(drop=True)
    
    # Split the labels
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Verify class distribution
    train_counts = y_train.sum(axis=0)
    test_counts = y_test.sum(axis=0)
    total_counts = y.sum(axis=0)
    
    print("\nClass distribution in dataset:")
    for i, label in enumerate(labels_df.columns):
        print(f"{label}: Total={int(total_counts[i])}, Train={int(train_counts[i])}, Test={int(test_counts[i])}")
    
    return X_train, X_test, y_train, y_test


def simple_cnn_model(input_shape, output_size, config):
    inputs = layers.Input(shape=(input_shape, 1))
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    outputs = layers.Dense(output_size, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model


def cnn_with_se_blocks(input_shape, output_size, config):
    inputs = layers.Input(shape=(input_shape, 1))
    x = layers.Conv1D(filters=config['layer_sizes'][0], kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(config['l2_reg']))(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    outputs = layers.Dense(output_size, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model


def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se_shape = (1, filters)
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    x = layers.multiply([input_tensor, se])
    return x


def mlp_model(input_dim, output_size, config):
    """
    Defines a Multilayer Perceptron (MLP) model.

    Args:
        input_dim (int): Number of input features.
        output_size (int): Number of output classes.
        config (dict): Configuration dictionary containing hyperparameters.

    Returns:
        keras.Model: Compiled MLP model.
    """
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=input_dim))
    model.add(layers.Dropout(config['dropout_rate']))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(config['dropout_rate']))
    model.add(layers.Dense(output_size, activation='sigmoid'))
    return model


def evaluation(results, X_test, y_test, zone_labels):
    """
    Calculates and visualizes evaluation metrics for each model, including damage detection,
    zone localization metrics, per-zone performance, ROC-AUC curves, and training/validation loss plots.

    Creates three separate figures:
    1. Damage Detection and Zone Localization Metrics (Accuracy, Precision, Recall, F1-Score)
    2. ROC-AUC Curve and Training vs Validation Loss
    3. Per-Zone Performance Metrics (Precision, Recall, F1-Score, Accuracy)

    Args:
        results (dict): Dictionary containing model results.
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray): True labels for test data.
        zone_labels (list): List of zone labels.

    Returns:
        dict: Dictionary containing calculated metrics for each model.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
        auc,
        confusion_matrix
    )
    from sklearn.utils.multiclass import type_of_target

    metrics = {}
    per_class_metrics = {}  # To store per-zone metrics for all models

    # Define base colors for each model
    model_names = list(results.keys())
    model_colors = {
        'RandomForest': '#1f77b4',     # Blue
        'Simple_CNN': '#ff7f0e',       # Orange
        'CNN_with_SE': '#2ca02c',      # Green
        'MLP': '#d62728',               # Red
        # Add more models here if needed
    }

    # Define metrics to plot, including accuracy
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1-score']

    # Assign each model a single color in the palette
    palette = {model_name: model_colors.get(model_name, '#333333') for model_name in model_names}

    for model_name, result in results.items():
        y_pred = result['predictions']

        # Determine if the task is multi-label
        if type_of_target(y_test) == 'multilabel-indicator':
            is_multilabel = True
        else:
            is_multilabel = False

        if is_multilabel:
            # Damage detection: presence of any damage in the test sample
            y_test_damage = (y_test.sum(axis=1) > 0).astype(int)
            y_pred_damage = (y_pred.sum(axis=1) > 0).astype(int)

            # Calculate metrics for damage detection
            damage_accuracy = accuracy_score(y_test_damage, y_pred_damage)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_damage, y_pred_damage, average='binary', zero_division=0)

            # Initialize variables for ROC AUC
            roc_auc = None

            # ROC AUC Score for models that provide probabilities
            if model_name == 'RandomForest':
                # Random Forest provides predict_proba method
                y_pred_proba = np.array([proba[:, 1] for proba in result['model'].predict_proba(X_test)]).T
            else:
                # Neural network models provide probabilities via predict()
                if model_name == 'MLP':
                    y_pred_proba = result['model'].predict(X_test)
                else:
                    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    y_pred_proba = result['model'].predict(X_test_reshaped)

            # Compute ROC AUC for each class
            roc_auc_list = []
            for i in range(y_test.shape[1]):
                if len(np.unique(y_test[:, i])) > 1:
                    roc_auc_score_i = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
                    roc_auc_list.append(roc_auc_score_i)
            roc_auc = np.mean(roc_auc_list) if roc_auc_list else 'Undefined'
            print(f"{model_name} ROC AUC Score: {roc_auc}")
            result['roc_auc'] = roc_auc

            # Zone localization metrics
            zone_accuracy = accuracy_score(y_test, y_pred)
            zone_classification_report = classification_report(
                y_test, y_pred, target_names=zone_labels, zero_division=0, output_dict=True)

            # Save results to metrics dictionary
            metrics[model_name] = {
                'damage_detection': {
                    'accuracy': damage_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'zone_localization': {
                    'accuracy': zone_accuracy,
                    'classification_report': zone_classification_report  # This is now a dict
                },
                'roc_auc': roc_auc,
                'config': result.get('config', {})
            }

            # Extract per-class metrics, including accuracy
            per_class_metrics[model_name] = {}
            for zone in zone_labels:
                if zone in zone_classification_report:
                    # Calculate accuracy for the current zone
                    zone_idx = zone_labels.tolist().index(zone)
                    y_true_zone = y_test[:, zone_idx]
                    y_pred_zone = y_pred[:, zone_idx]
                    zone_accuracy = accuracy_score(y_true_zone, y_pred_zone)

                    per_class_metrics[model_name][zone] = {
                        'accuracy': zone_accuracy,
                        'precision': zone_classification_report[zone]['precision'],
                        'recall': zone_classification_report[zone]['recall'],
                        'f1-score': zone_classification_report[zone]['f1-score'],
                        'support': zone_classification_report[zone]['support']
                    }
                else:
                    # If the class is not present in the report (e.g., no samples), set metrics to zero
                    per_class_metrics[model_name][zone] = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0
                    }

        else:
            # Handle non-multi-label cases if necessary
            # For completeness, but likely not needed in your scenario
            y_pred = result['predictions']
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(
                y_test, y_pred, target_names=zone_labels, zero_division=0, output_dict=True)
            metrics[model_name] = {
                'damage_detection': {
                    'accuracy': accuracy,
                    'precision': classification_rep['accuracy']['precision'],
                    'recall': classification_rep['accuracy']['recall'],
                    'f1_score': classification_rep['accuracy']['f1-score']
                },
                'zone_localization': {
                    'accuracy': accuracy,
                    'classification_report': classification_rep  # This is now a dict
                },
                'roc_auc': 'Undefined',  # Not calculated
                'config': result.get('config', {})
            }

    # Create separate figures as per user request

    # **Figure 1: Damage Detection and Zone Localization Metrics**
    plt.figure(figsize=(14, 12))
    plt.suptitle("Figure 1: Damage Detection and Zone Localization Metrics", fontsize=18)

    # Prepare data for Damage Detection Metrics
    damage_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    damage_data = []
    for model_name in model_names:
        for metric in damage_metrics:
            score = metrics[model_name]['damage_detection'][metric]
            damage_data.append({
                'Model': model_name,
                'Metric': metric.replace('_', ' ').title(),
                'Score': score
            })
    damage_df = pd.DataFrame(damage_data)

    # Prepare data for Zone Localization Metrics
    zone_metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    zone_data = []
    for model_name in model_names:
        for metric in zone_metrics:
            if metric == 'accuracy':
                score = metrics[model_name]['zone_localization']['accuracy']
            else:
                # Calculate the average metric across all zones
                scores = [metrics[model_name]['zone_localization']['classification_report'][zone][metric] 
                          for zone in zone_labels]
                score = np.mean(scores)
            zone_data.append({
                'Model': model_name,
                'Metric': metric.replace('_', ' ').title(),
                'Score': score
            })
    zone_df = pd.DataFrame(zone_data)

    # Subplot 1: Damage Detection Metrics
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='Score', hue='Metric', data=damage_df, palette='viridis', ci=None)
    plt.ylabel("Score")
    plt.title("Damage Detection Metrics")
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Subplot 2: Zone Localization Metrics
    plt.subplot(2, 1, 2)
    sns.barplot(x='Model', y='Score', hue='Metric', data=zone_df, palette='magma', ci=None)
    plt.ylabel("Score")
    plt.title("Zone Localization Metrics")
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
    plt.show()

    # **Figure 2: ROC-AUC Curve and Training vs Validation Loss**
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    fig.suptitle("Figure 2: ROC-AUC Curve and Training vs Validation Loss", fontsize=16)

    # Subplot 1: ROC-AUC Curve
    ax1 = axes[0]
    for model_name in model_names:
        result = results[model_name]
        if model_name == 'RandomForest':
            y_pred_proba = np.array([proba[:, 1] for proba in result['model'].predict_proba(X_test)]).T
        else:
            if model_name == 'MLP':
                y_pred_proba = result['model'].predict(X_test)
            else:
                X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                y_pred_proba = result['model'].predict(X_test_reshaped)

        # Compute ROC curve and AUC for each class and plot average
        fpr = dict()
        tpr = dict()
        roc_auc_dict = dict()

        for i in range(y_test.shape[1]):
            if len(np.unique(y_test[:, i])) > 1:
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
                roc_auc_dict[i] = auc(fpr[i], tpr[i])

        if len(roc_auc_dict) > 0:
            all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in fpr:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= len(fpr)
            ax1.plot(all_fpr, mean_tpr, color=model_colors.get(model_name, '#333333'),
                     label=f'{model_name} (AUC = {np.mean(list(roc_auc_dict.values())):.2f})')

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC-AUC Curve")
    ax1.legend(loc="lower right")

    # Subplot 2: Training vs Validation Loss
    ax2 = axes[1]
    for model_name in model_names:
        if model_name != 'RandomForest':
            history = results[model_name]['history']
            ax2.plot(history['loss'], label=f'{model_name} Training Loss')
            ax2.plot(history['val_loss'], label=f'{model_name} Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss for Neural Network Models')
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # **Figure 3: Per-Zone Performance Metrics**
    plt.figure(figsize=(20, 10))
    plt.suptitle("Figure 3: Per-Zone Performance Metrics", fontsize=16)

    # Prepare data for plotting
    plot_data = []
    for model_name in model_names:
        for metric in metrics_to_plot:
            for zone in zone_labels:
                score = per_class_metrics[model_name][zone][metric]
                plot_data.append({
                    'Zone': zone,
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': score
                })

    plot_df = pd.DataFrame(plot_data)

    # Create a grouped bar plot with spacing between models
    sns.barplot(x='Zone', y='Score', hue='Metric', data=plot_df, palette='Set2', ci=None)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Score")
    plt.title('Per-Zone Performance Metrics')
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # **Print Detailed Classification Reports**
    for model_name in model_names:
        print(f"\n{model_name} - Damage Detection Metrics:")
        print(f"Accuracy: {metrics[model_name]['damage_detection']['accuracy']:.4f}, "
              f"Precision: {metrics[model_name]['damage_detection']['precision']:.4f}, "
              f"Recall: {metrics[model_name]['damage_detection']['recall']:.4f}, "
              f"F1 Score: {metrics[model_name]['damage_detection']['f1_score']:.4f}")
        print(f"\n{model_name} - Zone Localization Metrics:")
        # Since we already have the classification report in dict form, we can print it nicely
        report_df = pd.DataFrame(metrics[model_name]['zone_localization']['classification_report']).transpose()
        print(report_df)

    # **Print Number of Samples per Class**
    print("\nNumber of Samples per Class in Test Set:")
    class_counts = y_test.sum(axis=0)
    for i, label in enumerate(zone_labels):
        print(f"{label}: {int(class_counts[i])}")

    return metrics


if __name__ == "__main__":
    # List of label files to process
    label_files = ["Damage_Labels_10.csv", "Damage_Labels_20.csv"]
    results_summary = {}

    # Load reference modes from CSV
    reference_modes_path = "reference_modes.csv"
    reference_modes = pd.read_csv(reference_modes_path)

    for labels_path in label_files:
        print(f"\nRunning with label set: {labels_path}\n")

        # Step 1: Load data and labels
        mode_shapes_data, labels_df = load_data_grouped_by_mode(labels_path=labels_path)

        # Ensure labels are loaded
        if labels_df is None or labels_df.empty:
            print(f"Skipping label set {labels_path} due to loading issues.")
            continue

        # Step 2: Extract features
        feature_df = extract_features(mode_shapes_data, reference_modes)

        # Step 3: Split data into training and testing sets using the revised function
        X_train, X_test, y_train, y_test = data_split(feature_df, labels_df, test_size=0.2)

        print(X_train.iloc[0])

        # Step 4: Encode non-numeric features after splitting
        X_train, X_test = encode_categorical_features(X_train, X_test)

        print(X_train.iloc[0])

        # Step 5: Normalize features for NN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # **Adjust Random Forest Hyperparameters**
        from sklearn.model_selection import RandomizedSearchCV

        # Define the base model
        base_rf = MultiOutputClassifier(RandomForestClassifier(random_state=42))

        # Updated hyperparameter grid
        rf_param_grid = {
            'estimator__n_estimators': [200, 250],
            'estimator__max_depth': [30, 40],
            'estimator__min_samples_split': [5, 6],
            'estimator__min_samples_leaf': [2],
            'estimator__max_features': ['log2'],
            'estimator__class_weight': ['balanced']
        }

        print("\nTuning Random Forest hyperparameters with extended grid...")
        rf_random_search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=rf_param_grid,
            n_iter=20,        # Adjust as needed
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )
        rf_random_search.fit(X_train, y_train)

        # Get the best model
        best_rf = rf_random_search.best_estimator_
        print(f"Best Random Forest parameters: {rf_random_search.best_params_}")

        # Evaluate the best model
        y_pred_rf = best_rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest validation accuracy: {rf_accuracy}")

        # **Neural Network Models**
        models_dict = {
            'Simple_CNN': simple_cnn_model,
            'CNN_with_SE': cnn_with_se_blocks,
            'MLP': mlp_model,  # Added MLP model
        }

        # Hyperparameter options
        # Define separate hyperparameters for CNNs and MLP
        cnn_layer_sizes_options = [[128]]  # For CNN_with_SE
        cnn_learning_rate_options = [0.001]
        cnn_dropout_rate_options = [0.3]
        cnn_l2_reg_options = [0.001]
        cnn_batch_size_options = [16]

        mlp_hidden_layers_options = [[128, 64]]  # Number of units in hidden layers
        mlp_learning_rate_options = [0.001]
        mlp_dropout_rate_options = [0.3]
        mlp_l2_reg_options = [0.001]  # Not directly used in MLP but can be integrated via regularizers
        mlp_batch_size_options = [16]

        # Create hyperparameter combinations
        hyperparameter_combinations_cnn = list(itertools.product(
            cnn_layer_sizes_options,
            cnn_learning_rate_options,
            cnn_dropout_rate_options,
            cnn_l2_reg_options,
            cnn_batch_size_options
        ))

        hyperparameter_combinations_mlp = list(itertools.product(
            mlp_hidden_layers_options,
            mlp_learning_rate_options,
            mlp_dropout_rate_options,
            mlp_l2_reg_options,
            mlp_batch_size_options
        ))

        # Initialize best models dictionary
        best_nn_models = {}

        for model_name, model_fn in models_dict.items():
            best_accuracy = 0
            best_config = None
            best_model = None
            best_history = None
            best_y_pred_nn = None

            if model_name == 'MLP':
                hyperparameter_combinations = hyperparameter_combinations_mlp
            else:
                hyperparameter_combinations = hyperparameter_combinations_cnn

            for (layer_sizes, learning_rate, dropout_rate, l2_reg, batch_size) in hyperparameter_combinations:
                if model_name == 'MLP':
                    nn_config = {
                        'hidden_layers': layer_sizes,  # For MLP, layer_sizes refer to hidden layers
                        'dropout_rate': dropout_rate,
                        'use_leaky_relu': False,
                        'l2_reg': l2_reg,
                        'learning_rate': learning_rate,
                        'epochs': 100,  # Increased epochs for MLP if needed
                        'batch_size': batch_size,
                        'patience': 10
                    }
                else:
                    nn_config = {
                        'layer_sizes': layer_sizes,
                        'dropout_rate': dropout_rate,
                        'use_leaky_relu': False,
                        'l2_reg': l2_reg,
                        'learning_rate': learning_rate,
                        'epochs': 50,
                        'batch_size': batch_size,
                        'patience': 10
                    }

                print(f"\nTesting {model_name} with config: {nn_config}")

                if model_name == 'MLP':
                    input_dim = X_train_scaled.shape[1]
                    output_size = y_train.shape[1]
                    nn_model = model_fn(input_dim, output_size, nn_config)
                else:
                    input_shape = X_train_scaled.shape[1]
                    output_size = y_train.shape[1]
                    nn_model = model_fn(input_shape, output_size, nn_config)

                if model_name == 'MLP':
                    nn_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=nn_config['learning_rate']),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                else:
                    nn_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=nn_config['learning_rate']),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )

                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=nn_config['patience'],
                    restore_best_weights=True
                )
                lr_reduction = callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )

                # Dynamic learning rate for Simple_CNN
                if model_name == 'Simple_CNN':
                    lr_schedule = callbacks.LearningRateScheduler(
                        lambda epoch: nn_config['learning_rate'] * (0.95 ** epoch), verbose=0)
                    callbacks_list = [early_stopping, lr_reduction, lr_schedule]
                else:
                    callbacks_list = [early_stopping, lr_reduction]

                # Compute class weights
                class_weight = compute_class_weights_dict(y_train)

                # Compute sample weights
                sample_weights = compute_sample_weights(y_train, class_weight)

                # Train the model using sample weights
                if model_name == 'MLP':
                    history = nn_model.fit(
                        X_train_scaled, y_train,
                        sample_weight=sample_weights,
                        epochs=nn_config['epochs'],
                        batch_size=nn_config['batch_size'],
                        validation_data=(X_test_scaled, y_test),
                        callbacks=callbacks_list,
                        verbose=0
                    )
                else:
                    # Ensure input data has the correct shape
                    X_train_reshaped = X_train_scaled.reshape(
                        (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
                    X_test_reshaped = X_test_scaled.reshape(
                        (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

                    history = nn_model.fit(
                        X_train_reshaped, y_train,
                        sample_weight=sample_weights,
                        epochs=nn_config['epochs'],
                        batch_size=nn_config['batch_size'],
                        validation_data=(X_test_reshaped, y_test),
                        callbacks=callbacks_list,
                        verbose=0
                    )

                if model_name == 'MLP':
                    y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype(int)
                else:
                    y_pred_nn = (nn_model.predict(X_test_reshaped) > 0.5).astype(int)

                accuracy_nn = accuracy_score(y_test, y_pred_nn)
                print(f"{model_name} validation accuracy: {accuracy_nn}")

                if accuracy_nn > best_accuracy:
                    best_accuracy = accuracy_nn
                    best_config = nn_config.copy()
                    best_model = nn_model
                    best_history = history.history
                    best_y_pred_nn = y_pred_nn  # Store predictions of the best model

            # Store the best model for this architecture
            best_nn_models[model_name] = {
                'model': best_model,
                'accuracy': best_accuracy,
                'predictions': best_y_pred_nn,
                'history': best_history,
                'config': best_config
            }

        # **Step 5: Evaluation**
        # Collect results for evaluation
        results = best_nn_models
        results['RandomForest'] = {
            'model': best_rf,
            'accuracy': rf_accuracy,
            'predictions': y_pred_rf,
            'config': rf_random_search.best_params_
        }

        # Evaluate models
        metrics = evaluation(results, X_test_scaled, y_test, labels_df.columns)

        # Store results
        results_summary[labels_path] = metrics
