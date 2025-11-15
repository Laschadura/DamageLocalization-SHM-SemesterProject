import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------- SKLEARN ----------------- #
from sklearn.model_selection import (
    train_test_split, 
    GroupShuffleSplit, 
    GroupKFold
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc
)

# ------------- TENSORFLOW / KERAS ----------- #
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers # type: ignore
import tensorflow.keras.backend as K # type: ignore

# Uses SSI modal parameter dataset
#
# Load mode shapes, modal parameters, and damage labels.
# Expand mode_shape into separate numeric columns.
# One-hot encode cluster.
# Remove mode_number.
# Split data into training and validation sets while preserving group integrity.
# Train ML models (RF + MLP).
#
# We have a multi-zone damage label matrix (1 => damage, 0 => no damage).
# For "overall damage detection," if any zone is 1, we consider the sample damaged.

# --------------- CONSTANTS --------------- #
MAC_THRESHOLD = 0.7
REFERENCE_LABELS = ["Mode A", "Mode B", "Mode C", "Mode D", "Noise"] 
# Mode A, B, C and D are mode shape representatives we have chosen as most likely to be structural modes (based on manual inspection of mode shape plots)
# If a sample is not assigned to one of the modes (if none of the MAC scores against any of the modes is above 0.7) we assign the sample to the Noise cluster.

# ---------- DATA LOADING & PREP ---------- #
def load_data_and_labels(
    mode_shapes_dir="ModeShapes", 
    modal_params_dir="ModalParameters", 
    labels_path="Damage_Labels_20.csv"
):
    """
    Load reference modes, mode shapes, modal parameters, and zone-based damage labels.
    Assumes that 'final_reference_modes.csv' is in the working directory.
    """
    reference_modes = pd.read_csv("final_reference_modes.csv").values.T

    # Collect mode shapes & modal parameters
    data = []
    pattern = r"Mode_Shapes_Test_(\d+)_(.+?)(?:_aug(\d+))?\.csv"
    for shape_file in glob.glob(os.path.join(mode_shapes_dir, "Mode_Shapes_Test_*_*.csv")):
        match = re.search(pattern, os.path.basename(shape_file))
        if match:
            test_id, excitation, aug_suffix = int(match.group(1)), match.group(2), match.group(3)
            is_augmented = bool(aug_suffix)
            shape_data = pd.read_csv(shape_file).values[1:]  # skip header row

            # Match the corresponding modal parameters file
            params_file_pattern = f"Modal_Parameters_Test_{test_id}_{excitation}"
            if is_augmented:
                params_file_pattern += f"_aug{aug_suffix}"
            params_file_pattern += ".csv"

            params_file_path = os.path.join(modal_params_dir, params_file_pattern)
            if os.path.exists(params_file_path):
                params_data = pd.read_csv(params_file_path).values[1:, 1:]  # skip header row, skip first col
                for idx, (mode_shape, params) in enumerate(zip(shape_data, params_data)):
                    # mode_shape is a 1D array of sensor values
                    data.append({
                        'file': shape_file,
                        'test_id': test_id,
                        'excitation': excitation,
                        'mode_number': idx + 1,  # We'll exclude it from final features
                        'mode_shape': mode_shape,
                        'frequency': params[0],
                        'damping': params[1],
                        'is_augmented': is_augmented
                    })
            else:
                print(f"Warning: No matching modal parameters for {shape_file}")
        else:
            print(f"Warning: Filename {shape_file} not matching pattern.")

    # Load labels
    labels_df = pd.read_csv(labels_path).set_index('Test Number')
    return reference_modes, data, labels_df


def calculate_mac_matrix(Phi, reference_modes):
    """
    Vectorized MAC calculation between each row of Phi and each row of reference_modes.
    Returns shape = (n_modes, n_references).
    """
    numerator = np.abs(Phi @ reference_modes.T) ** 2
    denom_modes = np.sum(Phi ** 2, axis=1, keepdims=True)
    denom_refs = np.sum(reference_modes ** 2, axis=1, keepdims=True).T
    mac = numerator / (denom_modes @ denom_refs)
    return mac


def filter_and_assign_clusters(mode_shapes_data, reference_modes, mac_threshold=MAC_THRESHOLD):
    """
    For each mode in mode_shapes_data, assign cluster label based on best matching reference mode
    if MAC >= mac_threshold; else label="Noise".
    """
    Phi = np.array([entry['mode_shape'] for entry in mode_shapes_data])
    mac_matrix = calculate_mac_matrix(Phi, reference_modes)
    assigned = []
    for i, entry in enumerate(mode_shapes_data):
        best_idx = np.argmax(mac_matrix[i])
        best_val = mac_matrix[i][best_idx]
        label = REFERENCE_LABELS[best_idx] if best_val >= mac_threshold else "Noise"
        assigned.append({**entry, 'cluster': label, 'mac_value': best_val})
    return assigned


def split_data(selected_modes, labels_df, test_size=0.2):
    """
    Create a DataFrame from selected_modes, merge with labels, group by (test_id, excitation),
    then do a group-aware train/test split.

    Adjustments:
    - Expand 'mode_shape' into multiple numeric columns (ms_0, ms_1, ...).
    - One-hot encode 'cluster'.
    - Exclude 'mode_number' from final features.
    """
    df = pd.DataFrame(selected_modes)

    # Extract test_id, excitation from filename
    pattern = r"Mode_Shapes_Test_(\d+)_([A-Z]+(?:_[A-Z]+)*(?:_\d+)?)"
    def extract_test_excitation(fname):
        m = re.search(pattern, fname)
        if m:
            return int(m.group(1)), m.group(2)
        return None, None

    df[['test_id','excitation']] = df['file'].apply(lambda x: pd.Series(extract_test_excitation(x)))
    if df[['test_id','excitation']].isnull().any().any():
        raise ValueError("Some test_id or excitation not extracted properly.")

    # Unique group identifier
    df['group'] = df.apply(lambda r: f"{r['test_id']}_{r['excitation']}", axis=1)

    # Merge labels
    df = df.merge(labels_df, left_on='test_id', right_index=True, how='left')
    label_cols = list(labels_df.columns)

    # ===== Expand 'mode_shape' into multiple numeric columns =====
    if len(df) > 0:
        shape_len = len(df['mode_shape'].iloc[0])
        ms_cols = [f"ms_{i}" for i in range(shape_len)]
        # Convert each row's mode_shape into a row of numeric columns
        df[ms_cols] = pd.DataFrame(df['mode_shape'].tolist(), index=df.index)
    else:
        ms_cols = []

    # One-hot encode cluster
    df = pd.get_dummies(df, columns=['cluster'], prefix='cl')

    # Remove columns we do NOT want in features
    exclude_cols = [
        'file', 'group', 'test_id', 'excitation', 'is_augmented',
        'mode_number', 'mode_shape'  # remove original array column
    ]

    # Final feature columns
    feature_cols = [
        c for c in df.columns
        if c not in label_cols
        and c not in exclude_cols
    ]

    # Build X, y
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    y = df[label_cols].values

    # GroupShuffleSplit
    groups = df['group']
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optional check
    if not ((y_train.sum(axis=0) > 0).all() and (y_test.sum(axis=0) > 0).all()):
        print("Warning: Not all classes are represented in each split.")
    return X_train, X_test, y_train, y_test


# -------------------- MLP MODEL -------------------- #
def SHM_custom_mlp(input_dim, output_dim, config):
    """
    Simple Multi-Layer Perceptron for multi-label classification.
    """
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    model.add(layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(config['l2_reg'])
    ))
    model.add(layers.Dropout(0.3))

    # Output: one neuron per zone, with sigmoid activation for multi-label
    model.add(layers.Dense(output_dim, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# --------------- TRAIN & EVAL --------------- #
def train_and_evaluate_models(X_train, y_train, X_test, y_test, zone_labels, config):
    """
    Train (1) RandomForest, (2) MLP, then evaluate on test set and return results.
    """
    # 1) RandomForest
    rf_model = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42
    ))
    print("\nTraining RandomForest...")
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    # 2) MLP
    print("\nTraining MLP...")
    mlp_model = SHM_custom_mlp(X_train.shape[1], y_train.shape[1], config)
    # We won't pass class weights directly for multi-label. 
    es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history_callback = callbacks.History()

    mlp_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=1,
        callbacks=[es, history_callback]
    )
    y_pred_mlp_prob = mlp_model.predict(X_test)
    y_pred_mlp = (y_pred_mlp_prob > 0.5).astype(int)
    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

    # Package results
    results = {
        'RandomForest': {
            'model': rf_model,
            'predictions': y_pred_rf,
            'accuracy': rf_accuracy
        },
        'MLP': {
            'model': mlp_model,
            'predictions': y_pred_mlp,
            'probabilities': y_pred_mlp_prob,
            'accuracy': mlp_accuracy,
            'history': history_callback.history
        }
    }
    return results


def evaluation(results, X_test, y_test, zone_labels):
    """
    Print final metrics and create summary plots.
    """
    metrics_dict = {}
    model_names = list(results.keys())

    for model_name in model_names:
        y_pred = results[model_name]['predictions']
        # Binary damage detection
        y_test_damage = (y_test.sum(axis=1) > 0).astype(int)
        y_pred_damage = (y_pred.sum(axis=1) > 0).astype(int)

        dd_acc = accuracy_score(y_test_damage, y_pred_damage)
        dd_prec, dd_rec, dd_f1, _ = precision_recall_fscore_support(
            y_test_damage, y_pred_damage, 
            average='binary', zero_division=0
        )

        # Zone-level metrics
        zone_acc = accuracy_score(y_test, y_pred)
        c_report = classification_report(
            y_test, y_pred, 
            target_names=zone_labels,
            zero_division=0
        )

        # If MLP, compute ROC-AUC (average across zones that have both 0/1)
        if model_name == 'MLP':
            y_pred_prob = results['MLP']['probabilities']
            roc_list = []
            for i in range(y_test.shape[1]):
                if len(np.unique(y_test[:, i])) == 2:
                    roc_list.append(roc_auc_score(y_test[:, i], y_pred_prob[:, i]))
            roc_val = np.mean(roc_list) if len(roc_list) > 0 else 'Undefined'
        else:
            roc_val = 'Not Applicable'

        metrics_dict[model_name] = {
            'damage_detection': {
                'accuracy': dd_acc,
                'precision': dd_prec,
                'recall': dd_rec,
                'f1_score': dd_f1
            },
            'zone_localization': {
                'accuracy': zone_acc,
                'classification_report': c_report
            },
            'roc_auc': roc_val
        }

    # ---- PLOT ---- #
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Evaluation Summary")

    # Bar: model accuracy (zone-level)
    zone_accuracies = [metrics_dict[m]['zone_localization']['accuracy'] for m in model_names]
    ax[0].bar(model_names, zone_accuracies, color=['skyblue','limegreen'])
    ax[0].set_title("Zone-Localization Accuracy")
    ax[0].set_ylabel("Accuracy")

    # Bar: damage detection accuracy
    dd_accuracies = [metrics_dict[m]['damage_detection']['accuracy'] for m in model_names]
    ax[1].bar(model_names, dd_accuracies, color=['skyblue','limegreen'])
    ax[1].set_title("Damage Detection Accuracy")
    ax[1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

    # Print classification reports
    for m in model_names:
        print(f"\n=== {m} ===")
        dm = metrics_dict[m]['damage_detection']
        print("Damage Detection Metrics:")
        print(f"  Accuracy:  {dm['accuracy']:.4f}")
        print(f"  Precision: {dm['precision']:.4f}")
        print(f"  Recall:    {dm['recall']:.4f}")
        print(f"  F1 Score:  {dm['f1_score']:.4f}")

        print("\nZone Localization Metrics:")
        print(metrics_dict[m]['zone_localization']['classification_report'])

    return metrics_dict


# ------------------ MAIN SCRIPT ------------------ #
if __name__ == "__main__":
    # Example: test both label files
    # label_files = ["Damage_Labels_10.csv"]
    label_files = ["Damage_Labels_10.csv", "Damage_Labels_20.csv"]

    # MLP config
    config = {
        'l2_reg': 0.0005,
        'learning_rate': 0.015,
        'epochs': 100,
        'batch_size': 16,
    }

    # Dictionary to store final results for each label file
    results_summary = {}

    for labels_path in label_files:
        print(f"\nRunning with label set: {labels_path}\n")

        # 1) Load Data
        reference_modes, mode_shapes_data, zone_labels_df = load_data_and_labels(labels_path=labels_path)
        zone_labels = list(zone_labels_df.columns)

        # 2) Assign clusters (MAC-based) to each mode shape
        new_mode_data = []
        for fname in pd.DataFrame(mode_shapes_data)['file'].unique():
            subset = [d for d in mode_shapes_data if d['file'] == fname]
            assigned = filter_and_assign_clusters(subset, reference_modes)
            new_mode_data.extend(assigned)

        # 3) Train/test split (group-wise), expanding mode_shape to multiple columns
        X_train, X_test, y_train, y_test = split_data(
            selected_modes=new_mode_data,
            labels_df=zone_labels_df,
            test_size=0.2
        )

        # 4) Cross-validation for MLP
        df_temp = pd.DataFrame(new_mode_data)
        df_temp = df_temp.merge(zone_labels_df, left_on='test_id', right_index=True, how='left')
        df_temp['group'] = df_temp.apply(lambda r: f"{r['test_id']}_{r['excitation']}", axis=1)
        
        # 5) Final train/eval on single 80/20 split
        results = train_and_evaluate_models(X_train, y_train, X_test, y_test, zone_labels, config)

        # 6) Evaluate
        metrics_out = evaluation(results, X_test, y_test, zone_labels)

        # Store
        results_summary[labels_path] = metrics_out

    # Print summary
    for lblf, summ in results_summary.items():
        print(f"\n===== Summary for {lblf} =====")
        for mname, mdict in summ.items():
            dacc = mdict['damage_detection']['accuracy']
            zacc = mdict['zone_localization']['accuracy']
            print(f"  {mname} => DamageAcc={dacc:.3f}, ZoneAcc={zacc:.3f}")
