import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import pennylane as qml
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


# Support running both as a module and as a script
try:
    from .train_quantum import QuantumLSTMModel
    from .quantum_devices import QuantumDeviceManager
except Exception:
    try:
        _SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _SRC_ROOT not in sys.path:
            sys.path.append(_SRC_ROOT)
        from quantum_model.train_quantum import QuantumLSTMModel
        from quantum_model.quantum_devices import QuantumDeviceManager
    except Exception:
        QuantumLSTMModel = None
        QuantumDeviceManager = None


warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Data_files")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def _coerce_to_2d_array(x, name):
    """Convert various saved structures (lists, dicts, object arrays) to a 2D numpy array.
    name is a hint like 'predictions' or 'labels'.
    """
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return x
    
    # Handle scalar or 0-d arrays
    if isinstance(x, (int, float, np.number)) or (isinstance(x, np.ndarray) and x.ndim == 0):
        return np.array([[x]])
    
    # If object array or list/tuple
    if isinstance(x, np.ndarray) and x.dtype == object:
        try:
            x = x.tolist()
        except:
            return np.array([[0]])  # fallback for problematic arrays
    
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.array([[]])
            
        if len(x) > 0 and isinstance(x[0], dict):
            # Try common keys
            for key in ("predictions", "y_pred", "labels", "y_true", "ground_truth"):
                try:
                    vals = [np.array(item[key]) for item in x if key in item]
                    if len(vals) == len(x):
                        return np.vstack(vals)
                except Exception:
                    pass
            # Otherwise stack first values
            try:
                vals = [np.array(list(item.values())[0]) for item in x]
                return np.vstack(vals)
            except Exception:
                pass
        # Regular list of arrays
        try:
            arrs = [np.array(item) for item in x]
            return np.vstack(arrs)
        except Exception:
            pass
    
    if isinstance(x, dict):
        # Special handling for hardware predictions structure
        if name == "predictions" and "hardware_calibrated" in x:
            try:
                return _coerce_to_2d_array(x["hardware_calibrated"], name)
            except:
                pass
        if name == "predictions" and "hardware_raw" in x:
            try:
                return _coerce_to_2d_array(x["hardware_raw"], name)
            except:
                pass
        
        # If keys are numeric-like, sort them
        try:
            keys = list(x.keys())
            if all(isinstance(k, (int, np.integer)) or (isinstance(k, str) and str(k).isdigit()) for k in keys):
                keys_sorted = sorted(keys, key=lambda k: int(k))
                vals = [np.array(x[k]) for k in keys_sorted]
                return np.vstack(vals)
        except Exception:
            pass
        
        # Single nested array under known key
        for key in ("predictions", "y_pred", name):
            if key in x:
                try:
                    xx = np.array(x[key])
                    return _coerce_to_2d_array(xx, name)
                except Exception:
                    pass
        
        # Fallback: stack values
        try:
            return np.vstack([np.array(v) for v in x.values()])
        except Exception:
            pass
    
    # Last resort: coerce to at least 2D
    try:
        x = np.array(x)
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim > 2:
            try:
                return x.reshape(x.shape[0], -1)
            except Exception:
                return x.reshape(1, -1)
        return x
    except Exception:
        # Ultimate fallback - return empty array
        return np.array([[0]])

def create_results_directories():
    """Create the results directory structure"""
    base_results_dir = RESULTS_DIR 
    
    # Create main results directory
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Create quantum subdirectories
    quantum_dir = os.path.join(base_results_dir, "quantum")
    os.makedirs(quantum_dir, exist_ok=True)
    
    # Create device-specific directories
    device_dirs = {
        "ideal": os.path.join(quantum_dir, "ideal"),
        "noisy": os.path.join(quantum_dir, "noisy"),
        "hardware": os.path.join(base_results_dir, "physical")
    }
    
    for device_type, dir_path in device_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
    
    return device_dirs

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration
N_QUBITS = 4
N_LAYERS = 3
BATCH_SIZE = 512
SHOTS = 1024
# Use absolute, project-rooted paths
MODEL_PATH = os.path.join(MODELS_DIR, "hyprid.pth")

# Dataset files
X_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_PATH = os.path.join(DATA_DIR, "y_test.npy")
LOC_PATH = os.path.join(DATA_DIR, "loc_test.npy")

class QLSTMModel(nn.Module):
    """
    Hybrid Quantum-Classical model for multi-step forecasting.
    A classical LSTM processes the sequence, and its output is fed
    into a quantum circuit for feature extraction, followed by a
    classical layer for multi-step prediction.
    """
    def __init__(self, n_features, n_lstm_units=4, n_qubits=2, num_layers=1, n_layers=1, output_len=72, device_type="ideal"):
        super(QLSTMModel, self).__init__()

        # 1. Classical LSTM Layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_lstm_units,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 2. Classical Layer to map LSTM output to Quantum input
        self.classical_to_quantum = nn.Linear(n_lstm_units, n_qubits)
        
        # 3. Quantum Layer setup
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_type = device_type
        
        if device_type == "ideal":
            self.dev = qml.device("lightning.qubit", wires=n_qubits)

        elif device_type == "noisy":
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_aer.noise import NoiseModel
            service = QiskitRuntimeService()
            backend = service.backend('ibm_brisbane')
            noise_model = NoiseModel.from_backend(backend)
            self.dev = qml.device("qiskit.aer", wires=n_qubits, 
                                    backend="aer_simulator", 
                                    noise_model=noise_model, shots=SHOTS)

        elif device_type == "hardware":
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.backend("ibm_brisbane")
            self.dev = qml.device("qiskit.remote", wires=n_qubits, backend=backend)
        
        @qml.qnode(self.dev, interface="torch")
        def q_circuit(inputs, weights):
            """A quantum circuit that acts as a feature extractor."""
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # 4. Create torch layer with proper weight initialization
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        
        # 5. Classical Layer to map quantum output to predictions
        self.quantum_to_output = nn.Linear(n_qubits, output_len)
        
    def forward(self, x):        
        # 1. Pass data through the classical LSTM
        lstm_out, _ = self.lstm(x)
        
        # 2. Extract features from the last timestep
        final_lstm_output = lstm_out[:, -1, :]
        
        # 3. Prepare the data for the quantum circuit
        quantum_input = self.classical_to_quantum(final_lstm_output)
        
        # 4. Pass the features through the quantum circuit
        quantum_features = self.q_layer(quantum_input)
        
        # 5. Map quantum features to output sequence
        output = self.quantum_to_output(quantum_features)
        
        # 6. Apply sigmoid activation to get probabilities
        return torch.sigmoid(output)

def load_data():
    """Load the test dataset or generate synthetic data if files don't exist"""

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.float32)
    loc = np.load(LOC_PATH)
    print("Loaded real test data from files")

    return X, y, loc


def plot_example_predictions(model, X_test, y_test, num_examples=3, device_type="ideal", output_dir=None):
    """Plot example predictions vs actual values"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        # Adjust num_examples if we have fewer samples than requested
        available_samples = len(X_test)
        actual_num_examples = min(num_examples, available_samples)
        if actual_num_examples == 0:
            print(f"No samples available for {device_type} example plots.")
            return
            
        sample_indices = np.random.choice(available_samples, actual_num_examples, replace=False)
        plt.figure(figsize=(15, 5 * actual_num_examples))

        for i, idx in enumerate(sample_indices):
            X_sample = torch.tensor(X_test[idx:idx+1]).to(device)
            y_true = y_test[idx]
            try:
                y_pred = model(X_sample).cpu().numpy()[0]
                y_pred = (y_pred > 0.5).astype(int)
            except Exception as e:
                print(f"Error in prediction for sample {idx}: {e}")
                y_pred = np.zeros_like(y_true)

            plt.subplot(actual_num_examples, 1, i+1)
            hours = np.arange(1, 72 + 1)
            plt.plot(hours, y_true, 'bo-', label='Actual', linewidth=2, markersize=4)
            plt.plot(hours, y_pred, 'ro--', label='Predicted', linewidth=2, markersize=4)
            plt.title(f'Example {i+1}: Air Quality Prediction - {device_type.title()} Model (Next 72 Hours)')
            plt.xlabel('Hours Ahead')
            plt.ylabel('Air Quality Class')
            plt.yticks([0, 1], ['Good', 'Poor'])
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        

        save_path = os.path.join(output_dir, f'example_predictions_{device_type}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
        print(f"  → Saved example predictions to {save_path}")

def plot_example_predictions_from_arrays(preds_array, labels_array, num_examples=3, device_type="hardware", output_dir=None):
    """Plot example predictions vs actual values from saved arrays"""
    preds_array = _coerce_to_2d_array(preds_array, "predictions")
    labels_array = _coerce_to_2d_array(labels_array, "labels")
    # Threshold probabilistic predictions for plotting
    if not np.issubdtype(preds_array.dtype, np.integer):
        try:
            preds_array = (preds_array > 0.5).astype(int)
        except Exception:
            preds_array = np.zeros_like(labels_array, dtype=int)
    if labels_array.size == 0 or preds_array.size == 0:
        print("No data available for hardware example plots.")
        return
    num_samples = labels_array.shape[0]
    if num_samples == 0:
        return
    sample_indices = np.random.choice(num_samples, min(num_examples, num_samples), replace=False)
    plt.figure(figsize=(15, 5 * len(sample_indices)))
    for i, idx in enumerate(sample_indices):
        y_true = np.array(labels_array[idx]).reshape(-1)
        y_pred = np.array(preds_array[idx]).reshape(-1)
        plt.subplot(len(sample_indices), 1, i+1)
        hours = np.arange(1, len(y_true) + 1)
        plt.plot(hours, y_true, 'bo-', label='Actual', linewidth=2, markersize=4)
        plt.plot(hours, y_pred, 'ro--', label='Predicted', linewidth=2, markersize=4)
        plt.title(f'Example {i+1}: Air Quality Prediction - {device_type.title()} Model (Next {len(y_true)} Hours)')
        plt.xlabel('Hours Ahead')
        plt.ylabel('Air Quality Class')
        plt.yticks([0, 1], ['Good', 'Poor'])
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'example_predictions_{device_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved example predictions to {save_path}")

def evaluate_model_per_location_and_hour(model, test_loader, device, location_indices_test, output_seq_len=72):
    """
    Evaluate the model with comprehensive metrics
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            try:
                outputs = model(X_batch)
                preds = (outputs > 0.5).float()
            except Exception as e:
                print(f"Error in batch processing: {e}")
                preds = torch.zeros_like(y_batch)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    preds_flat, labels_flat = all_preds.flatten(), all_labels.flatten()

    # All metrics Required
    accuracy = (preds_flat == labels_flat).mean()
    precision = precision_score(labels_flat, preds_flat, average="binary", zero_division=0)
    recall = recall_score(labels_flat, preds_flat, average="binary", zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, average="binary", zero_division=0)
    conf_matrix = confusion_matrix(labels_flat, preds_flat)

    # Per-location metrics
    unique_locations = np.unique(location_indices_test)
    location_metrics = {}
    for loc in unique_locations:
        idx = (location_indices_test == loc)
        if np.sum(idx) > 0:  # Only process if location has samples
            preds_loc, labels_loc = all_preds[idx].flatten(), all_labels[idx].flatten()
            if len(preds_loc) > 0:
                location_metrics[int(loc)] = {
                    "accuracy": (preds_loc == labels_loc).mean(),
                    "precision": precision_score(labels_loc, preds_loc, average="binary", zero_division=0),
                    "recall": recall_score(labels_loc, preds_loc, average="binary", zero_division=0),
                    "f1": f1_score(labels_loc, preds_loc, average="binary", zero_division=0),
                    "confusion_matrix": confusion_matrix(labels_loc, preds_loc, labels=[0, 1])
                }

    # Per-hour metrics
    hour_metrics = []
    for hour in range(output_seq_len):
        preds_hour, labels_hour = all_preds[:, hour], all_labels[:, hour]
        hour_metrics.append({
            "hour": hour + 1,
            "accuracy": (preds_hour == labels_hour).mean(),
            "precision": precision_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "recall": recall_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "f1": f1_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(labels_hour, preds_hour, labels=[0, 1])
        })

    return {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        },
        "per_location": location_metrics,
        "per_hour": hour_metrics,
        "preds_flat": preds_flat,  
        "labels_flat": labels_flat  
    }

def compute_metrics_from_arrays(preds_array, labels_array, location_indices=None):
    """Compute overall, per-location, and per-hour metrics from saved arrays"""
    preds = np.array(preds_array)
    labels = np.array(labels_array)

    # Normalize potential object/dict/list structures to 2D numeric arrays
    def _to_2d_array(x, name):
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return x
        # If object array or list/tuple
        if isinstance(x, np.ndarray) and x.dtype == object:
            x = list(x)
        if isinstance(x, (list, tuple)):
            # If list of dicts with a common key
            if len(x) > 0 and isinstance(x[0], dict):
                # Prefer keys that look like predictions or labels
                for key in ("predictions", "y_pred", "labels", "y_true", "ground_truth"):
                    try:
                        vals = [np.array(item[key]) for item in x if key in item]
                        if len(vals) == len(x):
                            return np.vstack(vals)
                    except Exception:
                        pass
                # Otherwise try values()
                try:
                    vals = [np.array(list(item.values())[0]) for item in x]
                    return np.vstack(vals)
                except Exception:
                    pass
            # Regular list of arrays
            try:
                arrs = [np.array(item) for item in x]
                return np.vstack(arrs)
            except Exception:
                pass
        if isinstance(x, dict):
            # If keys are numeric indices, sort them
            try:
                keys = list(x.keys())
                if all(isinstance(k, (int, np.integer)) or (isinstance(k, str) and str(k).isdigit()) for k in keys):
                    keys_sorted = sorted(keys, key=lambda k: int(k))
                    vals = [np.array(x[k]) for k in keys_sorted]
                    return np.vstack(vals)
            except Exception:
                pass
            # If map has a single nested array under a known key
            for key in ("predictions", "y_pred", name):
                if key in x:
                    try:
                        xx = np.array(x[key])
                        return _to_2d_array(xx, name)
                    except Exception:
                        pass
            # Fallback: stack values in insertion order
            try:
                return np.vstack([np.array(v) for v in x.values()])
            except Exception:
                pass
        # Last resort: try to coerce to 2D
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    preds = _coerce_to_2d_array(preds, "predictions")
    labels = _coerce_to_2d_array(labels, "labels")

    # Ensure shapes match
    if preds.size == 0 or labels.size == 0:
        raise RuntimeError("Empty predictions or labels array for hardware metrics.")
    if preds.shape != labels.shape:
        min_n = min(preds.shape[0], labels.shape[0])
        min_t = min(preds.shape[1], labels.shape[1])
        preds = preds[:min_n, :min_t]
        labels = labels[:min_n, :min_t]

    # Threshold predictions if they are probabilistic  
    if isinstance(preds, np.ndarray) and not np.issubdtype(preds.dtype, np.integer):
        preds = (preds > 0.5).astype(int)
    elif not isinstance(preds, np.ndarray):
        print(f"Warning: predictions is not a numpy array, it's {type(preds)}")
        preds = np.array([[0]])  # fallback
    preds_flat, labels_flat = preds.flatten(), labels.flatten()
    accuracy = (preds_flat == labels_flat).mean()
    precision = precision_score(labels_flat, preds_flat, average="binary", zero_division=0)
    recall = recall_score(labels_flat, preds_flat, average="binary", zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, average="binary", zero_division=0)
    conf_matrix = confusion_matrix(labels_flat, preds_flat)
    # Per-location metrics
    location_metrics = {}
    loc_idx_arr = None
    if location_indices is not None:
        try:
            loc_idx_arr = np.array(location_indices)
            if loc_idx_arr.ndim == 0:
                loc_idx_arr = None
            elif loc_idx_arr.ndim > 1:
                loc_idx_arr = loc_idx_arr.reshape(-1)
            if loc_idx_arr is not None and loc_idx_arr.shape[0] != preds.shape[0]:
                loc_idx_arr = None
        except Exception:
            loc_idx_arr = None
    if loc_idx_arr is not None:
        unique_locations = np.unique(loc_idx_arr)
        for loc in unique_locations.tolist():
            idx = (loc_idx_arr == loc)
            if np.sum(idx) > 0:
                preds_loc, labels_loc = preds[idx].flatten(), labels[idx].flatten()
                if preds_loc.size > 0:
                    location_metrics[int(loc)] = {
                        "accuracy": (preds_loc == labels_loc).mean(),
                        "precision": precision_score(labels_loc, preds_loc, average="binary", zero_division=0),
                        "recall": recall_score(labels_loc, preds_loc, average="binary", zero_division=0),
                        "f1": f1_score(labels_loc, preds_loc, average="binary", zero_division=0),
                        "confusion_matrix": confusion_matrix(labels_loc, preds_loc, labels=[0, 1])
                    }
    # Per-hour metrics
    output_seq_len = preds.shape[1]
    hour_metrics = []
    for hour in range(output_seq_len):
        preds_hour, labels_hour = preds[:, hour], labels[:, hour]
        hour_metrics.append({
            "hour": hour + 1,
            "accuracy": (preds_hour == labels_hour).mean(),
            "precision": precision_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "recall": recall_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "f1": f1_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(labels_hour, preds_hour, labels=[0, 1])
        })
    return {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        },
        "per_location": location_metrics,
        "per_hour": hour_metrics,
        "preds_flat": preds_flat,
        "labels_flat": labels_flat
    }

def _serialize_metrics_for_json(metrics):
    """Convert numpy arrays in metrics dict to JSON-serializable types"""
    def convert_cm(cm):
        try:
            return cm.tolist()
        except Exception:
            return cm
    def to_json_scalar(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    out = {
        "overall": {k: (convert_cm(v) if k == "confusion_matrix" else to_json_scalar(v))
                     for k, v in metrics.get("overall", {}).items()},
        "per_location": {},
        "per_hour": []
    }
    for loc, vals in metrics.get("per_location", {}).items():
        out["per_location"][str(loc)] = {k: (convert_cm(v) if k == "confusion_matrix" else to_json_scalar(v))
                                           for k, v in vals.items()}
    for h in metrics.get("per_hour", []):
        out["per_hour"].append({k: (convert_cm(v) if k == "confusion_matrix" else int(v) if k == "hour" else to_json_scalar(v))
                                 for k, v in h.items()})
    return out

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", device_type="ideal", output_dir=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} - {device_type.title()} Model")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'confusion_matrix_{device_type}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved confusion matrix to {save_path}")

def plot_per_location_accuracy(location_metrics, device_type="ideal", output_dir=None):
    """Plot accuracy per location"""
    locations = sorted(location_metrics.keys())
    accuracies = [location_metrics[loc]['accuracy'] for loc in locations]
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(locations)), accuracies, alpha=0.7, color='steelblue')
    plt.xlabel('Location Index')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Location - {device_type.title()} Model')
    plt.xticks(range(0, len(locations), max(1, len(locations)//20)), 
               [locations[i] for i in range(0, len(locations), max(1, len(locations)//20))], 
               rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    

    save_path = os.path.join(output_dir, f'per_location_accuracy_{device_type}.png')

    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved per-location accuracy to {save_path}")

def plot_per_hour_metrics(hour_metrics, device_type="ideal", output_dir=None):
    """Plot metrics per prediction hour"""
    hours = [h['hour'] for h in hour_metrics]
    accuracies = [h['accuracy'] for h in hour_metrics]
    precisions = [h['precision'] for h in hour_metrics]
    recalls = [h['recall'] for h in hour_metrics]
    f1_scores = [h['f1'] for h in hour_metrics]
    
    plt.figure(figsize=(15, 10))
    
    # Accuracy over hours
    plt.subplot(2, 2, 1)
    plt.plot(hours, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Prediction Hour')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Prediction Hour - {device_type.title()}')
    plt.grid(True, alpha=0.3)
    
    # Precision over hours
    plt.subplot(2, 2, 2)
    plt.plot(hours, precisions, 'g-', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Prediction Hour')
    plt.ylabel('Precision')
    plt.title(f'Precision vs Prediction Hour - {device_type.title()}')
    plt.grid(True, alpha=0.3)
    
    # Recall over hours
    plt.subplot(2, 2, 3)
    plt.plot(hours, recalls, 'r-', linewidth=2, marker='^', markersize=3)
    plt.xlabel('Prediction Hour')
    plt.ylabel('Recall')
    plt.title(f'Recall vs Prediction Hour - {device_type.title()}')
    plt.grid(True, alpha=0.3)
    
    # F1 Score over hours
    plt.subplot(2, 2, 4)
    plt.plot(hours, f1_scores, 'm-', linewidth=2, marker='d', markersize=3)
    plt.xlabel('Prediction Hour')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs Prediction Hour - {device_type.title()}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    

    save_path = os.path.join(output_dir, f'per_hour_metrics_{device_type}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved per-hour metrics to {save_path}")

def analyze_json_data(json_path="job_metadata.json"):
    """Analyze job metadata from JSON file"""
    print(f"\n=== ANALYZING {json_path} ===")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        job_history = data.get('job_history', [])
        print(f"Total job batches: {len(job_history)}")
        
        # Analyze execution statistics
        successful_batches = []
        failed_batches = []
        execution_times = []
        accuracy_data = []
        
        for i, batch in enumerate(job_history):
            batch_info = batch.get('batch_info', {})
            execution = batch.get('execution', {})
            results = batch.get('results_summary', {})
            
            if execution.get('success', False):
                successful_batches.append(i)
                if 'execution_time' in execution:
                    execution_times.append(execution['execution_time'])
                if 'best_ensemble_accuracy' in results:
                    accuracy_data.append(results['best_ensemble_accuracy'])
            else:
                failed_batches.append(i)
        
        print(f"Successful batches: {len(successful_batches)}")
        print(f"Failed batches: {len(failed_batches)}")
        
        if execution_times:
            print(f"Average execution time: {np.mean(execution_times):.2f} seconds")
            print(f"Total execution time: {np.sum(execution_times):.2f} seconds")
            
        if accuracy_data:
            print(f"Average accuracy: {np.mean(accuracy_data):.4f}")
            print(f"Best accuracy: {np.max(accuracy_data):.4f}")
            print(f"Worst accuracy: {np.min(accuracy_data):.4f}")
            
            # Plot accuracy over time
            plt.figure(figsize=(12, 6))
            plt.plot(accuracy_data, 'bo-', linewidth=2, markersize=6)
            plt.xlabel('Batch Number')
            plt.ylabel('Best Ensemble Accuracy')
            plt.title('Hardware Testing Accuracy Over Time')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('hardware_accuracy_over_time.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        # Plot execution times
        if execution_times:
            plt.figure(figsize=(12, 6))
            plt.plot(execution_times, 'ro-', linewidth=2, markersize=6)
            plt.xlabel('Batch Number')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Hardware Testing Execution Times')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('hardware_execution_times.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"Error analyzing JSON data: {e}")

def analyze_pkl_data(pkl_path="hardware_testing_results.pkl"):
    """Analyze pickled test results"""
    print(f"\n=== ANALYZING {pkl_path} ===")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data keys: {list(data.keys())}")
        
        # Analyze predictions and ground truth (coerce structures safely)
        predictions = data.get('predictions')
        probabilities = data.get('probabilities')
        ground_truth = data.get('ground_truth')

        if ground_truth is not None and (predictions is not None or probabilities is not None):
            gt_arr = _coerce_to_2d_array(ground_truth, 'labels')
            if predictions is None and probabilities is not None:
                pred_arr = _coerce_to_2d_array(probabilities, 'predictions')
            else:
                pred_arr = _coerce_to_2d_array(predictions, 'predictions')

            # Align shapes
            if pred_arr.size == 0 or gt_arr.size == 0:
                print("Empty arrays in PKL; skipping confusion matrix.")
            else:
                if pred_arr.shape != gt_arr.shape:
                    min_n = min(pred_arr.shape[0], gt_arr.shape[0])
                    min_t = min(pred_arr.shape[1], gt_arr.shape[1])
                    pred_arr = pred_arr[:min_n, :min_t]
                    gt_arr = gt_arr[:min_n, :min_t]

                # Threshold to binary if needed
                if not np.issubdtype(pred_arr.dtype, np.integer):
                    pred_arr = (pred_arr > 0.5).astype(int)
                if not np.issubdtype(gt_arr.dtype, np.integer):
                    gt_arr = (gt_arr > 0.5).astype(int)

                print(f"Predictions shape: {pred_arr.shape}")
                print(f"Ground truth shape: {gt_arr.shape}")

                # Calculate accuracy
                accuracy = accuracy_score(gt_arr.flatten(), pred_arr.flatten())
                print(f"Overall accuracy: {accuracy:.4f}")

                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(gt_arr.flatten(), pred_arr.flatten())
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Good', 'Poor'], yticklabels=['Good', 'Poor'])
                plt.title('Hardware Testing Results - Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig('hardware_confusion_matrix_pkl.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Analyze accuracies if available
        if 'accuracies' in data:
            accuracies = data['accuracies']
            # Handle potential dict structures in accuracies
            if isinstance(accuracies, dict):
                # Extract numeric values from dict
                acc_values = []
                for key, val in accuracies.items():
                    if isinstance(val, (int, float)):
                        acc_values.append(val)
                    elif isinstance(val, (list, tuple)):
                        acc_values.extend([v for v in val if isinstance(v, (int, float))])
                accuracies = acc_values if acc_values else [0]
            
            if len(accuracies) > 0:
                print(f"Accuracy statistics:")
                print(f"  Mean: {np.mean(accuracies):.4f}")
                print(f"  Std: {np.std(accuracies):.4f}")
                print(f"  Min: {np.min(accuracies):.4f}")
                print(f"  Max: {np.max(accuracies):.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
            plt.title('Distribution of Hardware Testing Accuracies')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('hardware_accuracy_distribution_pkl.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        # Analyze calibration info if available
        if 'calibration_info' in data:
            calibration_info = data['calibration_info']
            print(f"Calibration information available: {list(calibration_info.keys()) if isinstance(calibration_info, dict) else type(calibration_info)}")
            
    except Exception as e:
        print(f"Error analyzing PKL data: {e}")

def build_model(device_type="ideal", backend_name=None):
    """Build quantum model with specified device configuration"""
    n_lstm_units = 32
    num_layers = 1
    
    if QuantumLSTMModel is not None:
        device_manager = QuantumDeviceManager(
            backend_name=backend_name,
            shots=SHOTS,
            use_noise_model=(device_type == "noisy"),
            use_real_hardware=(device_type == "hardware")
        )
        
        model = QuantumLSTMModel(
            n_features=9,
            n_lstm_units=n_lstm_units,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            num_lstm_layers=num_layers,
            output_len=72,
            device_type=device_type,
            device_manager=device_manager
        )

    else:
        # Use simplified model
        model = QLSTMModel(
            n_features=9, 
            n_lstm_units=n_lstm_units,  
            n_qubits=N_QUBITS,
            num_layers=num_layers,
            n_layers=N_LAYERS,
            output_len=72,
            device_type=device_type
        )
    
    return model

def comprehensive_testing(num_samples=100):
    """Run comprehensive quantum testing across all device types"""
    print("=" * 80)
    print("COMPREHENSIVE QUANTUM TESTING")
    print("=" * 80)
    
    # Create results directory structure
    print("\nCreating results directory structure...")
    device_dirs = create_results_directories()
    for device_type, dir_path in device_dirs.items():
        print(f"  → {device_type}: {dir_path}")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"✅ Found GPU: {torch.cuda.get_device_name(0)}. Using CUDA.")
        device = torch.device("cuda")
    else:
        print("❌ No GPU found. Using CPU.")
        device = torch.device("cpu")
    
    # Check for data files (will generate synthetic data if not found)
    # Don't require model file - will use random weights if not available
    
    # Load data
    print("\nLoading data...")
    X_test, y_test, loc_test = load_data()
    print(f"Data shapes: X={X_test.shape}, y={y_test.shape}, loc={loc_test.shape}")
    
    # Use only a subset for testing
    if num_samples < len(X_test):
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_test = X_test[sample_indices]
        y_test = y_test[sample_indices]  
        loc_test = loc_test[sample_indices]
        print(f"Using {num_samples} samples for testing")
    
    # Create data loaders
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {}
    
    # Test different device types
    device_types = ["ideal", "noisy", "hardware"]
    backend_name = "ibm_brisbane"  # You can change this
    
    for device_type in device_types:
        print(f"\n" + "="*50)
        print(f"TESTING {device_type.upper()} DEVICE")
        print(f"" + "="*50)
        
        try:
            output_dir = device_dirs[device_type]
            if device_type in ("ideal", "noisy"):
                # Build and evaluate simulated devices
                print(f"Building {device_type} model...")
                model = build_model(device_type=device_type, backend_name=backend_name if device_type != "ideal" else None).to(device)
                print("Loading model weights...")
                try:
                    saved_state_dict = torch.load(MODEL_PATH, weights_only=True)
                    
                    # Create mapping for layer name differences
                    current_state_dict = model.state_dict()
                    mapped_state_dict = {}
                    
                    for saved_key, saved_tensor in saved_state_dict.items():
                        # Map q_layer.weights to quantum_layer.weights
                        if saved_key == 'q_layer.weights' and 'quantum_layer.weights' in current_state_dict:
                            mapped_state_dict['quantum_layer.weights'] = saved_tensor
                            print(f"  Mapped {saved_key} -> quantum_layer.weights")
                        # Keep other keys as they are if they exist in current model
                        elif saved_key in current_state_dict:
                            mapped_state_dict[saved_key] = saved_tensor
                    
                    # Load the mapped weights
                    model.load_state_dict(mapped_state_dict, strict=False)
                    print("✅ Weights loaded successfully with layer name mapping")
                    
                    # Report which weights were loaded vs missing
                    loaded_keys = set(mapped_state_dict.keys())
                    current_keys = set(current_state_dict.keys())
                    missing_keys = current_keys - loaded_keys
                    if missing_keys:
                        print(f"  Missing keys (will use random initialization): {list(missing_keys)}")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not load weights: {e}")
                    print("Using randomly initialized weights")
                print("Calculating detailed accuracy metrics...")
                metrics = evaluate_model_per_location_and_hour(model, test_loader, device, loc_test)
                # Plots
                print(f"Generating plots for {device_type} model...")
                plot_example_predictions(model, X_test, y_test, num_examples=3, device_type=device_type, output_dir=output_dir)
                plot_confusion_matrix(metrics['labels_flat'], metrics['preds_flat'], device_type=device_type, output_dir=output_dir)
                if metrics['per_location']:
                    plot_per_location_accuracy(metrics['per_location'], device_type=device_type, output_dir=output_dir)
                plot_per_hour_metrics(metrics['per_hour'], device_type=device_type, output_dir=output_dir)
            else:
                # Hardware: skip execution, load from saved files
                print("Skipping real hardware execution. Loading saved hardware results...")
                pkl_path = os.path.join(PROJECT_ROOT, "hardware_testing_results.pkl")
                
                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, 'rb') as f:
                            hardware_data = pickle.load(f)
                        
                        # Extract data using notebook approach
                        predictions = hardware_data.get('predictions', {})
                        ground_truth = hardware_data.get('ground_truth', [])
                        batch_info = hardware_data.get('batch_info', {})
                        
                        # Convert to numpy arrays like in the notebook (use hardware_raw like notebook)
                        hardware_predictions = np.array(predictions.get('hardware_raw', []))
                        ground_truth_array = np.array(ground_truth)
                        sample_indices = batch_info.get('sample_indices', [])
                        
                        print(f"Loaded hardware data - Predictions: {hardware_predictions.shape}, Ground truth: {ground_truth_array.shape}")
                        
                        # Calculate per-sample accuracies like in notebook
                        sample_accuracies = []
                        sample_f1_scores = []
                        
                        for i in range(len(hardware_predictions)):
                            if i < len(ground_truth_array):
                                # Calculate accuracy for this sample (hardware_raw predictions are already binary)
                                sample_acc = accuracy_score(ground_truth_array[i], hardware_predictions[i])
                                sample_accuracies.append(sample_acc)
                                
                                # Calculate F1 score
                                try:
                                    sample_f1 = f1_score(ground_truth_array[i], hardware_predictions[i], average='macro', zero_division=0)
                                    sample_f1_scores.append(sample_f1)
                                except:
                                    sample_f1_scores.append(0)
                        
                        # Overall metrics like in notebook
                        overall_accuracy = np.mean(sample_accuracies) if sample_accuracies else 0
                        overall_f1 = np.mean(sample_f1_scores) if sample_f1_scores else 0
                        
                        # Create metrics structure similar to other device types
                        flat_preds = hardware_predictions.flatten()  # hardware_raw is already binary
                        flat_labels = ground_truth_array.flatten()
                        
                        metrics = {
                            'overall': {
                                'accuracy': overall_accuracy,
                                'precision': precision_score(flat_labels, flat_preds, average='binary', zero_division=0),
                                'recall': recall_score(flat_labels, flat_preds, average='binary', zero_division=0),
                                'f1': overall_f1,
                                'confusion_matrix': confusion_matrix(flat_labels, flat_preds)
                            },
                            'per_location': {},  # Hardware data doesn't have location info
                            'per_hour': [],
                            'preds_flat': flat_preds,
                            'labels_flat': flat_labels
                        }
                        
                        # Calculate per-hour metrics
                        for hour in range(hardware_predictions.shape[1]):
                            preds_hour = hardware_predictions[:, hour]  # hardware_raw is already binary
                            labels_hour = ground_truth_array[:, hour]
                            metrics['per_hour'].append({
                                'hour': hour + 1,
                                'accuracy': accuracy_score(labels_hour, preds_hour),
                                'precision': precision_score(labels_hour, preds_hour, average='binary', zero_division=0),
                                'recall': recall_score(labels_hour, preds_hour, average='binary', zero_division=0),
                                'f1': f1_score(labels_hour, preds_hour, average='binary', zero_division=0),
                                'confusion_matrix': confusion_matrix(labels_hour, preds_hour, labels=[0, 1])
                            })
                        
                        print(f"Hardware metrics calculated - Overall accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}")
                        
                        # Generate plots using the processed data
                        plot_example_predictions_from_arrays(hardware_predictions, ground_truth_array, num_examples=3, device_type=device_type, output_dir=output_dir)
                        plot_confusion_matrix(metrics['labels_flat'], metrics['preds_flat'], device_type=device_type, output_dir=output_dir)
                        plot_per_hour_metrics(metrics['per_hour'], device_type=device_type, output_dir=output_dir)
                        
                    except Exception as e:
                        print(f"❌ Error processing hardware PKL data: {e}")
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to process hardware data: {e}")
                else:
                    raise RuntimeError("Hardware testing results PKL file not found")

            # Store summarized results for all device types
            results[device_type] = {
                'overall_metrics': metrics['overall'],
                'per_location_count': len(metrics['per_location']),
                'per_hour_count': len(metrics['per_hour'])
            }

            # Save per-device metrics JSON
            per_device_metrics_path = os.path.join(output_dir, f"metrics_{device_type}.json")
            with open(per_device_metrics_path, 'w') as f:
                json.dump(_serialize_metrics_for_json(metrics), f, indent=2)
            print(f"Saved per-device metrics to {per_device_metrics_path}")

            # Print overall results
            overall = metrics['overall']
            print(f"\n{device_type.upper()} RESULTS:")
            print(f"  Overall Accuracy:  {overall['accuracy']:.4f}")
            print(f"  Overall Precision: {overall['precision']:.4f}")
            print(f"  Overall Recall:    {overall['recall']:.4f}")
            print(f"  Overall F1 Score:  {overall['f1']:.4f}")

        except Exception as e:
            print(f"❌ Error testing {device_type} device: {e}")
            results[device_type] = {'error': str(e)}
            import traceback
            traceback.print_exc()
    
    # Analyze existing JSON and PKL files
    print(f"\n" + "="*50)
    print("ANALYZING EXISTING DATA FILES")
    print(f"" + "="*50)
    
    job_metadata_path = os.path.join(PROJECT_ROOT, "job_metadata.json")
    if os.path.exists(job_metadata_path):
        analyze_json_data(job_metadata_path)
    else:
        print("job_metadata.json not found")
        
    hardware_pkl_path = os.path.join(PROJECT_ROOT, "hardware_testing_results.pkl")
    if os.path.exists(hardware_pkl_path):
        analyze_pkl_data(hardware_pkl_path)
    else:
        print("hardware_testing_results.pkl not found")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"comprehensive_testing_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*50)
    print("TESTING COMPLETE")
    print(f"" + "="*50)
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\nSUMMARY:")
    for device_type, result in results.items():
        if 'error' in result:
            print(f"  {device_type.upper():10}: ❌ Error - {result['error']}")
        else:
            acc = result['overall_metrics']['accuracy']
            print(f"  {device_type.upper():10}: ✅ Accuracy = {acc:.4f}")
    
    print("\nAll plots and analysis have been generated and saved as PNG files.")
    return results

if __name__ == "__main__":
    try:
        # Run comprehensive testing
        results = comprehensive_testing(num_samples=50)  # Use smaller sample for testing
        print("\nTesting completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()