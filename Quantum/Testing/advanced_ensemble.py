#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import pennylane as qml

# Optional: silence warnings
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Config
N_QUBITS = 4
N_LAYERS = 3
BATCH_SIZE = 256
SHOTS = 1024
SAMPLES = 1000
MODEL_PATH = "best_qlstm_model_multistep.pth"

# Dataset files expected in CWD
X_PATH = "X_test.npy"
Y_PATH = "y_test.npy"
LOC_PATH = "loc_test.npy"


def load_data(sample_n=SAMPLES):
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    loc = np.load(LOC_PATH)

    if sample_n is not None and sample_n < len(X):
        idx = np.random.choice(len(X), sample_n, replace=False)
        X = X[idx]
        y = y[idx]
        loc = loc[idx]

    return X.astype(np.float32), y.astype(np.float32), loc


def make_noise_model():
    """Construct synthetic noise model using Qiskit Aer"""
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    except Exception as e:
        raise RuntimeError(
            "qiskit-aer is required for the noisy simulator. Install with `pip install qiskit-aer`."
        ) from e

    noise_model = NoiseModel()

    # Gate errors
    p1 = 0.01  # 1-qubit depolarizing prob
    p2 = 0.02  # 2-qubit depolarizing prob

    one_qubit_err = depolarizing_error(p1, 1)
    two_qubit_err = depolarizing_error(p2, 2)

    # Apply to common gates
    for g in ["u1", "u2", "u3", "rx", "ry", "rz", "x", "h"]:
        noise_model.add_all_qubit_quantum_error(one_qubit_err, g)
    noise_model.add_all_qubit_quantum_error(two_qubit_err, ["cx"])

    # Readout error
    ro_p = 0.02
    readout_err = ReadoutError([[1 - ro_p, ro_p], [ro_p, 1 - ro_p]])
    noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


def build_qnode(device):
    @qml.qnode(device, interface="torch")
    def q_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return q_circuit


class QLSTMModel(nn.Module):
    def __init__(self, qnode, n_features=9, n_lstm_units=32, num_layers=1, output_len=72):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_lstm_units,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classical_to_quantum = nn.Linear(n_lstm_units, N_QUBITS)
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.quantum_to_output = nn.Linear(N_QUBITS, output_len)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_lstm_output = lstm_out[:, -1, :]
        quantum_input = self.classical_to_quantum(final_lstm_output)
        quantum_features = self.q_layer(quantum_input)
        output = self.quantum_to_output(quantum_features)
        return torch.sigmoid(output)


def build_model(noisy=False):
    if noisy:
        noise_model = make_noise_model()
        dev = qml.device(
            "qiskit.aer",
            wires=N_QUBITS,
            backend="aer_simulator",
            noise_model=noise_model,
            shots=SHOTS,
        )
    else:
        dev = qml.device(
            "qiskit.aer",
            wires=N_QUBITS,
            backend="aer_simulator",
            shots=SHOTS,
        )

    qnode = build_qnode(dev)
    model = QLSTMModel(qnode)
    return model


def evaluate_with_probs(model, X, device):
    """Get raw probabilities (before thresholding) for ensemble methods"""
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            probs = model(xb).cpu().numpy()  # Raw sigmoid outputs [0, 1]
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def calibrate_probabilities(probs, y, method='temperature'):
    """Calibrate probabilities using temperature scaling or isotonic regression"""
    if method == 'temperature':
        # Temperature scaling (Platt scaling variant)
        from scipy.optimize import minimize_scalar
        
        def negative_log_likelihood(temp):
            calibrated = 1 / (1 + np.exp(-np.log(probs / (1 - probs + 1e-8)) / temp))
            return -np.sum(y * np.log(calibrated + 1e-8) + (1 - y) * np.log(1 - calibrated + 1e-8))
        
        result = minimize_scalar(negative_log_likelihood, bounds=(0.1, 10.0), method='bounded')
        temp = result.x
        calibrated = 1 / (1 + np.exp(-np.log(probs / (1 - probs + 1e-8)) / temp))
        return calibrated, temp
    
    elif method == 'isotonic':
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        calibrated = iso_reg.fit_transform(probs.flatten(), y.flatten())
        return calibrated.reshape(probs.shape), iso_reg
    
    return probs, None


def disagreement_focused_ensemble(probs_ideal, probs_noisy, y, train_idx, test_idx):
    """Train a model that focuses specifically on disagreement cases"""
    
    # Find where models disagree
    ideal_preds = (probs_ideal > 0.5).astype(int)
    noisy_preds = (probs_noisy > 0.5).astype(int)
    disagree_mask = (ideal_preds != noisy_preds)
    
    print(f"   Disagreement rate: {disagree_mask.mean():.1%}")
    
    # Build features for disagreement cases only
    N, T = probs_ideal.shape
    
    # Expand train/test masks to timestep level
    train_mask_full = np.repeat(np.isin(np.arange(N), train_idx), T)
    test_mask_full = np.repeat(np.isin(np.arange(N), test_idx), T)
    
    # Select only disagreement cases in training
    disagree_flat = disagree_mask.flatten()
    train_disagree = train_mask_full & disagree_flat
    test_disagree = test_mask_full & disagree_flat
    
    if train_disagree.sum() < 10:  # Too few disagreement cases
        return None, None
    
    # Features: probabilities, confidence, hour
    p_i_flat = probs_ideal.flatten()
    p_n_flat = probs_noisy.flatten()
    conf_i_flat = np.abs(p_i_flat - 0.5)
    conf_n_flat = np.abs(p_n_flat - 0.5)
    hours_flat = np.tile(np.arange(T), N) / (T - 1)
    
    X_features = np.column_stack([p_i_flat, p_n_flat, conf_i_flat, conf_n_flat, hours_flat])
    y_flat = y.flatten()
    
    # Train on disagreement cases only
    X_train = X_features[train_disagree]
    y_train = y_flat[train_disagree]
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Apply to all test cases (not just disagreements)
    X_test = X_features[test_mask_full]
    y_test = y_flat[test_mask_full]
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    return acc, clf


def per_hour_weights_ensemble(probs_ideal, probs_noisy, y, train_idx, test_idx):
    """Learn per-hour weights α_t for each timestep"""
    N, T = probs_ideal.shape
    
    # Per-hour weights
    weights_per_hour = np.zeros(T)
    
    train_probs_i = probs_ideal[train_idx]  # (N_train, T)
    train_probs_n = probs_noisy[train_idx]
    train_y = y[train_idx]
    
    for t in range(T):
        # For hour t, find optimal weight α_t
        p_i_t = train_probs_i[:, t]
        p_n_t = train_probs_n[:, t]
        y_t = train_y[:, t]
        
        # Try different weights and pick the best
        best_weight = 0.5
        best_acc = 0
        
        for alpha in np.linspace(0, 1, 11):
            ensemble_probs = alpha * p_i_t + (1 - alpha) * p_n_t
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            acc = accuracy_score(y_t, ensemble_preds)
            
            if acc > best_acc:
                best_acc = acc
                best_weight = alpha
        
        weights_per_hour[t] = best_weight
    
    # Apply to test set
    test_probs_i = probs_ideal[test_idx]
    test_probs_n = probs_noisy[test_idx]
    test_y = y[test_idx]
    
    ensemble_probs = weights_per_hour[None, :] * test_probs_i + (1 - weights_per_hour[None, :]) * test_probs_n
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    acc = accuracy_score(test_y.flatten(), ensemble_preds.flatten())
    return acc, weights_per_hour


def main():
    # Check files
    for p in [X_PATH, Y_PATH, LOC_PATH, MODEL_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    X, y, loc = load_data(SAMPLES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build models
    print("\nBuilding ideal model...")
    model_ideal = build_model(noisy=False).to(device)

    print("Building noisy model...")
    model_noisy = build_model(noisy=True).to(device)

    # Load identical weights
    print(f"\nLoading weights from {MODEL_PATH} into both models...")
    state = torch.load(MODEL_PATH, map_location="cpu")
    model_ideal.load_state_dict(state, strict=False)
    model_noisy.load_state_dict(state, strict=False)

    # Get predictions
    print("\nGetting predictions from both models...")
    probs_ideal = evaluate_with_probs(model_ideal, X, device)
    probs_noisy = evaluate_with_probs(model_noisy, X, device)

    # Split data
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Baseline accuracies
    ideal_preds = (probs_ideal > 0.5).astype(int)
    noisy_preds = (probs_noisy > 0.5).astype(int)
    voting_preds = np.where(ideal_preds == noisy_preds, ideal_preds, ideal_preds)
    
    ideal_acc_test = accuracy_score(y[test_idx].flatten(), ideal_preds[test_idx].flatten())
    noisy_acc_test = accuracy_score(y[test_idx].flatten(), noisy_preds[test_idx].flatten())
    voting_acc_test = accuracy_score(y[test_idx].flatten(), voting_preds[test_idx].flatten())
    
    print(f"\n📊 BASELINE ACCURACIES (Test Set):")
    print(f"   Ideal:  {ideal_acc_test:.4f}")
    print(f"   Noisy:  {noisy_acc_test:.4f}")
    print(f"   Voting: {voting_acc_test:.4f}")
    
    results = {
        'test_accuracies': {
            'ideal': float(ideal_acc_test),
            'noisy': float(noisy_acc_test),
            'voting': float(voting_acc_test)
        }
    }
    
    print(f"\n🔬 ADVANCED ENSEMBLE METHODS:")
    
    # Method 1: Calibration + Simple Average
    print("1. Calibrated averaging...")
    try:
        cal_probs_i, temp_i = calibrate_probabilities(probs_ideal[train_idx], y[train_idx])
        cal_probs_n, temp_n = calibrate_probabilities(probs_noisy[train_idx], y[train_idx])
        
        # Apply calibration to test set
        test_cal_i = 1 / (1 + np.exp(-np.log(probs_ideal[test_idx] / (1 - probs_ideal[test_idx] + 1e-8)) / temp_i))
        test_cal_n = 1 / (1 + np.exp(-np.log(probs_noisy[test_idx] / (1 - probs_noisy[test_idx] + 1e-8)) / temp_n))
        
        cal_avg_probs = (test_cal_i + test_cal_n) / 2
        cal_avg_acc = accuracy_score(y[test_idx].flatten(), (cal_avg_probs > 0.5).astype(int).flatten())
        
        print(f"   Calibrated Average: {cal_avg_acc:.4f} (vs voting: {cal_avg_acc - voting_acc_test:+.4f})")
        results['test_accuracies']['calibrated_average'] = float(cal_avg_acc)
    except Exception as e:
        print(f"   Calibrated Average: Failed ({e})")
    
    # Method 2: Disagreement-Focused Model
    print("2. Disagreement-focused ensemble...")
    disagree_acc, disagree_clf = disagreement_focused_ensemble(probs_ideal, probs_noisy, y, train_idx, test_idx)
    if disagree_acc is not None:
        print(f"   Disagreement Focus: {disagree_acc:.4f} (vs voting: {disagree_acc - voting_acc_test:+.4f})")
        results['test_accuracies']['disagreement_focused'] = float(disagree_acc)
    else:
        print("   Disagreement Focus: Failed (too few disagreements)")
    
    # Method 3: Per-Hour Weights
    print("3. Per-hour weighted ensemble...")
    hour_acc, hour_weights = per_hour_weights_ensemble(probs_ideal, probs_noisy, y, train_idx, test_idx)
    print(f"   Per-Hour Weights: {hour_acc:.4f} (vs voting: {hour_acc - voting_acc_test:+.4f})")
    print(f"   Weight range: [{hour_weights.min():.3f}, {hour_weights.max():.3f}] (0=noisy only, 1=ideal only)")
    results['test_accuracies']['per_hour_weights'] = float(hour_acc)
    results['hour_weights'] = hour_weights.tolist()
    
    # Find best method
    best_method = max(results['test_accuracies'].items(), key=lambda x: x[1])
    best_acc = best_method[1]
    best_name = best_method[0]
    
    print(f"\n🏆 BEST METHOD: {best_name}")
    print(f"   Accuracy: {best_acc:.4f}")
    print(f"   Improvement over voting: {best_acc - voting_acc_test:+.4f}")
    print(f"   Improvement over best individual: {best_acc - max(ideal_acc_test, noisy_acc_test):+.4f}")
    
    if best_acc > voting_acc_test:
        print(f"   ✅ BEATS VOTING by {(best_acc - voting_acc_test)*100:.2f}%!")
    else:
        print(f"   🤔 Voting still competitive")
    
    print(f"\n📋 SUMMARY:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
