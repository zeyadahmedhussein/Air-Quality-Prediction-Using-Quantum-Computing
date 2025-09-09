#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
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
OUTPUT_CSV = "ensemble_results.csv"

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


def ensemble_methods(probs_ideal, probs_noisy):
    """Apply different ensemble fusion strategies"""
    
    # Method 1: Simple averaging of probabilities
    avg_probs = (probs_ideal + probs_noisy) / 2
    avg_preds = (avg_probs > 0.5).astype(int)
    
    # Method 2: Majority voting (after individual thresholding)
    ideal_preds = (probs_ideal > 0.5).astype(int)
    noisy_preds = (probs_noisy > 0.5).astype(int)
    # When both agree, use that. When they disagree, use ideal (could also use noisy or random)
    voting_preds = np.where(ideal_preds == noisy_preds, ideal_preds, ideal_preds)
    
    # Method 3: Confidence-weighted averaging
    # Use distance from 0.5 as confidence measure
    ideal_conf = np.abs(probs_ideal - 0.5)
    noisy_conf = np.abs(probs_noisy - 0.5)
    total_conf = ideal_conf + noisy_conf + 1e-8  # Avoid div by zero
    
    # Weight by confidence
    weighted_probs = (probs_ideal * ideal_conf + probs_noisy * noisy_conf) / total_conf
    weighted_preds = (weighted_probs > 0.5).astype(int)
    
    # Method 4: Max confidence selection
    # For each prediction, pick the model that's more confident
    max_conf_preds = np.where(ideal_conf >= noisy_conf, ideal_preds, noisy_preds)
    
    # Method 5: Conservative ensemble (only predict positive if both are confident)
    # Positive if both models predict positive with >60% confidence
    conservative_preds = ((probs_ideal > 0.6) & (probs_noisy > 0.6)).astype(int)
    
    # Method 6: Optimistic ensemble (predict positive if either is confident)  
    optimistic_preds = ((probs_ideal > 0.6) | (probs_noisy > 0.6)).astype(int)
    
    return {
        'averaging': avg_preds,
        'voting': voting_preds,  
        'confidence_weighted': weighted_preds,
        'max_confidence': max_conf_preds,
        'conservative': conservative_preds,
        'optimistic': optimistic_preds
    }


def build_stacker_features(probs_ideal: np.ndarray, probs_noisy: np.ndarray, y: np.ndarray):
    """Build per-timestep feature matrix and labels for stacking.
    Shapes:
      probs_ideal, probs_noisy: (N, T)
      y: (N, T)
    Returns:
      X_feat: (N*T, D)
      y_flat: (N*T,)
    Features include: p_i, p_n, |p_i-0.5|, |p_n-0.5|, disagree, hour_index (0..T-1)
    """
    N, T = probs_ideal.shape
    p_i = probs_ideal.reshape(-1, 1)
    p_n = probs_noisy.reshape(-1, 1)
    m_i = np.abs(probs_ideal - 0.5).reshape(-1, 1)
    m_n = np.abs(probs_noisy - 0.5).reshape(-1, 1)
    disagree = ( (probs_ideal > 0.5) != (probs_noisy > 0.5) ).astype(np.float32).reshape(-1, 1)
    hours = np.tile(np.arange(T, dtype=np.float32), N).reshape(-1, 1) / max(1.0, (T - 1))

    X_feat = np.concatenate([p_i, p_n, m_i, m_n, disagree, hours], axis=1)
    y_flat = y.reshape(-1).astype(np.int32)
    return X_feat, y_flat


def train_logistic_stacker(probs_ideal, probs_noisy, y, train_sample_idx, test_sample_idx):
    """Train a logistic regression stacker on the training samples and evaluate on test samples.
    Returns (acc_test, acc_voting_test)
    """
    from sklearn.linear_model import LogisticRegression

    N, T = probs_ideal.shape
    # Build feature matrix and labels for all
    X_all, y_all = build_stacker_features(probs_ideal, probs_noisy, y)

    # Build masks expanded to all timesteps
    train_mask = np.repeat(np.isin(np.arange(N), train_sample_idx), T)
    test_mask = np.repeat(np.isin(np.arange(N), test_sample_idx), T)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    # Train logistic regression
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    clf.fit(X_train, y_train)

    # Predict
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc_test = accuracy_score(y_test, y_pred)

    # Voting baseline on the test subset
    ideal_preds = (probs_ideal > 0.5).astype(int)
    noisy_preds = (probs_noisy > 0.5).astype(int)
    voting = np.where(ideal_preds == noisy_preds, ideal_preds, ideal_preds)  # ideal tie-break
    voting_test = voting.reshape(-1)[test_mask]
    acc_voting_test = accuracy_score(y_test, voting_test)

    return acc_test, acc_voting_test


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
    try:
        model_ideal.load_state_dict(state, strict=False)
        model_noisy.load_state_dict(state, strict=False)
    except Exception:
        model_ideal.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
        model_noisy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)

    # Get predictions with probabilities for ensemble methods
    print("\nGetting predictions from both models...")
    probs_ideal = evaluate_with_probs(model_ideal, X, device)
    probs_noisy = evaluate_with_probs(model_noisy, X, device)
    
    # Individual model accuracies
    ideal_preds = (probs_ideal > 0.5).astype(int)
    noisy_preds = (probs_noisy > 0.5).astype(int)
    
    ideal_acc = accuracy_score(y.flatten(), ideal_preds.flatten())
    noisy_acc = accuracy_score(y.flatten(), noisy_preds.flatten())
    
    print(f"\nIndividual model accuracies:")
    print(f"  Ideal: {ideal_acc:.4f}")
    print(f"  Noisy: {noisy_acc:.4f}")

    # Split data for stacker training
    n_samples = len(X)
    n_train = int(0.7 * n_samples)  # 70% train, 30% test
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    print(f"\n🎯 TRAINING LOGISTIC REGRESSION STACKER:")
    print(f"   Training samples: {n_train}")
    print(f"   Test samples: {len(test_idx)}")
    
    # Train stacker
    stacker_acc, voting_acc = train_logistic_stacker(probs_ideal, probs_noisy, y, train_idx, test_idx)
    
    print(f"\n📊 STACKER RESULTS ON TEST SUBSET:")
    print(f"   Logistic Stacker: {stacker_acc:.4f}")
    print(f"   Voting Baseline:  {voting_acc:.4f}")
    print(f"   Stacker Gain:     {stacker_acc - voting_acc:+.4f}")
    
    if stacker_acc > voting_acc:
        print(f"   ✅ STACKER WINS! {(stacker_acc - voting_acc)*100:.2f}% improvement")
    else:
        print(f"   ❌ Voting still better by {(voting_acc - stacker_acc)*100:.2f}%")

    # Apply ensemble methods to full dataset for comparison
    print("\n📈 ENSEMBLE METHODS ON FULL DATASET:")
    ensemble_preds = ensemble_methods(probs_ideal, probs_noisy)
    
    results = {
        'ideal_accuracy': float(ideal_acc),
        'noisy_accuracy': float(noisy_acc),
        'stacker_test_accuracy': float(stacker_acc),
        'voting_test_accuracy': float(voting_acc),
        'stacker_gain': float(stacker_acc - voting_acc),
        'ensemble_accuracies': {}
    }
    
    print(f"Full dataset ensemble accuracies:")
    for method, preds in ensemble_preds.items():
        acc = accuracy_score(y.flatten(), preds.flatten())
        results['ensemble_accuracies'][method] = float(acc)
        improvement_ideal = acc - ideal_acc
        improvement_noisy = acc - noisy_acc
        best_improvement = max(improvement_ideal, improvement_noisy)
        
        print(f"  {method:20s}: {acc:.4f} (vs ideal: {improvement_ideal:+.4f}, vs noisy: {improvement_noisy:+.4f})")
        
        if best_improvement > 0:
            print(f"    ✅ IMPROVEMENT: {best_improvement:.4f} better than best individual model!")
    
    # Find best ensemble method
    best_method = max(results['ensemble_accuracies'].items(), key=lambda x: x[1])
    best_acc = best_method[1]
    best_name = best_method[0]
    
    print(f"\n🏆 BEST FULL DATASET METHOD: {best_name}")
    print(f"   Accuracy: {best_acc:.4f}")
    print(f"   Improvement over ideal: {best_acc - ideal_acc:+.4f}")
    print(f"   Improvement over noisy: {best_acc - noisy_acc:+.4f}")
    
    print(f"\n🔬 COMPARISON SUMMARY:")
    print(f"   Best Full Dataset Ensemble: {best_acc:.4f} ({best_name})")
    print(f"   Logistic Stacker (test):    {stacker_acc:.4f}")
    print(f"   Voting (test):              {voting_acc:.4f}")
    print(f"   Stacker advantage:          {stacker_acc - voting_acc:+.4f}")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(json.dumps(results, indent=2))
    
    # Save detailed per-example results for best method
    import csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "example_idx", "ideal_acc", "noisy_acc", f"best_ensemble_{best_name}_acc"
        ])
        
        best_preds = ensemble_preds[best_name]
        ideal_per_ex = (ideal_preds == y).mean(axis=1)
        noisy_per_ex = (noisy_preds == y).mean(axis=1)  
        best_per_ex = (best_preds == y).mean(axis=1)
        
        for i in range(len(X)):
            w.writerow([i, f"{ideal_per_ex[i]:.4f}", f"{noisy_per_ex[i]:.4f}", f"{best_per_ex[i]:.4f}"])
    
    print(f"\nSaved per-example results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
