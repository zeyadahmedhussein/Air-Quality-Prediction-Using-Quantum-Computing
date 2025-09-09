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
MODEL_PATH = "best_qlstm_model_multistep.pth"  # must exist in CWD
OUTPUT_CSV = "compare_ideal_vs_noisy_results.csv"

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
    """Construct a simple synthetic noise model using Qiskit Aer (no external backend).
    - Small depolarizing noise on 1- and 2-qubit gates
    - Small readout error
    """
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

    # Readout error (symmetric, mild)
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


def evaluate(model, X, y, device):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            out = (out > 0.5).astype(np.int32)
            preds.append(out)
            labels.append(yb.numpy().astype(np.int32))
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    # Flatten across sequence dimension to compute overall accuracy
    acc = accuracy_score(labels.flatten(), preds.flatten())
    per_sample_acc = (labels == preds).mean(axis=1)  # accuracy per example across 72 steps
    return acc, preds, labels, per_sample_acc


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
    # Some Torch versions require weights_only flag; handle robustly
    try:
        model_ideal.load_state_dict(state, strict=False)
        model_noisy.load_state_dict(state, strict=False)
    except Exception:
        model_ideal.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
        model_noisy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)

    # Evaluate
    print("\nEvaluating on 1000-sample subset (overall sequence-averaged accuracy)...")
    acc_ideal, preds_ideal, labels, per_ex_ideal = evaluate(model_ideal, X, y, device)
    acc_noisy, preds_noisy, _, per_ex_noisy = evaluate(model_noisy, X, y, device)

    print(f"Ideal accuracy: {acc_ideal:.4f}")
    print(f"Noisy accuracy: {acc_noisy:.4f}")

    # Compare correctness flips at the flattened level
    correct_ideal = (preds_ideal.flatten() == labels.flatten())
    correct_noisy = (preds_noisy.flatten() == labels.flatten())

    both_correct = np.sum(correct_ideal & correct_noisy)
    ideal_only = np.sum(correct_ideal & (~correct_noisy))
    noisy_only = np.sum((~correct_ideal) & correct_noisy)
    both_wrong = np.sum((~correct_ideal) & (~correct_noisy))

    print("\nFlip analysis (flattened across all steps):")
    print(f"  Both correct: {both_correct}")
    print(f"  Ideal only correct: {ideal_only}")
    print(f"  Noisy only correct: {noisy_only}")
    print(f"  Both wrong: {both_wrong}")

    # Per-example which side wins more often
    per_ex_better_noisy = np.sum(per_ex_noisy > per_ex_ideal)
    per_ex_better_ideal = np.sum(per_ex_ideal > per_ex_noisy)
    per_ex_tie = np.sum(np.isclose(per_ex_noisy, per_ex_ideal))

    print("\nPer-example wins (average over 72 outputs):")
    print(f"  Noisy better: {per_ex_better_noisy}")
    print(f"  Ideal better: {per_ex_better_ideal}")
    print(f"  Tie: {per_ex_tie}")

    # Save results
    summary = {
        "samples": int(X.shape[0]),
        "seq_len": int(y.shape[1]),
        "ideal_accuracy": float(acc_ideal),
        "noisy_accuracy": float(acc_noisy),
        "both_correct": int(both_correct),
        "ideal_only": int(ideal_only),
        "noisy_only": int(noisy_only),
        "both_wrong": int(both_wrong),
        "per_example_noisy_better": int(per_ex_better_noisy),
        "per_example_ideal_better": int(per_ex_better_ideal),
        "per_example_tie": int(per_ex_tie),
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    # Also write per-example comparison CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "per_ex_acc_ideal", "per_ex_acc_noisy",
        ])
        for i, (pi, pn) in enumerate(zip(per_ex_ideal, per_ex_noisy)):
            w.writerow([i, f"{pi:.6f}", f"{pn:.6f}"])

    print(f"\nSaved per-example comparison to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

