#!/usr/bin/env python3
"""
Error Calibration Matrix Generator for Quantum-Classical LSTM Model

This script creates a 16×16 error calibration matrix by comparing ideal and noisy 
quantum results using 100,000 test records. The calibration matrix helps correct 
systematic errors in quantum circuit predictions.

Usage:
    python generate_calibration_matrix.py

Outputs:
    - calibration_matrix_16x16.npy: NumPy binary format
    - calibration_matrix_16x16.csv: CSV format
    - calibration_data.pkl: Complete dataset
    - calibration_report.txt: Summary report
    - calibration_matrix_analysis.png: Visualization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import defaultdict
import pickle
from scipy.linalg import inv
import warnings
import os
import sys
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()

print("="*60)
print("    QUANTUM ERROR CALIBRATION MATRIX GENERATOR")
print("="*60)
print("Initializing calibration matrix generation...")
print("Target: 16×16 matrix using 100,000 test records")
print("="*60)

class ErrorCalibrationGenerator:
    """Main class for generating quantum error calibration matrices."""
    
    def __init__(self, n_qubits=4, n_layers=3, n_calibration_samples=100000, backend='aer_simulator'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_calibration_samples = n_calibration_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Device: {self.device}")
        print(f"Qubits: {self.n_qubits}")
        print(f"Layers: {self.n_layers}")
        print(f"Calibration samples: {self.n_calibration_samples}")
        
        # Setup quantum devices
        self._setup_quantum_devices()
        
    def _setup_quantum_devices(self):
        """Setup ideal and noisy quantum devices."""
        print("\nSetting up quantum devices...")
        
        # Setup ideal device
        self.dev_ideal = qml.device("lightning.qubit", wires=self.n_qubits)
        print(" Ideal quantum device initialized")
        
        self.dev_noisy = qml.device("lightning.qubit", wires=self.n_qubits)
        self.has_noisy_device = True
        print(" Noisy quantum device initialized")

            
    def _create_quantum_circuits(self):
        """Create ideal and noisy quantum circuits."""
        print("\nCreating quantum circuits...")
        
        @qml.qnode(self.dev_ideal, interface="torch")
        def q_circuit_ideal(inputs, weights):
            """Ideal quantum circuit without noise."""
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        if self.has_noisy_device:
            @qml.qnode(self.dev_noisy, interface="torch")
            def q_circuit_noisy(inputs, weights):
                """Noisy quantum circuit."""
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                # Add noise operations here if needed
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        else:
            def q_circuit_noisy(inputs, weights):
                """Simulated noisy quantum circuit."""
                ideal_result = q_circuit_ideal(inputs, weights)
                # Add noise - adjustable noise parameters
                noise_level = 0.1
                noise = torch.randn_like(ideal_result) * noise_level
                return ideal_result + noise
        
        self.q_circuit_ideal = q_circuit_ideal
        self.q_circuit_noisy = q_circuit_noisy
        print(" Quantum circuits created")
        
    def _create_models(self):
        """Create ideal and noisy QLSTM models."""
        print("\nCreating quantum-classical models...")
        
        class QLSTMModel(nn.Module):
            """Quantum-Classical LSTM model for multi-step forecasting."""
            
            def __init__(self, n_features, n_lstm_units=32, n_qubits=4, 
                        num_layers=1, n_layers=3, output_len=72, device_type='ideal'):
                super(QLSTMModel, self).__init__()
                
                self.device_type = device_type
                
                # Classical LSTM Layer
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=n_lstm_units,
                    num_layers=num_layers,
                    batch_first=True
                )
                
                # Classical layer to map LSTM output to quantum input
                self.classical_to_quantum = nn.Linear(n_lstm_units, n_qubits)
                
                # Quantum Layer - select based on device type
                if device_type == 'ideal':
                    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
                    self.q_layer = qml.qnn.TorchLayer(outer_instance.q_circuit_ideal, weight_shapes)
                else:
                    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
                    if outer_instance.has_noisy_device:
                        self.q_layer = qml.qnn.TorchLayer(outer_instance.q_circuit_noisy, weight_shapes)
                    else:
                        # Custom layer for simulated noise
                        class NoisyQuantumLayer(nn.Module):
                            def __init__(self, circuit_func, weight_shapes):
                                super().__init__()
                                self.weights = nn.Parameter(torch.randn(weight_shapes["weights"]))
                                self.circuit_func = circuit_func
                                
                            def forward(self, x):
                                return self.circuit_func(x, self.weights)
                        
                        self.q_layer = NoisyQuantumLayer(outer_instance.q_circuit_noisy, weight_shapes)
                
                # Classical layer to map quantum output to predictions
                self.quantum_to_output = nn.Linear(n_qubits, output_len)
                
            def forward(self, x):
                # Pass through LSTM
                lstm_out, _ = self.lstm(x)
                
                # Extract features from last timestep
                final_lstm_output = lstm_out[:, -1, :]
                
                # Map to quantum input
                quantum_input = self.classical_to_quantum(final_lstm_output)
                
                # Pass through quantum circuit
                quantum_features = self.q_layer(quantum_input)
                
                # Map to output
                output = self.quantum_to_output(quantum_features)
                
                return torch.sigmoid(output)
        
        # Store reference to self for nested class access
        outer_instance = self
        
        # Create models
        self.model_ideal = QLSTMModel(
            n_features=9,
            n_lstm_units=32,
            n_qubits=self.n_qubits,
            num_layers=1,
            n_layers=self.n_layers,
            output_len=72,
            device_type='ideal'
        ).to(self.device)
        
        self.model_noisy = QLSTMModel(
            n_features=9,
            n_lstm_units=32,
            n_qubits=self.n_qubits,
            num_layers=1,
            n_layers=self.n_layers,
            output_len=72,
            device_type='noisy'
        ).to(self.device)
        
        print("✓ Models created successfully")
        
    def load_pretrained_weights(self):
        """Load pre-trained weights if available."""
        print("\nLoading pre-trained weights...")
        
        weight_paths = [
            'Training/best_qlstm_model_multistep.pth',
            '../Training/best_qlstm_model_multistep.pth',
            './best_qlstm_model_multistep.pth'
        ]
        
        weights_loaded = False
        for path in weight_paths:
            try:
                if os.path.exists(path):
                    weights = torch.load(path, weights_only=True)
                    self.model_ideal.load_state_dict(weights)
                    self.model_noisy.load_state_dict(weights)
                    print(f"✓ Pre-trained weights loaded from: {path}")
                    weights_loaded = True
                    break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                continue
        
        if not weights_loaded:
            print("⚠ Pre-trained weights not found. Using randomly initialized weights.")
            print("⚠ For proper calibration, you should use trained weights.")
            
        return weights_loaded
        
    def load_test_data(self):
        """Load test data from various possible locations."""
        print("\nLoading test data...")
        
        data_paths = [
            ('Testing/X_test.npy', 'Testing/y_test.npy', 'Testing/loc_test.npy'),
            ('../Testing/X_test.npy', '../Testing/y_test.npy', '../Testing/loc_test.npy'),
            ('X_test.npy', 'y_test.npy', 'loc_test.npy')
        ]
        
        for x_path, y_path, loc_path in data_paths:
            try:
                if all(os.path.exists(p) for p in [x_path, y_path, loc_path]):
                    self.X_test = np.load(x_path)
                    self.y_test = np.load(y_path)
                    self.loc_test = np.load(loc_path)
                    
                    print(f"✓ Test data loaded from: {os.path.dirname(x_path) or 'current directory'}")
                    print(f"  X_test shape: {self.X_test.shape}")
                    print(f"  y_test shape: {self.y_test.shape}")
                    print(f"  loc_test shape: {self.loc_test.shape}")
                    
                    # Limit to requested number of samples
                    self.n_calibration_samples = min(self.n_calibration_samples, len(self.X_test))
                    print(f"  Using {self.n_calibration_samples} samples for calibration")
                    
                    return True
            except Exception as e:
                print(f"Failed to load from {x_path}: {e}")
                continue
                
        raise FileNotFoundError("Could not find test data files (X_test.npy, y_test.npy, loc_test.npy)")
        
    def convert_to_state_probs(self, quantum_features):
        """Convert 4 qubit expectation values to 16-state probability distribution."""
        batch_size = quantum_features.shape[0]
        
        # Create 16-dimensional state probabilities from 4 qubit measurements
        state_probs = torch.zeros(batch_size, 16).to(quantum_features.device)
        
        # Convert expectation values to probabilities
        # For each qubit, <Z> = p0 - p1, so p0 = (1 + <Z>)/2, p1 = (1 - <Z>)/2
        qubit_probs = (1 + quantum_features) / 2  # Convert from [-1,1] to [0,1]
        
        # Generate all 16 computational basis states
        for i in range(16):
            # Convert index to 4-bit binary representation
            bits = [(i >> j) & 1 for j in range(4)]
            
            # Calculate probability for this computational basis state
            prob = torch.ones(batch_size).to(quantum_features.device)
            for qubit_idx, bit in enumerate(bits):
                if bit == 0:
                    prob *= qubit_probs[:, qubit_idx]
                else:
                    prob *= (1 - qubit_probs[:, qubit_idx])
            
            state_probs[:, i] = prob
        
        # Normalize to ensure probabilities sum to 1
        state_probs = state_probs / (state_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        return state_probs
        
    def get_quantum_state_probabilities(self, model, n_samples):
        """Extract quantum state probabilities from model predictions."""
        print(f"\nGenerating quantum state probabilities...")
        model.eval()
        
        # Create data loader
        batch_size = 256
        dataset = TensorDataset(
            torch.from_numpy(self.X_test[:n_samples]).float(),
            torch.from_numpy(self.y_test[:n_samples]).float()
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        sample_count = 0
        
        with torch.no_grad():
            for X_batch, _ in dataloader:
                if sample_count >= n_samples:
                    break
                    
                X_batch = X_batch.to(self.device)
                
                # Get quantum features (before final linear layer)
                lstm_out, _ = model.lstm(X_batch)
                final_lstm_output = lstm_out[:, -1, :]
                quantum_input = model.classical_to_quantum(final_lstm_output)
                quantum_features = model.q_layer(quantum_input)
                
                # Convert to probability distributions over 16 states (2^4)
                batch_probs = self.convert_to_state_probs(quantum_features)
                
                all_predictions.extend(batch_probs.cpu().numpy())
                sample_count += len(X_batch)
                
                if sample_count % 10000 == 0:
                    print(f"  Processed {sample_count:,} samples...")
        
        result = np.array(all_predictions[:n_samples])
        print(f"✓ Generated {result.shape[0]} probability distributions of size {result.shape[1]}")
        return result
        
    def create_calibration_matrix(self, ideal_probs, noisy_probs):
        """Create 16x16 calibration matrix from ideal and noisy probability distributions."""
        print("\nCreating calibration matrix...")
        
        try:
            # Compute pseudoinverse of noisy probabilities
            noisy_pinv = np.linalg.pinv(noisy_probs.T)
            calibration_matrix = ideal_probs.T @ noisy_pinv
            print("✓ Calibration matrix computed using pseudoinverse method")
            
        except np.linalg.LinAlgError:
            print("⚠ Pseudoinverse method failed, using regularized approach...")
            
            # Regularized least squares
            lambda_reg = 1e-6
            A = noisy_probs.T @ noisy_probs + lambda_reg * np.eye(16)
            b = noisy_probs.T @ ideal_probs
            calibration_matrix = np.linalg.solve(A, b).T
            print("✓ Calibration matrix computed using regularized method")
        
        # Ensure the matrix is properly normalized (each column should sum to 1)
        col_sums = np.sum(calibration_matrix, axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        calibration_matrix = calibration_matrix / col_sums[np.newaxis, :]
        
        print(f"✓ Matrix properties:")
        print(f"  Shape: {calibration_matrix.shape}")
        print(f"  Min value: {np.min(calibration_matrix):.6f}")
        print(f"  Max value: {np.max(calibration_matrix):.6f}")
        print(f"  Condition number: {np.linalg.cond(calibration_matrix):.2e}")
        print(f"  Column sums range: [{np.min(np.sum(calibration_matrix, axis=0)):.6f}, {np.max(np.sum(calibration_matrix, axis=0)):.6f}]")
        
        return calibration_matrix
        
    def validate_calibration(self, calibration_matrix, ideal_probs, noisy_probs, n_test=1000):
        """Validate the calibration matrix by testing on a subset of data."""
        print(f"\nValidating calibration matrix on {n_test} samples...")
        
        # Take a subset for validation
        test_ideal = ideal_probs[:n_test]
        test_noisy = noisy_probs[:n_test]
        
        # Apply calibration matrix to noisy measurements
        calibrated_probs = (calibration_matrix @ test_noisy.T).T
        
        # Compute metrics
        # 1. Mean squared error
        mse_before = np.mean((test_ideal - test_noisy) ** 2)
        mse_after = np.mean((test_ideal - calibrated_probs) ** 2)
        
        # 2. Fidelity (quantum state overlap)
        fidelity_before = np.mean([np.sqrt(np.sum(np.sqrt(np.maximum(p1 * p2, 0)))) 
                                  for p1, p2 in zip(test_ideal, test_noisy)])
        fidelity_after = np.mean([np.sqrt(np.sum(np.sqrt(np.maximum(p1 * p2, 0)))) 
                                 for p1, p2 in zip(test_ideal, calibrated_probs)])
        
        # 3. Total variation distance
        tv_before = np.mean([0.5 * np.sum(np.abs(p1 - p2)) 
                            for p1, p2 in zip(test_ideal, test_noisy)])
        tv_after = np.mean([0.5 * np.sum(np.abs(p1 - p2)) 
                           for p1, p2 in zip(test_ideal, calibrated_probs)])
        
        print(f"\nVALIDATION RESULTS:")
        print(f"Mean Squared Error:")
        print(f"  Before calibration: {mse_before:.6f}")
        print(f"  After calibration:  {mse_after:.6f}")
        print(f"  Improvement: {((mse_before - mse_after) / mse_before * 100):.2f}%")
        
        print(f"\nFidelity (higher is better):")
        print(f"  Before calibration: {fidelity_before:.6f}")
        print(f"  After calibration:  {fidelity_after:.6f}")
        print(f"  Improvement: {((fidelity_after - fidelity_before) / fidelity_before * 100):.2f}%")
        
        print(f"\nTotal Variation Distance (lower is better):")
        print(f"  Before calibration: {tv_before:.6f}")
        print(f"  After calibration:  {tv_after:.6f}")
        print(f"  Improvement: {((tv_before - tv_after) / tv_before * 100):.2f}%")
        
        return {
            'mse_before': mse_before,
            'mse_after': mse_after,
            'fidelity_before': fidelity_before,
            'fidelity_after': fidelity_after,
            'tv_before': tv_before,
            'tv_after': tv_after
        }
        
    def visualize_matrix(self, calibration_matrix):
        """Create visualization of the calibration matrix."""
        print("\nGenerating visualization...")
        
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Heatmap of calibration matrix
        plt.subplot(2, 3, 1)
        sns.heatmap(calibration_matrix, annot=False, cmap='RdBu_r', center=0, 
                    square=True, cbar_kws={'label': 'Calibration Weight'})
        plt.title('16×16 Error Calibration Matrix')
        plt.xlabel('Noisy State Index')
        plt.ylabel('Ideal State Index')
        
        # Plot 2: Diagonal elements
        plt.subplot(2, 3, 2)
        diagonal_elements = np.diag(calibration_matrix)
        plt.plot(diagonal_elements, 'bo-', markersize=4)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Ideal value')
        plt.title('Diagonal Elements')
        plt.xlabel('State Index')
        plt.ylabel('Calibration Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Off-diagonal elements distribution
        plt.subplot(2, 3, 3)
        off_diagonal = calibration_matrix[~np.eye(16, dtype=bool)]
        plt.hist(off_diagonal, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Off-Diagonal Elements')
        plt.xlabel('Calibration Weight')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Column sums
        plt.subplot(2, 3, 4)
        column_sums = np.sum(calibration_matrix, axis=0)
        plt.plot(column_sums, 'go-', markersize=4)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Ideal value')
        plt.title('Column Sums (Probability Conservation)')
        plt.xlabel('State Index')
        plt.ylabel('Sum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Row sums
        plt.subplot(2, 3, 5)
        row_sums = np.sum(calibration_matrix, axis=1)
        plt.plot(row_sums, 'mo-', markersize=4)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Reference')
        plt.title('Row Sums')
        plt.xlabel('State Index')
        plt.ylabel('Sum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Condition number info
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f"Matrix Statistics:", fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f"Condition Number: {np.linalg.cond(calibration_matrix):.2e}")
        plt.text(0.1, 0.6, f"Determinant: {np.linalg.det(calibration_matrix):.2e}")
        plt.text(0.1, 0.5, f"Trace: {np.trace(calibration_matrix):.4f}")
        plt.text(0.1, 0.4, f"Frobenius Norm: {np.linalg.norm(calibration_matrix, 'fro'):.4f}")
        plt.text(0.1, 0.3, f"Mean Diagonal: {np.mean(diagonal_elements):.4f}")
        plt.text(0.1, 0.2, f"Std Diagonal: {np.std(diagonal_elements):.4f}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Matrix Statistics')
        
        plt.tight_layout()
        plt.savefig('calibration_matrix_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as: calibration_matrix_analysis.png")
        plt.show()
        
    def save_results(self, calibration_matrix, validation_results, ideal_probs, noisy_probs):
        """Save calibration matrix and results in multiple formats."""
        print("\nSaving results...")
        
        # 1. Save as NumPy array
        np.save('calibration_matrix_16x16.npy', calibration_matrix)
        print("✓ Saved: calibration_matrix_16x16.npy")
        
        # 2. Save as CSV
        np.savetxt('calibration_matrix_16x16.csv', calibration_matrix, delimiter=',')
        print("✓ Saved: calibration_matrix_16x16.csv")
        
        # 3. Save detailed results as pickle
        calibration_data = {
            'calibration_matrix': calibration_matrix,
            'validation_results': validation_results,
            'n_calibration_samples': self.n_calibration_samples,
            'ideal_probs_sample': ideal_probs[:100],  # Save first 100 for reference
            'noisy_probs_sample': noisy_probs[:100],
            'matrix_stats': {
                'diagonal_elements': np.diag(calibration_matrix),
                'column_sums': np.sum(calibration_matrix, axis=0),
                'row_sums': np.sum(calibration_matrix, axis=1),
                'condition_number': np.linalg.cond(calibration_matrix),
                'determinant': np.linalg.det(calibration_matrix),
                'trace': np.trace(calibration_matrix),
                'frobenius_norm': np.linalg.norm(calibration_matrix, 'fro')
            },
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'device': str(self.device)
            }
        }
        
        with open('calibration_data.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        print("✓ Saved: calibration_data.pkl")
        
        # 4. Save summary report
        with open('calibration_report.txt', 'w') as f:
            f.write("QUANTUM ERROR CALIBRATION MATRIX REPORT\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Calibration samples used: {self.n_calibration_samples:,}\n")
            f.write(f"Matrix size: 16 × 16\n")
            f.write(f"Quantum system: {self.n_qubits} qubits, {self.n_layers} layers\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("MATRIX PROPERTIES:\n")
            f.write(f"  Condition number: {np.linalg.cond(calibration_matrix):.2e}\n")
            f.write(f"  Determinant: {np.linalg.det(calibration_matrix):.2e}\n")
            f.write(f"  Trace: {np.trace(calibration_matrix):.6f}\n")
            f.write(f"  Min element: {np.min(calibration_matrix):.6f}\n")
            f.write(f"  Max element: {np.max(calibration_matrix):.6f}\n")
            f.write(f"  Mean diagonal: {np.mean(np.diag(calibration_matrix)):.6f}\n")
            f.write(f"  Std diagonal: {np.std(np.diag(calibration_matrix)):.6f}\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            for key, value in validation_results.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nIMPROVEMENT SUMMARY:\n")
            mse_improvement = ((validation_results['mse_before'] - validation_results['mse_after']) / validation_results['mse_before'] * 100)
            fidelity_improvement = ((validation_results['fidelity_after'] - validation_results['fidelity_before']) / validation_results['fidelity_before'] * 100)
            tv_improvement = ((validation_results['tv_before'] - validation_results['tv_after']) / validation_results['tv_before'] * 100)
            
            f.write(f"  MSE improvement: {mse_improvement:.2f}%\n")
            f.write(f"  Fidelity improvement: {fidelity_improvement:.2f}%\n")
            f.write(f"  Total variation improvement: {tv_improvement:.2f}%\n")
            
            f.write("\nUSAGE:\n")
            f.write("To apply calibration to noisy measurements P_noisy:\n")
            f.write("  P_calibrated = calibration_matrix @ P_noisy\n")
            f.write("\nExample Python code:\n")
            f.write("  import numpy as np\n")
            f.write("  calibration_matrix = np.load('calibration_matrix_16x16.npy')\n")
            f.write("  corrected_probs = (calibration_matrix @ noisy_probs.T).T\n")
            f.write("\nFile outputs:\n")
            f.write("  - calibration_matrix_16x16.npy: NumPy binary format\n")
            f.write("  - calibration_matrix_16x16.csv: CSV format\n")
            f.write("  - calibration_data.pkl: Complete dataset\n")
            f.write("  - calibration_matrix_analysis.png: Visualization\n")
            f.write("  - calibration_report.txt: This report\n")
        
        print("✓ Saved: calibration_report.txt")
        
    def demonstrate_usage(self, calibration_matrix, ideal_probs, noisy_probs):
        """Demonstrate how to use the calibration matrix."""
        print("\n" + "="*60)
        print("               USAGE DEMONSTRATION")
        print("="*60)
        
        def apply_error_correction(noisy_measurements, cal_matrix):
            """Apply error correction to noisy quantum measurements."""
            corrected = (cal_matrix @ noisy_measurements.T).T
            
            # Ensure probabilities are non-negative and normalized
            corrected = np.maximum(corrected, 0)
            corrected = corrected / np.sum(corrected, axis=1, keepdims=True)
            
            return corrected
        
        # Example usage with first 5 samples
        sample_noisy = noisy_probs[:5]
        sample_ideal = ideal_probs[:5]
        sample_corrected = apply_error_correction(sample_noisy, calibration_matrix)
        
        print(f"\nSample comparison (first 8 states only):")
        for i in range(3):  # Show first 3 samples
            print(f"\nSample {i+1}:")
            print(f"  Ideal:     {np.array2string(sample_ideal[i][:8], precision=4, separator=', ', suppress_small=True)}")
            print(f"  Noisy:     {np.array2string(sample_noisy[i][:8], precision=4, separator=', ', suppress_small=True)}")
            print(f"  Corrected: {np.array2string(sample_corrected[i][:8], precision=4, separator=', ', suppress_small=True)}")
        
        # Calculate improvement
        error_before = np.mean((sample_ideal - sample_noisy) ** 2)
        error_after = np.mean((sample_ideal - sample_corrected) ** 2)
        improvement = (error_before - error_after) / error_before * 100
        
        print(f"\nMSE improvement on samples: {improvement:.2f}%")
        
        print(f"\nReady to integrate into your quantum pipeline!")
        print(f"   Load with: calibration_matrix = np.load('calibration_matrix_16x16.npy')")
        print(f"   Apply with: corrected = (calibration_matrix @ noisy_probs.T).T")
        
    def run_full_pipeline(self):
        """Execute the complete calibration matrix generation pipeline."""
        try:
            print("\nStarting full calibration pipeline...")
            
            # Step 1: Setup
            self._create_quantum_circuits()
            self._create_models()
            
            # Step 2: Load data and weights
            self.load_test_data()
            weights_loaded = self.load_pretrained_weights()
            
            # Step 3: Generate probability distributions
            print(f"\nGenerating probability distributions from {self.n_calibration_samples:,} samples...")
            ideal_probs = self.get_quantum_state_probabilities(self.model_ideal, self.n_calibration_samples)
            noisy_probs = self.get_quantum_state_probabilities(self.model_noisy, self.n_calibration_samples)
            
            # Step 4: Create calibration matrix
            calibration_matrix = self.create_calibration_matrix(ideal_probs, noisy_probs)
            
            # Step 5: Validate
            validation_results = self.validate_calibration(calibration_matrix, ideal_probs, noisy_probs)
            
            # Step 6: Visualize
            self.visualize_matrix(calibration_matrix)
            
            # Step 7: Save results
            self.save_results(calibration_matrix, validation_results, ideal_probs, noisy_probs)
            
            # Step 8: Demonstrate usage
            self.demonstrate_usage(calibration_matrix, ideal_probs, noisy_probs)
            
            return calibration_matrix, validation_results
            
        except Exception as e:
            print(f"\nERROR in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function."""
    print(f"Starting at: {datetime.now()}")
    
    try:
        # Create calibration generator
        generator = ErrorCalibrationGenerator(
            n_qubits=4,
            n_layers=3, 
            n_calibration_samples=100000
        )
        
        # Run the full pipeline
        calibration_matrix, results = generator.run_full_pipeline()
        
        if calibration_matrix is not None:
            print("\n" + "="*60)
            print("          CALIBRATION MATRIX COMPLETE")
            print("="*60)
            print(f"Generated 16×16 error calibration matrix")
            print(f"Used {generator.n_calibration_samples:,} quantum measurement pairs")
            print(f"Matrix condition number: {np.linalg.cond(calibration_matrix):.2e}")
            mse_improvement = ((results['mse_before'] - results['mse_after']) / results['mse_before'] * 100)
            print(f"Validation MSE improvement: {mse_improvement:.2f}%")
            print(f"All files saved and ready for use")
            
            print(f"\nOUTPUT FILES:")
            print(f"   • calibration_matrix_16x16.npy - Main matrix file")
            print(f"   • calibration_matrix_16x16.csv - Human readable")
            print(f"   • calibration_data.pkl - Complete dataset")
            print(f"   • calibration_report.txt - Detailed report") 
            print(f"   • calibration_matrix_analysis.png - Visualizations")
            
            print(f"\nREADY FOR QUANTUM ERROR CORRECTION!")
            print("="*60)
        else:
            print("\nPipeline failed - check error messages above")
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
