"""
Quantum LSTM training module for NILE Competition
Supports both local simulation and IBM Quantum backends
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import argparse
import os
import json
import pickle
from datetime import datetime

# Quantum imports
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import BackendEstimator, Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeOptions, SamplerV2
from qiskit_aer import AerSimulator
import pennylane as qml


class QuantumCircuitLayer(nn.Module):
    """
    Quantum circuit layer that can work with both PennyLane and Qiskit
    """
    def __init__(self, n_qubits, n_layers, use_qiskit=False, backend_name=None):
        super(QuantumCircuitLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_qiskit = use_qiskit
        self.backend_name = backend_name
        
        if use_qiskit:
            self._init_qiskit_circuit()
        else:
            self._init_pennylane_circuit()
    
    def _init_pennylane_circuit(self):
        """Initialize PennyLane circuit for local simulation"""
        dev = qml.device("lightning.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def _init_qiskit_circuit(self):
        """Initialize Qiskit circuit for IBM Quantum backends"""
        # Create parameterized quantum circuit
        self.params = ParameterVector('θ', self.n_qubits + self.n_layers * self.n_qubits * 3)
        self.qc = QuantumCircuit(self.n_qubits)
        
        # Data encoding
        for i in range(self.n_qubits):
            self.qc.rx(self.params[i], i)
        
        # Variational layers
        param_idx = self.n_qubits
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                self.qc.rz(self.params[param_idx], qubit)
                self.qc.ry(self.params[param_idx + 1], qubit)
                self.qc.rz(self.params[param_idx + 2], qubit)
                param_idx += 3
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                self.qc.cnot(i, i + 1)
            if self.n_qubits > 1:
                self.qc.cnot(self.n_qubits - 1, 0)
        
        # Observable
        self.observables = [SparsePauliOp.from_operator(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # Initialize estimator
        if self.backend_name:
            # Use IBM Quantum backend
            service = QiskitRuntimeService()
            backend = service.backend(self.backend_name)
            self.estimator = BackendEstimator(backend)
        else:
            # Use local simulator
            self.estimator = Estimator()
    
    def forward(self, x):
        if self.use_qiskit:
            return self._forward_qiskit(x)
        else:
            return self._forward_pennylane(x)
    
    def _forward_pennylane(self, x):
        """Forward pass with PennyLane"""
        # Map input features to quantum circuit parameters
        quantum_input = x[:, :self.n_qubits]  # Take first n_qubits features
        return self.qlayer(quantum_input)
    
    def _forward_qiskit(self, x):
        """Forward pass with Qiskit (simplified for demonstration)"""
        # This is a simplified implementation
        # In practice, you'd need to handle batching and parameter binding properly
        batch_size = x.size(0)
        results = torch.zeros(batch_size, self.n_qubits)
        
        for i in range(batch_size):
            # Create parameter values
            data_params = x[i, :self.n_qubits].detach().numpy()
            # Initialize variational parameters (this would come from nn.Parameter in practice)
            var_params = np.random.normal(0, 0.1, self.n_layers * self.n_qubits * 3)
            all_params = np.concatenate([data_params, var_params])
            
            # Bind parameters and run circuit
            bound_circuit = self.qc.bind_parameters(all_params)
            job = self.estimator.run([bound_circuit] * self.n_qubits, self.observables)
            result = job.result()
            
            results[i] = torch.tensor([val for val in result.values])
        
        return results


class QuantumLSTMModel(nn.Module):
    """
    Hybrid Quantum-Classical LSTM model for multi-step forecasting
    """
    def __init__(self, n_features, n_lstm_units=32, n_qubits=4, n_layers=3, 
                 num_lstm_layers=1, output_len=72, use_qiskit=False, backend_name=None):
        super(QuantumLSTMModel, self).__init__()
        
        # Classical LSTM Layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Map LSTM output to quantum circuit input
        self.classical_to_quantum = nn.Linear(n_lstm_units, n_qubits)
        
        # Quantum Layer
        self.quantum_layer = QuantumCircuitLayer(n_qubits, n_layers, use_qiskit, backend_name)
        
        # Map quantum output to final predictions
        self.quantum_to_output = nn.Linear(n_qubits, output_len)
        
    def forward(self, x):
        # Classical LSTM processing
        lstm_out, _ = self.lstm(x)
        final_lstm_output = lstm_out[:, -1, :]  # Take last timestep
        
        # Prepare quantum input
        quantum_input = self.classical_to_quantum(final_lstm_output)
        
        # Quantum processing
        quantum_features = self.quantum_layer(quantum_input)
        
        # Final prediction
        output = self.quantum_to_output(quantum_features)
        
        return torch.sigmoid(output)


def train_model(model, train_loader, val_loader, epochs=25, patience=5, model_save_path=None, device=None):
    """Train the quantum LSTM model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting training on {device}...")
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    
    if model_save_path is None:
        model_save_path = 'best_quantum_lstm_model.pth'
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss, train_correct, train_samples = 0, 0, 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_samples += y_batch.size(0) * y_batch.size(1)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss /= train_samples
        train_acc = train_correct / train_samples
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_samples += y_batch.size(0) * y_batch.size(1)
        
        val_loss /= val_samples
        val_acc = val_correct / val_samples
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        scheduler.step(val_loss)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Duration: {epoch_duration:.2f}s")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print("  -> Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    return model, history


def plot_training_history(history, save_path=None):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (per timestep)')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('quantum_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def load_training_data(data_dir):
    """Load preprocessed training data"""
    print(f"Loading training data from {data_dir}")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    loc_train = np.load(os.path.join(data_dir, 'loc_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    loc_val = np.load(os.path.join(data_dir, 'loc_val.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, y_train, loc_train, X_val, y_val, loc_val


def main():
    parser = argparse.ArgumentParser(description="Train Quantum LSTM model")
    parser.add_argument("--data_dir", required=True, help="Directory containing preprocessed data")
    parser.add_argument("--model_save_path", default="best_quantum_lstm_model.pth", 
                       help="Path to save the best model")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--n_lstm_units", type=int, default=32, help="Number of LSTM units")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of quantum layers")
    parser.add_argument("--num_lstm_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--n_features", type=int, default=9, help="Number of input features")
    parser.add_argument("--output_len", type=int, default=72, help="Output sequence length")
    parser.add_argument("--use_qiskit", action="store_true", help="Use Qiskit backend instead of PennyLane")
    parser.add_argument("--backend", help="Qiskit backend name (for IBM Quantum)")
    parser.add_argument("--plot_history", action="store_true", help="Plot training history")
    
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, loc_train, X_val, y_val, loc_val = load_training_data(args.data_dir)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = QuantumLSTMModel(
        n_features=args.n_features,
        n_lstm_units=args.n_lstm_units,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        num_lstm_layers=args.num_lstm_layers,
        output_len=args.output_len,
        use_qiskit=args.use_qiskit,
        backend_name=args.backend
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")
    print(f"Using quantum backend: {'Qiskit' if args.use_qiskit else 'PennyLane'}")
    if args.backend:
        print(f"IBM Quantum backend: {args.backend}")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, patience=args.patience,
        model_save_path=args.model_save_path
    )
    
    # Save training history
    history_path = args.model_save_path.replace('.pth', '_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(dict(history), f)
    print(f"Training history saved to {history_path}")
    
    # Save model config
    config = {
        'n_features': args.n_features,
        'n_lstm_units': args.n_lstm_units,
        'n_qubits': args.n_qubits,
        'n_layers': args.n_layers,
        'num_lstm_layers': args.num_lstm_layers,
        'output_len': args.output_len,
        'use_qiskit': args.use_qiskit,
        'backend_name': args.backend,
        'total_params': total_params,
        'final_val_loss': min(history['val_loss']),
        'final_val_accuracy': max(history['val_accuracy']),
        'training_time': datetime.now().isoformat()
    }
    
    config_path = args.model_save_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to {config_path}")
    
    # Plot training history if requested
    if args.plot_history:
        plot_path = args.model_save_path.replace('.pth', '_history.png')
        plot_training_history(history, plot_path)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
