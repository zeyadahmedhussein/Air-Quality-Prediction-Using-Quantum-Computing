"""
Quantum LSTM Training Module
Follows classical model structure and Phase 3 IBM Quantum Deployment Guidelines
Supports ideal simulation, noisy simulation, and real hardware execution
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
import pennylane as qml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from quantum_devices import QuantumDeviceManager, create_quantum_circuit


class QuantumLSTMModel(nn.Module):
    """
    Quantum-Classical LSTM Model for multi-step time series forecasting
    Following classical model pattern with quantum enhancement
    """
    def __init__(self, n_features, n_lstm_units=32, n_qubits=4, n_layers=3, 
                 num_lstm_layers=1, output_len=72, device_type="ideal", 
                 device_manager=None):
        super(QuantumLSTMModel, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_type = device_type
        
        # Classical LSTM Layer (same as classical model)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0.0
        )
        
        # Classical layer to map LSTM output to quantum input
        self.classical_to_quantum = nn.Linear(n_lstm_units, n_qubits)
        
        # Initialize quantum device and circuit
        if device_manager is None:
            device_manager = QuantumDeviceManager()
        
        self.device_manager = device_manager
        self.quantum_device = device_manager.create_device(n_qubits, device_type)
        self.quantum_circuit = create_quantum_circuit(n_qubits, n_layers, self.quantum_device)
        
        # Create PennyLane quantum layer
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(self.quantum_circuit, weight_shapes)
        
        # Classical layer to map quantum output to predictions
        self.quantum_to_output = nn.Linear(n_qubits, output_len)
        
        print(f"✓ Quantum LSTM model initialized with device type: {device_type}")
    
    def forward(self, x):
        """Forward pass through hybrid quantum-classical network"""
        # Classical LSTM processing (same as classical model)
        lstm_out, _ = self.lstm(x)
        final_lstm_output = lstm_out[:, -1, :]  # Take last timestep
        
        # Prepare quantum input
        quantum_input = self.classical_to_quantum(final_lstm_output)
        
        # Quantum processing
        quantum_features = self.quantum_layer(quantum_input)
        
        # Final prediction
        output = self.quantum_to_output(quantum_features)
        
        return torch.sigmoid(output)




def train_model_pytorch(model, train_loader, val_loader, epochs=25, patience=5, model_save_path=None):
    """Train the quantum LSTM model following classical training pattern"""
    print("Starting training with PyTorch + Quantum...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower lr for quantum training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    
    if model_save_path is None:
        model_save_path = 'hybrid.pth'
    
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
    parser.add_argument("--model_save_path", default="hybrid.pth", 
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
    
    # Device configuration arguments
    parser.add_argument("--device_type", choices=["ideal", "noisy", "hardware"], 
                       default="ideal", help="Type of quantum device")
    parser.add_argument("--backend_name", help="IBM Quantum backend name (e.g., ibm_brisbane)")
    parser.add_argument("--shots", type=int, default=1024, help="Number of quantum shots")
    parser.add_argument("--use_noise_model", action="store_true", 
                       help="Use noise model from backend for realistic simulation")
    parser.add_argument("--use_real_hardware", action="store_true", 
                       help="Use real IBM Quantum hardware via qiskit.remote")
    parser.add_argument("--plot_history", action="store_true", help="Plot training history")
    
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, loc_train, X_val, y_val, loc_val = load_training_data(args.data_dir)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create quantum device manager
    device_manager = QuantumDeviceManager(
        backend_name=args.backend_name,
        shots=args.shots,
        use_noise_model=args.use_noise_model or args.device_type == "noisy",
        use_real_hardware=args.use_real_hardware or args.device_type == "hardware"
    )
    
    print(f"\nDevice Configuration:")
    print(f"  Device type: {args.device_type}")
    if args.backend_name:
        print(f"  Backend: {args.backend_name}")
    print(f"  Shots: {args.shots}")
    print(f"  Use noise model: {device_manager.use_noise_model}")
    print(f"  Use real hardware: {device_manager.use_real_hardware}")
    
    # Create model with device manager
    model = QuantumLSTMModel(
        n_features=args.n_features,
        n_lstm_units=args.n_lstm_units,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        num_lstm_layers=args.num_lstm_layers,
        output_len=args.output_len,
        device_type=args.device_type,
        device_manager=device_manager
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params}")
    print(f"Quantum device info: {device_manager.get_device_info()}")
    
    # Train model
    model, history = train_model_pytorch(
        model, train_loader, val_loader,
        epochs=args.epochs, patience=args.patience,
        model_save_path=args.model_save_path
    )
    
    # Save training history
    history_path = args.model_save_path.replace('.pth', '_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(dict(history), f)
    print(f"Training history saved to {history_path}")
    
    # Save model config with device information
    config = {
        'n_features': args.n_features,
        'n_lstm_units': args.n_lstm_units,
        'n_qubits': args.n_qubits,
        'n_layers': args.n_layers,
        'num_lstm_layers': args.num_lstm_layers,
        'output_len': args.output_len,
        'device_type': args.device_type,
        'backend_name': args.backend_name,
        'shots': args.shots,
        'use_noise_model': device_manager.use_noise_model,
        'use_real_hardware': device_manager.use_real_hardware,
        'device_info': device_manager.get_device_info(),
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
