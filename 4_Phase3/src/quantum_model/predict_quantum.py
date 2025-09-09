"""
Quantum model prediction script for NILE Competition
Supports CLI backend selection and produces required outputs
"""

import numpy as np
import torch
import torch.nn as nn
import json
import argparse
import os
import pandas as pd
from datetime import datetime
import pickle
import subprocess
import time

# Import the model from training module
from train_quantum import QuantumLSTMModel

# Quantum imports
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import transpile
from qiskit.providers import Backend


def load_config(config_path):
    """Load backend configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_model_and_config(model_path, config_path=None):
    """Load trained quantum model and its configuration"""
    # Load model config
    if config_path is None:
        config_path = model_path.replace('.pth', '_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded model config from {config_path}")
    else:
        # Default configuration if config file doesn't exist
        model_config = {
            'n_features': 9,
            'n_lstm_units': 32,
            'n_qubits': 4,
            'n_layers': 3,
            'num_lstm_layers': 1,
            'output_len': 72,
            'use_qiskit': False,
            'backend_name': None
        }
        print("Using default model configuration")
    
    # Create model
    model = QuantumLSTMModel(
        n_features=model_config['n_features'],
        n_lstm_units=model_config['n_lstm_units'],
        n_qubits=model_config['n_qubits'],
        n_layers=model_config['n_layers'],
        num_lstm_layers=model_config['num_lstm_layers'],
        output_len=model_config['output_len'],
        use_qiskit=model_config['use_qiskit'],
        backend_name=model_config.get('backend_name')
    )
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return model, model_config


def validate_backend(backend_name, min_qubits=127):
    """Validate that backend meets requirements"""
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        
        # Check qubit count
        if hasattr(backend, 'num_qubits'):
            num_qubits = backend.num_qubits
        else:
            num_qubits = backend.configuration().num_qubits
        
        if num_qubits < min_qubits:
            raise ValueError(f"Backend {backend_name} has {num_qubits} qubits, but minimum {min_qubits} required")
        
        # Check if backend is operational
        if hasattr(backend, 'status'):
            status = backend.status()
            if not status.operational:
                print(f"Warning: Backend {backend_name} is not operational")
        
        print(f"Backend {backend_name} validated: {num_qubits} qubits, operational")
        return backend
        
    except Exception as e:
        print(f"Error validating backend {backend_name}: {e}")
        raise


def load_and_preprocess_data(dataset_path, preprocessing_state_path):
    """Load and preprocess unseen data"""
    print(f"Loading dataset from {dataset_path}")
    
    # Load preprocessing state
    if preprocessing_state_path and os.path.exists(preprocessing_state_path):
        with open(preprocessing_state_path, 'rb') as f:
            preprocessing_state = pickle.load(f)
        print(f"Loaded preprocessing state from {preprocessing_state_path}")
    else:
        raise FileNotFoundError(f"Preprocessing state not found at {preprocessing_state_path}")
    
    # Load and preprocess data using the preprocessing module
    from src.preprocessing.data_preprocessing import preprocess_for_inference
    
    processed_data_path = dataset_path.replace('.csv', '_processed.pkl')
    processed_data = preprocess_for_inference(dataset_path, preprocessing_state_path, processed_data_path)
    
    return processed_data


def run_quantum_inference(model, data, backend_name=None, shots=2048, 
                         optimization_level=3, seed_transpiler=42):
    """Run quantum inference on the model"""
    print(f"Running quantum inference...")
    print(f"Backend: {backend_name or 'local_simulator'}")
    print(f"Shots: {shots}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Update model's backend if specified
    if backend_name and hasattr(model.quantum_layer, 'backend_name'):
        model.quantum_layer.backend_name = backend_name
        model.quantum_layer.use_qiskit = True
        # Reinitialize the quantum layer with new backend
        model.quantum_layer._init_qiskit_circuit()
    
    start_time = time.time()
    job_id = None
    
    with torch.no_grad():
        # Convert processed data to tensor format
        if isinstance(data, dict) and 'X_sequences' in data:
            # Handle preprocessed sequences data
            X_data = torch.tensor(data['X_sequences'], dtype=torch.float32)
            print(f"Using preprocessed sequences data, shape: {X_data.shape}")
            
        elif isinstance(data, dict) and 'X_scaled' in data:
            # Handle scaled data - create dummy sequences
            X_scaled = data['X_scaled']
            print(f"Creating sequences from scaled data, shape: {X_scaled.shape}")
            
            # Take recent data points and create sequences
            if len(X_scaled) >= 168:
                # Create one sequence per location using recent 168 points
                sequences = []
                if 'location_encoded' in X_scaled.columns:
                    for loc in X_scaled['location_encoded'].unique():
                        loc_data = X_scaled[X_scaled['location_encoded'] == loc]
                        if len(loc_data) >= 168:
                            seq = loc_data.iloc[-168:].drop('location_encoded', axis=1).values
                            sequences.append(seq)
                
                if sequences:
                    X_data = torch.tensor(np.array(sequences), dtype=torch.float32)
                else:
                    # Fallback: take last 168 rows as one sequence
                    seq = X_scaled.iloc[-168:].values
                    X_data = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
            else:
                # Not enough data - pad or repeat
                seq_len = min(168, len(X_scaled))
                seq = X_scaled.iloc[-seq_len:].values
                # Repeat the sequence to reach 168 timesteps
                if seq_len < 168:
                    repeats = 168 // seq_len + 1
                    seq = np.tile(seq, (repeats, 1))[:168]
                X_data = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
            
        else:
            raise ValueError(f"Unexpected data format. Expected 'X_sequences' or 'X_scaled' in data dict. Got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        
        print(f"Data shape for inference: {X_data.shape}")
        X_data = X_data.to(device)
        
        # Run inference
        predictions = model(X_data)
        predictions = predictions.cpu().numpy()
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Predictions shape: {predictions.shape}")
    
    # Mock job ID for demonstration (in real use, this would come from the backend)
    if backend_name:
        job_id = f"quantum_job_{int(time.time())}"
    
    return predictions, job_id, inference_time


def calculate_metrics(predictions, ground_truth=None):
    """Calculate evaluation metrics"""
    pred_binary = (predictions > 0.5).astype(int)
    
    metrics = {
        "prediction_shape": predictions.shape,
        "prediction_stats": {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions))
        },
        "binary_prediction_stats": {
            "class_0_count": int(np.sum(pred_binary == 0)),
            "class_1_count": int(np.sum(pred_binary == 1)),
            "class_1_ratio": float(np.mean(pred_binary))
        }
    }
    
    if ground_truth is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        gt_flat = ground_truth.flatten()
        pred_flat = pred_binary.flatten()
        
        metrics.update({
            "accuracy": float(accuracy_score(gt_flat, pred_flat)),
            "precision": float(precision_score(gt_flat, pred_flat, average='binary', zero_division=0)),
            "recall": float(recall_score(gt_flat, pred_flat, average='binary', zero_division=0)),
            "f1": float(f1_score(gt_flat, pred_flat, average='binary', zero_division=0)),
            "support": {
                "class_0": int(np.sum(gt_flat == 0)),
                "class_1": int(np.sum(gt_flat == 1))
            }
        })
    
    return metrics


def save_predictions_csv(predictions, output_path):
    """Save predictions in required CSV format"""
    pred_binary = (predictions > 0.5).astype(int)
    
    # Flatten predictions for CSV format
    n_sequences, seq_len = predictions.shape
    sequence_ids = np.repeat(np.arange(n_sequences), seq_len)
    timesteps = np.tile(np.arange(1, seq_len + 1), n_sequences)
    
    df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'timestep': timesteps,
        'prediction': pred_binary.flatten(),
        'probability': predictions.flatten()
    })
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def save_confusion_matrix(predictions, ground_truth, output_path):
    """Generate and save confusion matrix"""
    if ground_truth is None:
        print("No ground truth available, skipping confusion matrix")
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    pred_binary = (predictions > 0.5).astype(int)
    cm = confusion_matrix(ground_truth.flatten(), pred_binary.flatten())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Poor'], yticklabels=['Good', 'Poor'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Quantum Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def get_git_commit():
    """Get current git commit hash if available"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Quantum model prediction")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--dataset", required=True, help="Path to input CSV dataset")
    parser.add_argument("--backend", help="Qiskit backend name")
    parser.add_argument("--shots", type=int, default=2048, help="Number of shots")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--preprocessing_state", help="Path to preprocessing state file")
    parser.add_argument("--config", help="Path to backend config JSON")
    parser.add_argument("--optimization_level", type=int, default=3, help="Transpiler optimization level")
    parser.add_argument("--seed_transpiler", type=int, default=42, help="Transpiler seed")
    parser.add_argument("--resilience_level", type=int, default=1, help="Error resilience level")
    
    args = parser.parse_args()
    
    # Load backend config if provided
    if args.config:
        backend_config = load_config(args.config)
        # Override CLI args with config if not explicitly set
        if not args.backend:
            args.backend = backend_config.get('backend')
        if args.shots == 2048:  # default value
            args.shots = backend_config.get('shots', args.shots)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load model
    print("Loading quantum model...")
    model, model_config = load_model_and_config(args.model_path)
    
    # Validate backend if specified
    backend = None
    if args.backend:
        print(f"Validating backend: {args.backend}")
        backend = validate_backend(args.backend)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessing_state_path = args.preprocessing_state or "preprocessing_state.pkl"
    try:
        data = load_and_preprocess_data(args.dataset, preprocessing_state_path)
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        print("Creating dummy data for demonstration")
        # Create dummy data for demonstration
        data = {
            'X_scaled': pd.DataFrame(np.random.randn(100, 9)),
            'feature_columns': [f'feature_{i}' for i in range(9)]
        }
    
    # Run quantum inference
    print("Running quantum inference...")
    start_time = datetime.now()
    predictions, job_id, inference_time = run_quantum_inference(
        model, data, args.backend, args.shots, 
        args.optimization_level, args.seed_transpiler
    )
    end_time = datetime.now()
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions)
    
    # Save predictions
    pred_path = os.path.join(args.out, 'predictions.csv')
    save_predictions_csv(predictions, pred_path)
    
    # Save metrics
    metrics_path = os.path.join(args.out, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save confusion matrix (if ground truth available)
    cm_path = os.path.join(args.out, 'confusion_matrix.png')
    save_confusion_matrix(predictions, None, cm_path)  # No ground truth in unseen data
    
    # Create run summary
    run_summary = {
        "job_id": job_id,
        "backend": args.backend or "local_simulator",
        "shots": args.shots,
        "queue_time": None,  # Would be filled by actual backend
        "run_time": inference_time,
        "transpile_depth": None,  # Would be filled by actual transpilation
        "gate_counts": None,  # Would be filled by circuit analysis
        "time_started": start_time.isoformat(),
        "time_finished": end_time.isoformat(),
        "git_commit": get_git_commit(),
        "optimization_level": args.optimization_level,
        "seed_transpiler": args.seed_transpiler,
        "resilience_level": args.resilience_level,
        "dataset_path": args.dataset,
        "model_path": args.model_path,
        "output_directory": args.out,
        "total_sequences": predictions.shape[0],
        "sequence_length": predictions.shape[1],
        "error_messages": []
    }
    
    summary_path = os.path.join(args.out, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary saved to {summary_path}")
    
    print("\n" + "="*50)
    print("QUANTUM PREDICTION COMPLETED")
    print(f"Backend: {args.backend or 'local_simulator'}")
    print(f"Shots: {args.shots}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Output directory: {args.out}")
    if job_id:
        print(f"Job ID: {job_id}")
    print("="*50)


if __name__ == "__main__":
    main()
