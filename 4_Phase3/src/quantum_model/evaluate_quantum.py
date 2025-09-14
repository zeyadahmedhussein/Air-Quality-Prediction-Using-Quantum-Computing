"""
Quantum LSTM Evaluation Module
Following classical evaluation pattern with quantum device support
Supports ideal simulation, noisy simulation, and real hardware evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import seaborn as sns
import argparse
import os
import json
import pandas as pd
from datetime import datetime
import pickle

# Import the model and device classes
from .train_quantum import QuantumLSTMModel
from .quantum_devices import QuantumDeviceManager

def load_model_and_data(model_path, data_dir, config_path=None):
    """Load trained quantum model and test data"""
    print(f"Loading test data from {data_dir}")
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    loc_test = np.load(os.path.join(data_dir, 'loc_test.npy'))
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Load model configuration
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
            'device_type': 'ideal',
            'backend_name': None,
            'shots': 1024,
            'use_noise_model': False,
            'use_real_hardware': False
        }
        print("Using default model configuration")
    
    # Create device manager from config
    device_manager = QuantumDeviceManager(
        backend_name=model_config.get('backend_name'),
        shots=model_config.get('shots', 1024),
        use_noise_model=model_config.get('use_noise_model', False),
        use_real_hardware=model_config.get('use_real_hardware', False)
    )
    
    # Create and load model
    model = QuantumLSTMModel(
        n_features=model_config['n_features'],
        n_lstm_units=model_config['n_lstm_units'],
        n_qubits=model_config['n_qubits'],
        n_layers=model_config['n_layers'],
        num_lstm_layers=model_config['num_lstm_layers'],
        output_len=model_config['output_len'],
        device_type=model_config.get('device_type', 'ideal'),
        device_manager=device_manager
    )
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")
    
    return model, X_test, y_test, loc_test, device, model_config


def evaluate_model_comprehensive(model, test_loader, device, location_indices_test, output_seq_len=72):
    """
    Comprehensive quantum model evaluation with all required metrics
    Same structure as classical evaluation
    """
    model.eval()
    all_preds, all_labels = [], []
    
    print("Evaluating quantum model...")
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            try:
                outputs = model(X_batch)
                preds = (outputs > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1} batches")
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Skip this batch and continue
                continue
    
    if not all_preds:
        raise RuntimeError("No batches were processed successfully")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    preds_flat, labels_flat = all_preds.flatten(), all_labels.flatten()
    
    # Overall metrics
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, average="binary", zero_division=0)
    recall = recall_score(labels_flat, preds_flat, average="binary", zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, average="binary", zero_division=0)
    conf_matrix = confusion_matrix(labels_flat, preds_flat)
    
    # Support per class
    support_class_0 = np.sum(labels_flat == 0)
    support_class_1 = np.sum(labels_flat == 1)
    
    print(f"\nOverall Quantum Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Support Class 0: {support_class_0}")
    print(f"Support Class 1: {support_class_1}")
    
    # Per-location metrics
    unique_locations = np.unique(location_indices_test)
    location_metrics = {}
    for loc in unique_locations:
        idx = (location_indices_test == loc)
        preds_loc, labels_loc = all_preds[idx].flatten(), all_labels[idx].flatten()
        
        if len(labels_loc) > 0:
            location_metrics[int(loc)] = {
                "accuracy": accuracy_score(labels_loc, preds_loc),
                "precision": precision_score(labels_loc, preds_loc, average="binary", zero_division=0),
                "recall": recall_score(labels_loc, preds_loc, average="binary", zero_division=0),
                "f1": f1_score(labels_loc, preds_loc, average="binary", zero_division=0),
                "support": len(labels_loc)
            }
    
    # Per-hour metrics
    hour_metrics = []
    for hour in range(output_seq_len):
        preds_hour, labels_hour = all_preds[:, hour], all_labels[:, hour]
        hour_metrics.append({
            "hour": hour + 1,
            "accuracy": accuracy_score(labels_hour, preds_hour),
            "precision": precision_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "recall": recall_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "f1": f1_score(labels_hour, preds_hour, average="binary", zero_division=0),
            "support": len(labels_hour)
        })
    
    return {
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": {"class_0": int(support_class_0), "class_1": int(support_class_1)},
            "confusion_matrix": conf_matrix.tolist()
        },
        "per_location": location_metrics,
        "per_hour": hour_metrics,
        "predictions": all_preds.tolist(),
        "ground_truth": all_labels.tolist()
    }


def plot_confusion_matrix(confusion_matrix, save_path, title="Confusion Matrix - Quantum Model"):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Good", "Poor"], yticklabels=["Good", "Poor"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_example_predictions(model, X_test, y_test, save_path, num_examples=3):
    """Plot example predictions vs actual values"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        sample_indices = np.random.choice(len(X_test), num_examples, replace=False)
        plt.figure(figsize=(15, 5 * num_examples))
        
        for i, idx in enumerate(sample_indices):
            X_sample = torch.tensor(X_test[idx:idx+1]).to(device)
            y_true = y_test[idx]
            
            try:
                y_pred = model(X_sample).cpu().numpy()[0]
                y_pred = (y_pred > 0.5).astype(int)
                
                plt.subplot(num_examples, 1, i+1)
                hours = np.arange(1, 72 + 1)
                plt.plot(hours, y_true, 'bo-', label='Actual', markersize=3)
                plt.plot(hours, y_pred, 'ro--', label='Predicted (Quantum)', markersize=3)
                plt.title(f'Example {i+1}: Quantum Air Quality Prediction (Next 72 Hours)')
                plt.xlabel('Hours Ahead')
                plt.ylabel('Air Quality Class')
                plt.yticks([0, 1], ['Good', 'Poor'])
                plt.legend()
                plt.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Error plotting example {i+1}: {e}")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Example predictions saved to {save_path}")


def save_predictions_csv(predictions, ground_truth, location_indices, save_path):
    """Save predictions in CSV format as required by competition"""
    pred_flat = np.array(predictions).reshape(-1)
    gt_flat = np.array(ground_truth).reshape(-1)
    
    # Create sequence and timestep indices
    n_sequences, seq_len = np.array(predictions).shape
    sequence_ids = np.repeat(np.arange(n_sequences), seq_len)
    timesteps = np.tile(np.arange(1, seq_len + 1), n_sequences)
    location_ids = np.repeat(location_indices, seq_len)
    
    df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'timestep': timesteps,
        'location_id': location_ids,
        'prediction': pred_flat.astype(int),
        'ground_truth': gt_flat.astype(int)
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quantum LSTM model")
    parser.add_argument("--model_path", required=True, help="Path to trained quantum model")
    parser.add_argument("--data_dir", required=True, help="Directory containing test data")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--config_path", help="Path to model configuration file")
    parser.add_argument("--save_predictions", action="store_true", help="Save detailed predictions")
    
    # Device override arguments (optional - will use model config by default)
    parser.add_argument("--device_type", choices=["ideal", "noisy", "hardware"], 
                       help="Override device type from model config")
    parser.add_argument("--backend_name", help="Override backend name from model config")
    parser.add_argument("--shots", type=int, help="Override shots from model config")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    model, X_test, y_test, loc_test, device, model_config = load_model_and_data(
        args.model_path, args.data_dir, args.config_path
    )
    
    # Override device configuration if specified
    if args.device_type or args.backend_name or args.shots:
        print("\nOverriding device configuration...")
        device_manager = QuantumDeviceManager(
            backend_name=args.backend_name or model_config.get('backend_name'),
            shots=args.shots or model_config.get('shots', 1024),
            use_noise_model=args.device_type == "noisy" if args.device_type else model_config.get('use_noise_model', False),
            use_real_hardware=args.device_type == "hardware" if args.device_type else model_config.get('use_real_hardware', False)
        )
        
        # Update model with new device
        model.device_manager = device_manager
        model.quantum_device = device_manager.create_device(model.n_qubits, args.device_type or model_config.get('device_type', 'ideal'))
        from .quantum_devices import create_quantum_circuit
        model.quantum_circuit = create_quantum_circuit(model.n_qubits, model.n_layers, model.quantum_device)
        weight_shapes = {"weights": (model.n_layers, model.n_qubits, 3)}
        model.quantum_layer = qml.qnn.TorchLayer(model.quantum_circuit, weight_shapes)
    
    print(f"\nEvaluating with configuration:")
    print(f"  Device type: {model_config.get('device_type', 'ideal')}")
    print(f"  Backend: {model_config.get('backend_name', 'None')}")
    print(f"  Shots: {model_config.get('shots', 1024)}")
    
    # Create data loader
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    try:
        metrics = evaluate_model_comprehensive(model, test_loader, device, loc_test)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This might be due to quantum device issues. Check your configuration.")
        return 1
    
    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Plot and save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(np.array(metrics['overall']['confusion_matrix']), cm_path)
    
    # Plot example predictions
    examples_path = os.path.join(args.output_dir, 'example_predictions.png')
    plot_example_predictions(model, X_test, y_test, examples_path)
    
    # Save predictions if requested
    if args.save_predictions:
        pred_path = os.path.join(args.output_dir, 'predictions.csv')
        save_predictions_csv(metrics['predictions'], metrics['ground_truth'], loc_test, pred_path)
    
    # Save run summary
    run_summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "timestamp": datetime.now().isoformat(),
        "test_samples": len(X_test),
        "overall_accuracy": metrics['overall']['accuracy'],
        "overall_f1": metrics['overall']['f1'],
        "device_used": str(device),
        "quantum_device_config": model_config.get('device_info', {}),
        "model_type": "quantum_lstm"
    }
    
    summary_path = os.path.join(args.output_dir, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary saved to {summary_path}")
    
    print("\n" + "="*60)
    print("QUANTUM EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Overall F1 Score: {metrics['overall']['f1']:.4f}")
    print(f"Quantum Device: {model_config.get('device_type', 'ideal')}")
    if model_config.get('backend_name'):
        print(f"Backend: {model_config['backend_name']}")
    print(f"Results saved in: {args.output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
