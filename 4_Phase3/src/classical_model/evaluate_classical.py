"""
Classical LSTM model evaluation module for NILE Competition
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

# Import the model class from training module
from train_classical import ClassicalLSTMModel


def load_model_and_data(model_path, data_dir):
    """Load trained model and test data"""
    print(f"Loading test data from {data_dir}")
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    loc_test = np.load(os.path.join(data_dir, 'loc_test.npy'))
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Create and load model
    model = ClassicalLSTMModel(n_features=9, n_lstm_units=128, num_layers=4, output_len=72)
    
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
    
    return model, X_test, y_test, loc_test, device


def evaluate_model_comprehensive(model, test_loader, device, location_indices_test, output_seq_len=72):
    """
    Comprehensive model evaluation with all required metrics
    """
    model.eval()
    all_preds, all_labels = [], []
    
    print("Evaluating model...")
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} batches")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    preds_flat, labels_flat = all_preds.flatten(), all_labels.flatten()
    
    # Overall metrics
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat, average="binary", zero_division=0)
    recall = recall_score(labels_flat, preds_flat, average="binary", zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, average="binary", zero_division=0)
    conf_matrix = confusion_matrix(labels_flat, preds_flat)
    
    # Support per class (number of samples per class)
    support_class_0 = np.sum(labels_flat == 0)
    support_class_1 = np.sum(labels_flat == 1)
    
    print(f"\nOverall Metrics:")
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


def plot_confusion_matrix(confusion_matrix, save_path, title="Confusion Matrix"):
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
            y_pred = model(X_sample).cpu().numpy()[0]
            y_pred = (y_pred > 0.5).astype(int)
            
            plt.subplot(num_examples, 1, i+1)
            hours = np.arange(1, 72 + 1)
            plt.plot(hours, y_true, 'bo-', label='Actual', markersize=3)
            plt.plot(hours, y_pred, 'ro--', label='Predicted', markersize=3)
            plt.title(f'Example {i+1}: Air Quality Prediction (Next 72 Hours)')
            plt.xlabel('Hours Ahead')
            plt.ylabel('Air Quality Class')
            plt.yticks([0, 1], ['Good', 'Poor'])
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Example predictions saved to {save_path}")


def save_predictions_csv(predictions, ground_truth, location_indices, save_path):
    """Save predictions in CSV format as required by competition"""
    # Flatten predictions for CSV format
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
    parser = argparse.ArgumentParser(description="Evaluate classical LSTM model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_dir", required=True, help="Directory containing test data")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--save_predictions", action="store_true", help="Save detailed predictions")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    model, X_test, y_test, loc_test, device = load_model_and_data(args.model_path, args.data_dir)
    
    # Create data loader
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    metrics = evaluate_model_comprehensive(model, test_loader, device, loc_test)
    
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
        "device_used": str(device)
    }
    
    summary_path = os.path.join(args.output_dir, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary saved to {summary_path}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Overall F1 Score: {metrics['overall']['f1']:.4f}")
    print(f"Results saved in: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
