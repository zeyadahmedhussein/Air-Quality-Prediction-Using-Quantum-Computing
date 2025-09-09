"""
Main evaluation script for NILE Competition
Orchestrates the full evaluation pipeline from data loading to result generation
"""

import argparse
import os
import sys
import json
import subprocess
import time
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path


def load_backend_config(config_path):
    """Load backend configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def validate_inputs(dataset_path, mode, backend=None):
    """Validate input parameters"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if mode not in ['physical', 'simulator']:
        raise ValueError(f"Mode must be 'physical' or 'simulator', got: {mode}")
    
    if mode == 'physical' and not backend:
        raise ValueError("Backend must be specified for physical mode")


def setup_output_directory(output_dir, mode):
    """Setup output directory structure"""
    output_path = os.path.join(output_dir, mode)
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['logs', 'temp']:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
    
    return output_path


def run_preprocessing(dataset_path, output_dir, preprocessing_state_path):
    """Run data preprocessing if needed"""
    print("Step 1: Data Preprocessing")
    
    processed_data_path = os.path.join(output_dir, 'temp', 'processed_data.pkl')
    
    # Check if preprocessing state exists
    if not os.path.exists(preprocessing_state_path):
        print(f"Warning: Preprocessing state not found at {preprocessing_state_path}")
        print("This may indicate that training preprocessing hasn't been run.")
        return None
    
    try:
        # Import and use preprocessing module
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.preprocessing.data_preprocessing import preprocess_for_inference
        
        print(f"Preprocessing data from {dataset_path}")
        processed_data = preprocess_for_inference(dataset_path, preprocessing_state_path, processed_data_path)
        print(f"Preprocessed data saved to {processed_data_path}")
        return processed_data_path
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Continuing with raw data - model may need to handle this")
        return None


def run_quantum_prediction(model_path, dataset_path, backend, shots, output_dir, 
                         preprocessing_state_path, config_path):
    """Run quantum model prediction"""
    print("Step 2: Quantum Model Prediction")
    
    # Construct command for quantum prediction
    cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), 'quantum_model', 'predict_quantum.py'),
        '--model_path', model_path,
        '--dataset', dataset_path,
        '--out', output_dir,
        '--shots', str(shots)
    ]
    
    if backend:
        cmd.extend(['--backend', backend])
    
    if preprocessing_state_path and os.path.exists(preprocessing_state_path):
        cmd.extend(['--preprocessing_state', preprocessing_state_path])
    
    if config_path:
        cmd.extend(['--config', config_path])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the prediction
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"Error in quantum prediction: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        
        print("Quantum prediction completed successfully")
        print(result.stdout)
        return True
        
    except subprocess.TimeoutExpired:
        print("Quantum prediction timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Error running quantum prediction: {e}")
        return False


def evaluate_against_ground_truth(predictions_path, ground_truth_path, output_dir):
    """Evaluate predictions against ground truth if available"""
    print("Step 3: Evaluation Against Ground Truth")
    
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        print("No ground truth provided - skipping evaluation")
        return None
    
    try:
        # Load predictions and ground truth
        predictions_df = pd.read_csv(predictions_path)
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Extract predictions and ground truth arrays
        predictions = predictions_df['prediction'].values
        ground_truth = ground_truth_df['ground_truth'].values if 'ground_truth' in ground_truth_df.columns else None
        
        if ground_truth is None:
            print("Ground truth column not found in provided file")
            return None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average='binary', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='binary', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='binary', zero_division=0)
        cm = confusion_matrix(ground_truth, predictions)
        
        # Calculate support per class
        unique, counts = np.unique(ground_truth, return_counts=True)
        support = dict(zip(unique.astype(int), counts.astype(int)))
        
        evaluation_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': support,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(predictions)
        }
        
        # Save evaluation results
        eval_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Evaluation results saved to {eval_path}")
        return evaluation_results
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None


def collect_job_information(run_summary_path, output_dir):
    """Collect job information for evidence"""
    print("Step 4: Collecting Job Information")
    
    job_info = {
        'timestamp': datetime.now().isoformat(),
        'backend': 'local_simulator',
        'job_id': 'N/A',
        'mode': 'simulator',
        'shots': 1024,
        'notes': 'Local evaluation run'
    }
    
    # Try to extract information from run summary
    if os.path.exists(run_summary_path):
        try:
            with open(run_summary_path, 'r') as f:
                run_summary = json.load(f)
            
            job_info.update({
                'backend': run_summary.get('backend', 'local_simulator'),
                'job_id': run_summary.get('job_id', 'N/A'),
                'shots': run_summary.get('shots', 1024),
                'mode': 'physical' if run_summary.get('backend') != 'local_simulator' else 'simulator'
            })
            
        except Exception as e:
            print(f"Error reading run summary: {e}")
    
    return job_info


def update_job_ids_csv(job_info, evidence_dir):
    """Update the job IDs CSV file"""
    job_ids_path = os.path.join(evidence_dir, 'job_ids.csv')
    
    # Create DataFrame with job information
    job_df = pd.DataFrame([job_info])
    
    # Append to existing file or create new one
    if os.path.exists(job_ids_path):
        existing_df = pd.read_csv(job_ids_path)
        combined_df = pd.concat([existing_df, job_df], ignore_index=True)
    else:
        # Create header if file doesn't exist
        combined_df = job_df
    
    combined_df.to_csv(job_ids_path, index=False)
    print(f"Job information updated in {job_ids_path}")


def create_final_summary(output_dir, mode, job_info, evaluation_results=None):
    """Create final evaluation summary"""
    print("Step 5: Creating Final Summary")
    
    summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'mode': mode,
        'output_directory': output_dir,
        'job_information': job_info,
        'evaluation_results': evaluation_results,
        'artifacts_created': [
            'predictions.csv',
            'metrics.json', 
            'run_summary.json'
        ]
    }
    
    # Add confusion matrix if ground truth evaluation was performed
    if evaluation_results:
        summary['artifacts_created'].append('evaluation_metrics.json')
    
    # Check for actual artifacts
    summary['artifacts_status'] = {}
    for artifact in summary['artifacts_created']:
        artifact_path = os.path.join(output_dir, artifact)
        summary['artifacts_status'][artifact] = os.path.exists(artifact_path)
    
    # Save final summary
    final_summary_path = os.path.join(output_dir, 'final_evaluation_summary.json')
    with open(final_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Final evaluation summary saved to {final_summary_path}")
    return summary


def print_evaluation_report(summary, mode, backend=None):
    """Print final evaluation report"""
    print("\n" + "="*60)
    print("NILE COMPETITION EVALUATION COMPLETED")
    print("="*60)
    print(f"Mode: {mode}")
    print(f"Backend: {backend or 'local_simulator'}")
    print(f"Output Directory: {summary['output_directory']}")
    print(f"Timestamp: {summary['evaluation_timestamp']}")
    
    if summary['job_information']['job_id'] != 'N/A':
        print(f"Job ID: {summary['job_information']['job_id']}")
    
    print(f"Shots: {summary['job_information']['shots']}")
    
    print("\nArtifacts Created:")
    for artifact, exists in summary['artifacts_status'].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {artifact}")
    
    if summary['evaluation_results']:
        eval_results = summary['evaluation_results']
        print("\nEvaluation Metrics:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall: {eval_results['recall']:.4f}")
        print(f"  F1 Score: {eval_results['f1']:.4f}")
        print(f"  Total Samples: {eval_results['total_samples']}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="NILE Competition Evaluation Pipeline")
    parser.add_argument("--dataset", required=True, help="Path to unseen dataset CSV")
    parser.add_argument("--backend", help="Qiskit backend name for physical mode")
    parser.add_argument("--shots", type=int, default=2048, help="Number of shots")
    parser.add_argument("--mode", choices=['physical', 'simulator'], required=True,
                       help="Evaluation mode")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model_path", help="Path to trained quantum model")
    parser.add_argument("--config", help="Path to backend configuration JSON")
    parser.add_argument("--preprocessing_state", help="Path to preprocessing state file")
    parser.add_argument("--ground_truth", help="Path to ground truth CSV for evaluation")
    
    args = parser.parse_args()
    
    # Validate inputs
    try:
        validate_inputs(args.dataset, args.mode, args.backend)
    except (FileNotFoundError, ValueError) as e:
        print(f"Input validation error: {e}")
        return 1
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_backend_config(args.config)
        if not args.backend and config.get('backend'):
            args.backend = config['backend']
        if args.shots == 2048 and config.get('shots'):  # Using default
            args.shots = config['shots']
    
    # Setup output directory
    try:
        output_dir = setup_output_directory(args.out, args.mode)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error setting up output directory: {e}")
        return 1
    
    # Default model path if not provided
    if not args.model_path:
        args.model_path = "models/best_quantum_lstm_model.pth"
        print(f"Using default model path: {args.model_path}")
    
    # Default preprocessing state path if not provided
    if not args.preprocessing_state:
        args.preprocessing_state = "data/preprocessing_state.pkl"
        print(f"Using default preprocessing state path: {args.preprocessing_state}")
    
    # Step 1: Data Preprocessing
    processed_data_path = run_preprocessing(args.dataset, output_dir, args.preprocessing_state)
    
    # Step 2: Run Quantum Prediction
    prediction_success = run_quantum_prediction(
        args.model_path, args.dataset, args.backend, args.shots, 
        output_dir, args.preprocessing_state, args.config
    )
    
    if not prediction_success:
        print("Prediction failed - cannot continue evaluation")
        return 1
    
    # Step 3: Evaluate against ground truth if provided
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    evaluation_results = evaluate_against_ground_truth(
        predictions_path, args.ground_truth, output_dir
    )
    
    # Step 4: Collect job information
    run_summary_path = os.path.join(output_dir, 'run_summary.json')
    job_info = collect_job_information(run_summary_path, output_dir)
    
    # Step 5: Update evidence collection
    evidence_dir = os.path.dirname(os.path.dirname(output_dir))  # Go up two levels to find evidence dir
    evidence_dir = os.path.join(evidence_dir, 'evidence')
    if os.path.exists(evidence_dir):
        update_job_ids_csv(job_info, evidence_dir)
    else:
        print(f"Evidence directory not found at {evidence_dir}")
    
    # Step 6: Create final summary
    summary = create_final_summary(output_dir, args.mode, job_info, evaluation_results)
    
    # Print final report
    print_evaluation_report(summary, args.mode, args.backend)
    
    print("\\nEvaluation pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
