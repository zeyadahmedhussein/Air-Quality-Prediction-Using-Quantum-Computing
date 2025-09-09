# Results Directory

This directory contains evaluation results from quantum model runs.

## Structure

- `physical/`: Results from runs on physical quantum backends
- `simulator/`: Results from runs on quantum simulators

## Output Files

Each subdirectory will contain:

- `predictions.csv`: Model predictions in required format
- `metrics.json`: Evaluation metrics including accuracy, precision, recall, F1, and support
- `confusion_matrix.png`: Confusion matrix visualization  
- `run_summary.json`: Detailed run information including job IDs, timing, and backend info

## File Formats

### predictions.csv
- `sequence_id`: Identifier for each input sequence
- `timestep`: Hour ahead (1-72)
- `prediction`: Binary prediction (0 or 1) 
- `probability`: Prediction probability (0.0-1.0)

### metrics.json
```json
{
  "accuracy": 0.85,
  "precision": 0.82, 
  "recall": 0.88,
  "f1": 0.85,
  "support": {"class_0": 1500, "class_1": 500}
}
```

### run_summary.json
Contains job execution details, backend information, timing data, and any error messages.
