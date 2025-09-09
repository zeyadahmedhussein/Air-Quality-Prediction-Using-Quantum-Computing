# NILE Competition - Hybrid Quantum-Classical LSTM

A hybrid quantum-classical machine learning solution for air quality prediction, combining the temporal modeling capabilities of classical LSTMs with quantum variational circuits for enhanced pattern recognition.

## Overview

This project implements a hybrid quantum-classical LSTM model for multi-step time series forecasting (72-hour air quality prediction). The model uses classical LSTM layers for temporal feature extraction followed by quantum variational circuits for enhanced pattern recognition and prediction.

## Key Features

- **Multi-step Time Series Forecasting**: Predicts air quality 72 hours into the future
- **Hybrid Architecture**: Combines classical LSTM with quantum variational layers
- **Backend Flexibility**: Supports both quantum simulators and IBM Quantum physical backends
- **Competition Compliant**: Meets all NILE Competition requirements
- **Error Mitigation**: Includes quantum error mitigation and calibration techniques
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Project Structure

```
├── src/                          # Source code
│   ├── preprocessing/            # Data preprocessing modules
│   │   └── data_preprocessing.py
│   ├── classical_model/          # Classical model training/evaluation
│   │   ├── train_classical.py
│   │   └── evaluate_classical.py
│   ├── quantum_model/            # Quantum model modules
│   │   ├── train_quantum.py
│   │   └── predict_quantum.py
│   └── run_evaluation.py         # Main evaluation pipeline
├── configs/                      # Configuration files
│   └── qiskit_backend_config.json
├── data/                         # Data directory
├── results/                      # Output results
│   ├── physical/                 # Physical backend results
│   └── simulator/                # Simulator results
├── evidence/                     # Evidence and documentation
│   ├── job_ids.csv
│   ├── logs/
│   └── screenshots/
├── MANIFEST.json                 # Submission manifest
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone or extract this project
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Data Preprocessing
```bash
python src/preprocessing/data_preprocessing.py \
  --mode train \
  --data_path data/training.parquet \
  --output_dir data/processed/
```

### 2. Model Training
```bash
# Classical LSTM
python src/classical_model/train_classical.py \
  --data_dir data/processed/ \
  --epochs 50 \
  --model_save_path models/classical_model.pth

# Quantum LSTM 
python src/quantum_model/train_quantum.py \
  --data_dir data/processed/ \
  --epochs 25 \
  --use_qiskit \
  --backend ibm_brisbane
```

### 3. Evaluation (Competition Format)
```bash
python src/run_evaluation.py \
  --dataset data/unseen.csv \
  --backend ibm_brisbane \
  --shots 2048 \
  --mode physical \
  --out results/
```

## Competition Requirements Compliance

### ✅ Backend Selection & Configuration
- Configurable backend selection via `configs/qiskit_backend_config.json`
- CLI support for backend switching
- Supports ≥127 qubit backends (ibm_brisbane default)
- Respects all quantum backend constraints

### ✅ Data I/O Specifications  
- Consumes local CSV: `data/unseen.csv`
- Outputs predictions: `results/physical/predictions.csv`
- Generates metrics: `results/physical/metrics.json` 
- Creates confusion matrix: `results/physical/confusion_matrix.png`

### ✅ One-Command Evaluation
- Main script: `src/run_evaluation.py`
- Orchestrates full pipeline: preprocessing → inference → evaluation
- Generates complete `run_summary.json` with job IDs and timing
- CLI example provided in MANIFEST.json

### ✅ Evidence Pack
- Machine-readable `MANIFEST.json`
- Job tracking: `evidence/job_ids.csv`
- Screenshot storage: `evidence/screenshots/`
- Execution logs: `evidence/logs/`

## Model Architecture

### Classical Component
- **LSTM Units**: 32 hidden units
- **Layers**: 1 LSTM layer
- **Input Features**: 9 environmental variables
- **Output**: 72-hour sequence predictions

### Quantum Component  
- **Qubits**: 4 qubits (expandable)
- **Layers**: 3 quantum variational layers
- **Gates**: RX, RY, RZ, CNOT
- **Ansatz**: Strongly Entangling Layers
- **Encoding**: Angle embedding

### Training Details
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross Entropy
- **Early Stopping**: Yes (patience=5)
- **Batch Size**: 256/512 (quantum/classical)

## Command Reference

### Backend Configuration
```bash
# Edit backend settings
vim configs/qiskit_backend_config.json

# Use different backend
python src/quantum_model/predict_quantum.py \
  --backend ibm_kyoto \
  --shots 4096 \
  ...
```

### Quantum Prediction
```bash
python src/quantum_model/predict_quantum.py \
  --model_path models/quantum_model.pth \
  --backend ibm_brisbane \
  --shots 2048 \
  --dataset data/unseen.csv \
  --out results/physical/
```

### Full Evaluation Pipeline
```bash
python src/run_evaluation.py \
  --dataset data/unseen.csv \
  --backend <PHYSICAL_BACKEND> \
  --shots 2048 \
  --mode physical \
  --out results/
```

## Performance Expectations

- **Accuracy**: >85%
- **F1 Score**: >80%  
- **Inference Time**: <300s for 1000 sequences
- **Quantum Jobs**: Compatible with IBM Quantum backends ≥127 qubits

## Troubleshooting

### Common Issues

1. **Backend Not Available**
   - Check IBM Quantum account access
   - Verify backend name and availability
   - Use simulator mode for testing

2. **Memory Issues**
   - Reduce batch size in training scripts
   - Use gradient checkpointing if available
   - Monitor system resources

3. **Quantum Jobs Failing**
   - Check queue times on IBM Quantum
   - Verify circuit depth is reasonable
   - Review shot count limits

### Debug Mode
```bash
# Enable verbose logging
export QISKIT_LOGGING_LEVEL=DEBUG
python src/run_evaluation.py --dataset data/unseen.csv ...
```

## Contributing

This codebase is structured for the NILE Competition. Key extension points:

- Add new quantum ansätze in `src/quantum_model/train_quantum.py`
- Extend preprocessing in `src/preprocessing/data_preprocessing.py`  
- Add evaluation metrics in `src/run_evaluation.py`

## License

Developed for the NILE Competition. See competition terms for usage rights.

## Team Information

- **Team**: NILE_TEAM_001
- **Model**: Hybrid Quantum-Classical LSTM
- **Framework**: Qiskit + PyTorch + PennyLane
- **Submission Date**: 2024-09-09

---

For detailed technical specifications, see `MANIFEST.json`.
