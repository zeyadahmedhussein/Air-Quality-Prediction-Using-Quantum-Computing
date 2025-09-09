# Data Directory

This directory contains datasets and preprocessing artifacts for the NILE Competition.

## Files Structure

### Raw Data
- `air_quality_data.parquet`: Original training dataset in Parquet format
- `unseen.csv`: Unseen dataset in CSV format for evaluation

### Processed Data (processed/ subdirectory)
- `preprocessing_state.pkl`: Saved preprocessing state from training phase
- `scaler.pkl`: Fitted scaler for feature normalization
- `X_train.npy`, `y_train.npy`, etc.: Training data splits
- `unseen_processed.pkl`: Processed unseen data for inference

## CSV Data Format

The unseen CSV dataset should contain the following columns:
- `time`: Timestamp (ISO format)
- `lat`, `lon`: Geographic coordinates (float)
- `PM25_MERRA2`, `DUCMASS`, `TOTANGSTR`, `DUFLUXV`, `SSFLUXV`, `DUFLUXU`, `BCCMASS`, `SSSMASS25`: Environmental feature columns (float)
- `class`: Target variable ("Good", "Moderate", etc. - optional for unseen data)

## Preprocessing Pipeline

### For Training Data (Parquet)
```bash
python src/preprocessing/data_preprocessing.py \
  --mode train \
  --data_path data/air_quality_data.parquet \
  --output_dir data/processed/
```

### For Unseen Data (CSV)
```bash
python src/preprocessing/data_preprocessing.py \
  --mode inference \
  --data_path data/unseen.csv \
  --preprocessing_state data/processed/preprocessing_state.pkl \
  --output_path data/processed/unseen_processed.pkl
```

## Data Transformations

1. **Geographic Binning**: Lat/lon coordinates are binned into a 20x20 grid
2. **Location Encoding**: Grid positions are encoded using LabelEncoder
3. **Time Features**: Year, month, day, hour extracted from timestamps
4. **Feature Scaling**: StandardScaler applied to all numerical features
5. **Sequence Creation**: Time series sequences of length 168 (input) + 72 (output) hours
6. **Target Encoding**: Binary classification (Good/Moderate = 1, others = 0)

## Directory Usage

The preprocessing module automatically handles both Parquet and CSV formats, using the saved preprocessing state to ensure consistency between training and inference transformations.
