import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import gc
import pickle
import joblib
import argparse
import os

def transform_data(file_path, file_format='parquet', lat_bins=None, lon_bins=None, le_location=None):
    """Transform raw data into features for model training/testing"""
    # Read data based on file format
    if file_format == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_format == 'csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'parquet' or 'csv'.")

    # Use provided bins or compute from data
    if lat_bins is None or lon_bins is None:
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        grid_size = 20
        lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
        lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)

    # Create grid positions
    lat_positions = pd.cut(df['lat'], bins=lat_bins, labels=False, include_lowest=True)
    lon_positions = pd.cut(df['lon'], bins=lon_bins, labels=False, include_lowest=True)
    df['location'] = lat_positions * 20 + lon_positions
    df['location'] = df['location'].fillna(0).astype(int)

    # Process class and time features
    df['class'] = df['class'].isin(['Good', 'Moderate']).astype(int)
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour

    # Select columns
    list_of_columns = ['class', 'PM25_MERRA2', 'DUCMASS', 'TOTANGSTR', 'DUFLUXV', 'SSFLUXV', 'DUFLUXU', 'BCCMASS', 'SSSMASS25', 'location']
    selected_columns = list_of_columns + ['year', 'month', 'day', 'hour']
    df = df[selected_columns]

    # Encode location
    if le_location is None:
        le_location = LabelEncoder()
        df['location_encoded'] = le_location.fit_transform(df['location'])
    else:
        # Handle unseen locations in test data
        df['location_encoded'] = df['location'].map(
            lambda x: le_location.transform([x])[0] if x in le_location.classes_ else -1
        )

    df = df.sort_values(['location_encoded', 'year', 'month', 'day', 'hour'])

    # Define feature columns
    feature_columns = [col for col in df.columns if col not in [
        'class', 'location', 'year', 'month', 'day', 'hour', 'location_encoded'
    ]]
    df.drop(columns='location', inplace=True)
    feature_columns.append('location_encoded')

    return df, feature_columns, lat_bins, lon_bins, le_location

def create_sequences_memory_efficient(df, feature_columns, scaler=None,
                                    input_len=168, output_len=72, stride=24):
    """Create sequences for multi-step forecasting"""
    print(f"Creating sequences with input length={input_len}, output length={output_len}...")

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])
    else:
        X_scaled = scaler.transform(df[feature_columns])

    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)
    X_sequences, y_sequences, location_indices = [], [], []
    unique_locations = df['location_encoded'].unique()

    for i, loc in enumerate(unique_locations):
        loc_df = df[df['location_encoded'] == loc]
        loc_X = X_scaled.loc[loc_df.index]
        loc_y = loc_df['class'].values

        max_start_idx = len(loc_df) - input_len - output_len

        for j in range(0, max_start_idx, stride):
            X_seq = loc_X.iloc[j : j + input_len].values
            y_target = loc_y[j + input_len : j + input_len + output_len]

            X_sequences.append(X_seq)
            y_sequences.append(y_target)
            location_indices.append(loc)

        if (i+1) % 100 == 0:
            print(f"Processed location {i+1}/{len(unique_locations)}")

    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    location_indices = np.array(location_indices)

    print(f"Total sequences: {X_sequences.shape[0]}")
    print(f"Input sequence shape: {X_sequences.shape}")
    print(f"Output sequence shape: {y_sequences.shape}")

    return X_sequences, y_sequences, location_indices, scaler

def preprocess_for_inference(data_path, preprocessing_state_path, output_path=None):
    """Preprocess data for inference using saved preprocessing state"""
    # Load preprocessing state
    with open(preprocessing_state_path, 'rb') as f:
        state = pickle.load(f)
    
    feature_columns = state['feature_columns']
    lat_bins = state['lat_bins']
    lon_bins = state['lon_bins']
    le_location = state['le_location']
    scaler = state['scaler']
    
    # Detect file format
    file_format = 'csv' if data_path.endswith('.csv') else 'parquet'
    
    # Transform data using existing preprocessing state
    df, _, _, _, _ = transform_data(
        data_path, file_format=file_format, 
        lat_bins=lat_bins, lon_bins=lon_bins, le_location=le_location
    )
    
    # Create sequences using existing scaler
    X_sequences, y_sequences, location_indices, _ = create_sequences_memory_efficient(
        df, feature_columns, scaler=scaler, input_len=168, output_len=72, stride=24
    )
    
    if output_path:
        # Save processed data
        data_dict = {
            'X_sequences': X_sequences,
            'y_sequences': y_sequences,
            'location_indices': location_indices
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    return X_sequences, y_sequences, location_indices

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess air quality data.')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset file (parquet or csv format)')
    parser.add_argument('--file_format', type=str, choices=['parquet', 'csv'], default='parquet',
                        help='File format of the dataset: "parquet" or "csv" (default: parquet)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Processing mode: "train" to split data, "test" to transform without splitting')
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file not found: {args.data_path}")

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Transform data
    df, feature_columns, lat_bins, lon_bins, le_location = transform_data(
        args.data_path, file_format=args.file_format
    )

    # Create sequences
    X_sequences, y_sequences, location_indices, scaler = create_sequences_memory_efficient(
        df, feature_columns, input_len=168, output_len=72, stride=24
    )

    if args.mode == 'train':
        # Split data
        X_train, X_temp, y_train, y_temp, loc_train, loc_temp = train_test_split(
            X_sequences, y_sequences, location_indices,
            test_size=0.4, random_state=42, stratify=location_indices
        )

        X_test, X_val, y_test, y_val, loc_test, loc_val = train_test_split(
            X_temp, y_temp, loc_temp,
            test_size=0.5, random_state=42, stratify=loc_temp
        )

        del X_sequences, y_sequences
        gc.collect()

        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Validation target shape: {y_val.shape}")
        print(f"Test target shape: {y_test.shape}")

        # Create output directories
        os.makedirs('data/Data_files', exist_ok=True)
        
        # Save split data
        np.save('data/Data_files/X_train.npy', X_train)
        np.save('data/Data_files/y_train.npy', y_train)
        np.save('data/Data_files/loc_train.npy', loc_train)
        np.save('data/Data_files/X_test.npy', X_test)
        np.save('data/Data_files/y_test.npy', y_test)
        np.save('data/Data_files/X_val.npy', X_val)
        np.save('data/Data_files/y_val.npy', y_val)
        np.save('data/Data_files/loc_val.npy', loc_val)
        np.save('data/Data_files/loc_test.npy', loc_test)
        
        # Save preprocessing state for inference use
        preprocessing_state = {
            'feature_columns': feature_columns,
            'lat_bins': lat_bins,
            'lon_bins': lon_bins,
            'le_location': le_location,
            'scaler': scaler
        }
        
        os.makedirs('data', exist_ok=True)
        with open('data/preprocessing_state.pkl', 'wb') as f:
            pickle.dump(preprocessing_state, f)
        print("Preprocessing state saved to data/preprocessing_state.pkl")
        
        print("Saved 9 files for train/test/val splits")
    else:  # test mode
        print(f"Transformed data shape: {X_sequences.shape}")
        print(f"Target shape: {y_sequences.shape}")
        
        # Create output directories
        os.makedirs('data/Data_files', exist_ok=True)
        
        # Save transformed data without splitting
        np.save('data/Data_files/X_test.npy', X_sequences)
        np.save('data/Data_files/y_test.npy', y_sequences)
        np.save('data/Data_files/loc_test.npy', location_indices)
        
        print("Saved 3 files for transformed data")

if __name__ == "__main__":
    main()