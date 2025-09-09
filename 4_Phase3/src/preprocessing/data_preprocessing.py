"""
Data preprocessing module for NILE Competition
Handles data transformation, feature engineering, and sequence creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import gc
import pickle
import joblib
import argparse
import os


def transform_data(file_path, lat_bins=None, lon_bins=None, le_location=None):
    """Transform raw data into features for model training/testing"""
    # Handle both parquet and CSV files
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)
    
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


def preprocess_for_training(data_path, output_dir, input_len=168, output_len=72, stride=24):
    """Full preprocessing pipeline for training data"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}")
    df, feature_columns, lat_bins, lon_bins, le_location = transform_data(data_path)
    
    print("Creating sequences...")
    X_sequences, y_sequences, location_indices, scaler = create_sequences_memory_efficient(
        df, feature_columns, input_len=input_len, output_len=output_len, stride=stride
    )
    
    # Split data
    print("Splitting data...")
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
    
    # Save data
    print("Saving preprocessed data...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'loc_train.npy'), loc_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'loc_val.npy'), loc_val)
    np.save(os.path.join(output_dir, 'loc_test.npy'), loc_test)
    np.save(os.path.join(output_dir, 'location_indices.npy'), location_indices)
    
    # Save preprocessing state
    preprocessing_state = {
        'lat_bins': lat_bins,
        'lon_bins': lon_bins,
        'le_location': le_location,
        'feature_columns': feature_columns,
        'scaler': scaler
    }
    
    with open(os.path.join(output_dir, 'preprocessing_state.pkl'), 'wb') as f:
        pickle.dump(preprocessing_state, f)
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    print(f"Preprocessing complete. Data saved to {output_dir}")
    return preprocessing_state


def preprocess_for_inference(data_path, preprocessing_state_path, output_path, 
                            input_len=168, output_len=72):
    """Preprocess unseen CSV data for inference using saved preprocessing state"""
    print(f"Loading preprocessing state from {preprocessing_state_path}")
    with open(preprocessing_state_path, 'rb') as f:
        state = pickle.load(f)
    
    print(f"Processing data from {data_path}")
    df, _, _, _, _ = transform_data(
        data_path, 
        lat_bins=state['lat_bins'],
        lon_bins=state['lon_bins'],
        le_location=state['le_location']
    )
    
    # Scale features using saved scaler
    X_scaled = state['scaler'].transform(df[state['feature_columns']])
    X_scaled_df = pd.DataFrame(X_scaled, columns=state['feature_columns'], index=df.index)
    
    # Create sequences for inference
    print("Creating sequences for inference...")
    X_sequences, y_sequences, location_indices = [], [], []
    unique_locations = df['location_encoded'].unique()
    
    for i, loc in enumerate(unique_locations):
        loc_df = df[df['location_encoded'] == loc]
        loc_X = X_scaled_df.loc[loc_df.index]
        
        # For inference, we might not have target values
        loc_y = loc_df['class'].values if 'class' in loc_df.columns else np.zeros(len(loc_df))
        
        # Create sequences - we need at least input_len points
        if len(loc_df) >= input_len:
            # Take the most recent sequence for each location
            start_idx = len(loc_df) - input_len
            X_seq = loc_X.iloc[start_idx:].values
            
            X_sequences.append(X_seq)
            location_indices.append(loc)
            
            # For targets, if we have enough data points after input sequence
            if len(loc_df) >= input_len + output_len:
                y_target = loc_y[start_idx + input_len:start_idx + input_len + output_len]
            else:
                # Pad with zeros if we don't have enough future data
                available_future = len(loc_df) - input_len
                if available_future > 0:
                    y_existing = loc_y[start_idx + input_len:]
                    y_padding = np.zeros(max(0, output_len - available_future))
                    y_target = np.concatenate([y_existing, y_padding])
                else:
                    y_target = np.zeros(output_len)
            
            # Ensure y_target is exactly output_len length
            if len(y_target) != output_len:
                if len(y_target) > output_len:
                    y_target = y_target[:output_len]
                else:
                    y_target = np.pad(y_target, (0, output_len - len(y_target)), 'constant')
                    
            y_sequences.append(y_target)
            
        if (i+1) % 50 == 0:
            print(f"Processed location {i+1}/{len(unique_locations)}")
    
    if len(X_sequences) == 0:
        raise ValueError(f"No valid sequences found. Need at least {input_len} time steps per location.")
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    location_indices = np.array(location_indices)
    
    print(f"Created {len(X_sequences)} sequences for inference")
    print(f"Input sequence shape: {X_sequences.shape}")
    print(f"Output sequence shape: {y_sequences.shape}")
    
    processed_data = {
        'X_sequences': X_sequences,
        'y_sequences': y_sequences,
        'location_indices': location_indices,
        'X_scaled': X_scaled_df,
        'df': df,
        'feature_columns': state['feature_columns'],
        'preprocessing_state': state
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed inference data saved to {output_path}")
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing for NILE Competition")
    parser.add_argument("--mode", choices=["train", "inference"], required=True,
                       help="Preprocessing mode: train or inference")
    parser.add_argument("--data_path", required=True, help="Path to input data file")
    parser.add_argument("--output_dir", help="Output directory for training data")
    parser.add_argument("--output_path", help="Output path for inference data")
    parser.add_argument("--preprocessing_state", help="Path to preprocessing state file (for inference)")
    parser.add_argument("--input_len", type=int, default=168, help="Input sequence length")
    parser.add_argument("--output_len", type=int, default=72, help="Output sequence length")
    parser.add_argument("--stride", type=int, default=24, help="Sequence stride")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.output_dir:
            raise ValueError("--output_dir is required for training mode")
        preprocess_for_training(args.data_path, args.output_dir, 
                              args.input_len, args.output_len, args.stride)
    
    elif args.mode == "inference":
        if not args.output_path or not args.preprocessing_state:
            raise ValueError("--output_path and --preprocessing_state are required for inference mode")
        preprocess_for_inference(args.data_path, args.preprocessing_state, args.output_path)
