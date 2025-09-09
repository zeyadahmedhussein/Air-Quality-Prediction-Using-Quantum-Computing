#!/usr/bin/env python3
"""
Test script for NILE Competition CSV preprocessing pipeline
Tests the complete flow from CSV unseen data to predictions
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def test_data_exists():
    """Test that required data files exist"""
    print("Testing data file existence...")
    
    required_files = [
        "data/unseen.csv",
        "data/processed/preprocessing_state.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({file_size:,} bytes)")
    
    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False
    
    print("  ✓ All required files exist")
    return True

def test_csv_format():
    """Test CSV file format and content"""
    print("\nTesting CSV format...")
    
    try:
        df = pd.read_csv("data/unseen.csv")
        print(f"  ✓ CSV loaded successfully: {df.shape}")
        
        # Check required columns
        required_cols = ['time', 'lat', 'lon', 'class', 'PM25_MERRA2', 'DUCMASS', 
                        'TOTANGSTR', 'DUFLUXV', 'SSFLUXV', 'DUFLUXU', 'BCCMASS', 'SSSMASS25']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return False
        
        print(f"  ✓ All required columns present")
        
        # Check data types
        print(f"  ✓ Time column format: {df['time'].iloc[0]}")
        print(f"  ✓ Lat range: {df['lat'].min():.2f} to {df['lat'].max():.2f}")
        print(f"  ✓ Lon range: {df['lon'].min():.2f} to {df['lon'].max():.2f}")
        print(f"  ✓ Classes: {df['class'].unique()}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading CSV: {e}")
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\nTesting preprocessing...")
    
    cmd = [
        sys.executable, "src/preprocessing/data_preprocessing.py",
        "--mode", "inference",
        "--data_path", "data/unseen.csv", 
        "--preprocessing_state", "data/processed/preprocessing_state.pkl",
        "--output_path", "data/processed/test_processed.pkl"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("  ✓ Preprocessing completed successfully")
            print(f"    Output: {result.stdout.strip().split('/')[-1]}")
            
            # Check output file
            if os.path.exists("data/processed/test_processed.pkl"):
                print("  ✓ Processed file created")
                return True
            else:
                print("  ✗ Processed file not found")
                return False
        else:
            print(f"  ✗ Preprocessing failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ✗ Preprocessing timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error running preprocessing: {e}")
        return False

def test_directory_structure():
    """Test required directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src/preprocessing",
        "src/quantum_model", 
        "src/classical_model",
        "data/processed",
        "results/physical",
        "results/simulator",
        "evidence/logs",
        "evidence/screenshots",
        "models"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"  ✓ {dir_path}")
    
    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False
    
    return True

def test_config_files():
    """Test configuration files"""
    print("\nTesting configuration files...")
    
    config_files = [
        "configs/qiskit_backend_config.json",
        "MANIFEST.json",
        "requirements.txt"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✓ {config_file}")
            
            if config_file.endswith('.json'):
                try:
                    with open(config_file, 'r') as f:
                        json.load(f)
                    print(f"    ✓ Valid JSON format")
                except json.JSONDecodeError:
                    print(f"    ✗ Invalid JSON format")
                    return False
        else:
            print(f"  ✗ Missing: {config_file}")
            return False
    
    return True

def create_test_report(results):
    """Create test report"""
    print(f"\n{'='*60}")
    print("CSV PREPROCESSING PIPELINE TEST REPORT")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 All tests passed! The CSV preprocessing pipeline is ready.")
        print("\nNext steps:")
        print("1. Run training preprocessing on parquet data")
        print("2. Train quantum and classical models")
        print("3. Test full evaluation pipeline")
        
        print(f"\nQuick test command:")
        print(f"python src/run_evaluation.py --dataset data/unseen.csv --backend ibm_brisbane --shots 2048 --mode simulator --out results/")
    else:
        print(f"\n❌ Some tests failed. Please fix the issues before proceeding.")
        
    return passed_tests == total_tests

def main():
    """Run all tests"""
    print("NILE Competition - CSV Preprocessing Pipeline Tests")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    test_results = {
        "Data Files Exist": test_data_exists(),
        "CSV Format Valid": test_csv_format(),
        "Directory Structure": test_directory_structure(),
        "Config Files": test_config_files(),
        "Preprocessing Pipeline": test_preprocessing()
    }
    
    # Create report
    success = create_test_report(test_results)
    
    # Clean up test files
    test_files = ["data/processed/test_processed.pkl"]
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
