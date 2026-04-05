#!/usr/bin/env python3
"""
Find LSTM Surprisal Data
Check which files contain LSTM surprisal scores
"""

import pickle
import pandas as pd
import os

print("\n" + "="*70)
print(" SEARCHING FOR LSTM SURPRISAL DATA")
print("="*70 + "\n")

files_to_check = [
    'data/processed/all_variants_final.pkl',
    'data/processed/TRUE_UNIQUE_variants.pkl',
    'data/processed/filtered_pairwise_dataset.pkl',
    'data/processed/pairwise_dataset.pkl',
]

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f"⚠️  Not found: {filepath}\n")
        continue
    
    print(f"{'='*70}")
    print(f"FILE: {filepath}")
    print(f"{'='*70}\n")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check data type
        if isinstance(data, pd.DataFrame):
            print(f"Type: DataFrame")
            print(f"Rows: {len(data):,}")
            print(f"Columns ({len(data.columns)}):")
            for col in data.columns:
                print(f"  - {col}")
            
            # Check for LSTM-related columns
            lstm_cols = [col for col in data.columns if 'lstm' in col.lower() or 'surprisal' in col.lower()]
            if lstm_cols:
                print(f"\n✅ FOUND LSTM-RELATED COLUMNS:")
                for col in lstm_cols:
                    print(f"  - {col}")
                    print(f"    Mean: {data[col].mean():.4f}")
                    print(f"    Min: {data[col].min():.4f}")
                    print(f"    Max: {data[col].max():.4f}")
            
            print(f"\nFirst row sample:")
            print(data.iloc[0])
            
        elif isinstance(data, list):
            print(f"Type: List")
            print(f"Length: {len(data):,}")
            
            if len(data) > 0:
                first_item = data[0]
                print(f"\nFirst item type: {type(first_item)}")
                
                if isinstance(first_item, dict):
                    print(f"Dict keys: {list(first_item.keys())}")
                    
                    # Check for LSTM-related keys
                    lstm_keys = [k for k in first_item.keys() if 'lstm' in str(k).lower() or 'surprisal' in str(k).lower()]
                    if lstm_keys:
                        print(f"\n✅ FOUND LSTM-RELATED KEYS:")
                        for key in lstm_keys:
                            print(f"  - {key}: {first_item[key]}")
                    
                    print(f"\nFull first item:")
                    for k, v in first_item.items():
                        if isinstance(v, (list, dict)) and len(str(v)) > 50:
                            print(f"  {k}: {type(v)} (length: {len(v) if hasattr(v, '__len__') else 'N/A'})")
                        else:
                            print(f"  {k}: {v}")
                else:
                    print(f"First item: {first_item}")
        
        else:
            print(f"Type: {type(data)}")
            if hasattr(data, 'keys'):
                print(f"Keys: {list(data.keys())[:10]}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error loading: {e}\n")

print("="*70)
print(" SEARCH COMPLETE")
print("="*70 + "\n")