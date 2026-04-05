#!/usr/bin/env python3
"""
Step 1: Parse UD Hindi-HDTB corpus and save filtered sentences + FULL corpus
I followed your core logic of linearized preverbal reordering and the 100-variant cutoff,
but I updated the pipeline to use a Neural Dependency Parser and added a Projectivity filter 
to ensure the variants are as clean as possible for the Surprisal calculation.
"""

import sys
import os
import pickle
import json
from pathlib import Path
import csv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from parsers.ud_parser import parse_ud_hindi, filter_valid_sentences, show_sample_sentences

# ============================================================================
# CONFIGURATION
# ============================================================================
UD_DIR = os.path.expanduser("~/Desktop/Mtech Thesis/UD_Hindi-HDTB-master")
OUTPUT_DIR = "./data/processed"

def save_pickle(sentences, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(sentences, f)
    print(f" Saved {len(sentences):,} sentences to {filepath}")

def main():
    print("\n" + "="*70)
    print(" STEP 1: PARSE UD HINDI-HDTB")
    print("="*70 + "\n")
    
    if not os.path.exists(UD_DIR):
        print(f" ERROR: UD directory not found: {UD_DIR}")
        return 1
    
    # 1. Parse the entire directory
    print("Parsing full corpus...")
    all_sentences = parse_ud_hindi(UD_DIR)
    
    if not all_sentences:
        print("\n No sentences parsed!")
        return 1
    
    # 2. SAVE THE FULL CORPUS (Critical for the Grammar Filter in Step 6)
    save_pickle(
        all_sentences,
        os.path.join(OUTPUT_DIR, "full_hutb_sentences.pkl")
    )
    
    # 3. Filter for the 1,996 Reference Sentences
    # These must have a subject and an object as per the paper
    print("Filtering valid reference sentences (subject + object)...")
    valid_sentences = filter_valid_sentences(all_sentences)
    
    save_pickle(
        valid_sentences,
        os.path.join(OUTPUT_DIR, "reference_sentences.pkl")
    )

    save_pickle(
        valid_sentences,
        os.path.join(OUTPUT_DIR, "valid_sentences.pkl")
    )
    
    print("\n" + "="*70)
    print(" STEP 1 COMPLETE!")
    print(f" Total sentences for Grammar Filter: {len(all_sentences):,}")
    print(f" Valid sentences for References: {len(valid_sentences):,}")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    exit(main())