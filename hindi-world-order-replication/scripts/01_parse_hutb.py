#!/usr/bin/env python3
"""
Step 1: Parse UD Hindi-HDTB corpus and save filtered sentences
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
# CONFIGURATION - UPDATE THIS PATH!
# ============================================================================

UD_DIR = os.path.expanduser("~/Desktop/Mtech Thesis/UD_Hindi-HDTB-master")
OUTPUT_DIR = "./data/processed"


# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_pickle(sentences, filepath):
    """Save e2as pickle"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(sentences, f)
    print(f" Saved {len(sentences):,} sentences to {filepath}")


def save_json(sentences, filepath):
    """Save as JSON"""
    data = []
    for sent in sentences:
        sent_dict = {
            'sent_id': sent.sent_id,
            'text': sent.text,
            'root_idx': sent.root_idx,
            'length': len(sent),
            'words': [
                {
                    'idx': w.idx,
                    'form': w.form,
                    'lemma': w.lemma,
                    'upos': w.upos,
                    'head': w.head,
                    'deprel': w.deprel
                }
                for w in sent.words
            ]
        }
        data.append(sent_dict)
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f" Saved {len(sentences):,} sentences to {filepath}")


def save_csv(sentences, filepath):
    """Save metadata as CSV"""
    data = []
    for sent in sentences:
        row = {
            'sent_id': sent.sent_id,
            'text': sent.text,
            'length': len(sent),
            'root_verb': sent.root_word.form if sent.root_word else '',
            'root_idx': sent.root_idx or 0,
            'has_subject': sent.has_subject(),
            'has_object': sent.has_object(),
            'num_preverbal': len(sent.get_preverbal_constituents()),
        }
        data.append(row)
    
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV without pandas
    fieldnames = [
        'sent_id',
        'text',
        'length',
        'root_verb',
        'root_idx',
        'has_subject',
        'has_object',
        'num_preverbal',
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"Saved metadata for {len(data):,} sentences to {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" STEP 1: PARSE UD HINDI-HDTB CORPUS")
    print("="*70 + "\n")
    
    # Verify path
    if not os.path.exists(UD_DIR):
        print(f" ERROR: UD directory not found!")
        print(f"   Looking for: {UD_DIR}")
        print(f"\n   Please update UD_DIR in scripts/01_parse_hutb.py")
        return 1
    
    print(f"UD Directory: {UD_DIR}\n")
    
    # Parse corpus
    print("Parsing corpus...")
    all_sentences = parse_ud_hindi(UD_DIR)
    
    if not all_sentences:
        print("\n No sentences parsed!")
        return 1
    
    # Filter
    print("Filtering sentences...")
    valid_sentences = filter_valid_sentences(all_sentences)
    
    # Samples
    print("Sample sentences:")
    show_sample_sentences(valid_sentences, n=3)
    
    # Save
    print("Saving data...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    save_pickle(
        all_sentences,
        os.path.join(OUTPUT_DIR, "all_sentences.pkl")
    )
    
    save_pickle(
        valid_sentences,
        os.path.join(OUTPUT_DIR, "valid_sentences.pkl")
    )
    
    save_json(
        valid_sentences[:100],
        os.path.join(OUTPUT_DIR, "sample_100_sentences.json")
    )
    
    save_csv(
        valid_sentences,
        os.path.join(OUTPUT_DIR, "valid_sentences_metadata.csv")
    )
    
    # Summary
    print("\n" + "="*70)
    print(" STEP 1 COMPLETE!")
    print("="*70)
    print(f"\nStatistics:")
    print(f"  Total sentences parsed: {len(all_sentences):,}")
    print(f"  Valid sentences (subject + object): {len(valid_sentences):,}")
    print(f"  Target for paper: 1,996")
    print(f"  Status: {' SUFFICIENT' if len(valid_sentences) >= 1996 else '️ NEED MORE'}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"  1. all_sentences.pkl")
    print(f"  2. valid_sentences.pkl")
    print(f"  3. sample_100_sentences.json")
    print(f"  4. valid_sentences_metadata.csv")
    print(f"\n Next: Run scripts/02_generate_variants.py")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
