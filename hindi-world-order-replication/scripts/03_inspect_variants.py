#!/usr/bin/env python3
import sys
import os
import pickle

# Add src to path so pickle can find the classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import classes needed for unpickling
from parsers.ud_parser import Sentence, Word

# Load data
print("Loading data...")

with open('data/processed/reference_sentences.pkl', 'rb') as f:
    refs = pickle.load(f)

with open('data/processed/all_variants.pkl', 'rb') as f:
    variants = pickle.load(f)

with open('data/processed/pairwise_dataset.pkl', 'rb') as f:
    pairs = pickle.load(f)

print("\n" + "="*60)
print("DATASET INSPECTION")
print("="*60)
print(f"Reference sentences: {len(refs):,}")
print(f"All variants: {len(variants):,}")
print(f"Pairwise comparisons: {len(pairs):,}")

print(f"\nFirst reference sentence:")
print(f"  ID: {refs[0].sent_id}")
print(f"  Text: {refs[0].text}")
print(f"  Length: {len(refs[0])} words")

print(f"\nFirst variant:")
v = variants[0]
print(f"  Sent ID: {v['sent_id']}")
print(f"  Preverbal words: {' '.join(v['preverbal_words'])}")
print(f"  Root: {v['root_form']}")
print(f"  Is reference: {v['is_reference']}")

print(f"\nFirst pair:")
p = pairs[0]
print(f"  Sent ID: {p['sent_id']}")
print(f"  Label: {p['label']} ({'Reference preferred' if p['label']==1 else 'Variant preferred'})")
print(f"  Type: {p['pair_type']}")
print("="*60 + "\n")