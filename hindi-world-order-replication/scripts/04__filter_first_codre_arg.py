#!/usr/bin/env python3
"""
Filter sentences where FIRST preverbal constituent is Subject or Object
Following professor's criteria
"""

import sys
import os
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsers.ud_parser import Sentence, Word

print("\n" + "="*70)
print(" FILTERING: First Constituent Must Be Subject/Object")
print("="*70 + "\n")

# Load valid sentences
INPUT_FILE = "./data/processed/valid_sentences.pkl"

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    exit(1)

print(f"Loading sentences from: {INPUT_FILE}")

with open(INPUT_FILE, 'rb') as f:
    sentences = pickle.load(f)

print(f"  → Loaded {len(sentences):,} sentences\n")

# Core argument relations (Subject and Object types)
CORE_ARGUMENTS = {
    'nsubj',       # nominal subject
    'nsubj:pass',  # passive subject  
    'obj',         # object
    'iobj',        # indirect object
    'csubj',       # clausal subject
    'csubj:pass'   # passive clausal subject
}

# Filter sentences
print("Filtering sentences...")

filtered = []
stats = {
    'total': len(sentences),
    'excluded_length': 0,
    'excluded_first': 0,
    'excluded_no_preverbal': 0,
    'included': 0
}

excluded_examples = []
included_examples = []

for sent in sentences:
    # Get preverbal constituents
    preverbal = sent.get_preverbal_constituents()
    
    # Skip if no preverbal constituents
    if len(preverbal) == 0:
        stats['excluded_no_preverbal'] += 1
        continue
    
    # Must have 2-4 preverbal constituents for variant generation
    if not (2 <= len(preverbal) <= 4):
        stats['excluded_length'] += 1
        continue
    
    # Get first constituent
    first = preverbal[0]
    
    # CRITICAL FILTER: First must be subject or object
    if first.deprel not in CORE_ARGUMENTS:
        stats['excluded_first'] += 1
        if len(excluded_examples) < 10:
            excluded_examples.append((sent, first))
        continue
    
    # Passed all filters
    filtered.append(sent)
    stats['included'] += 1
    
    if len(included_examples) < 10:
        included_examples.append((sent, first))

# Print statistics
print("\n" + "="*70)
print("FILTERING RESULTS")
print("="*70)
print(f"Total input sentences: {stats['total']:,}")
print(f"\nExclusions:")
print(f"  - No preverbal constituents: {stats['excluded_no_preverbal']:,}")
print(f"  - Wrong length (not 2-4): {stats['excluded_length']:,}")
print(f"  - First not subj/obj: {stats['excluded_first']:,}")
print(f"\n REMAINING: {stats['included']:,} sentences")
print(f"   Filter rate: {100 * stats['included'] / stats['total']:.1f}%")
print("="*70 + "\n")

# Show excluded examples
if excluded_examples:
    print("Examples of EXCLUDED sentences (first constituent not subj/obj):\n")
    
    for i, (sent, first_const) in enumerate(excluded_examples[:5], 1):
        preverbal = sent.get_preverbal_constituents()
        print(f"{i}. Text: {sent.text}")
        print(f"   First constituent: '{first_const.form}' (relation: {first_const.deprel})")
        print(f"   All preverbal: {[f'{w.form}({w.deprel})' for w in preverbal]}")
        print(f"   Excluded: First is '{first_const.deprel}', not subject/object\n")

# Show included examples
if included_examples:
    print("="*70)
    print("Examples of INCLUDED sentences (first is subj/obj):\n")
    
    for i, (sent, first_const) in enumerate(included_examples[:5], 1):
        preverbal = sent.get_preverbal_constituents()
        print(f"{i}. Text: {sent.text}")
        print(f"   First constituent: '{first_const.form}' (relation: {first_const.deprel})")
        print(f"   All preverbal: {[f'{w.form}({w.deprel})' for w in preverbal]}")
        print(f"   Included: First is '{first_const.deprel}' (subject/object)\n")

# Save filtered sentences
OUTPUT_DIR = "./data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "first_core_arg_sentences.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(filtered, f)

print("="*70)
print(f"SAVED: {len(filtered):,} filtered sentences")
print(f"   Output: {OUTPUT_FILE}")
print("="*70 + "\n")

print("NEXT STEPS:")
print("1.  Filtering complete")
print("2. ⏳ Re-generate variants with filtered sentences")
print("   → Run: python scripts/05_generate_variants_filtered.py")
print()