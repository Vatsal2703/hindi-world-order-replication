#!/usr/bin/env python3
"""
Variant Generation - Standalone Version
Generates word order variants from parsed sentences
"""

import os
import sys
import pickle
from itertools import permutations
from collections import defaultdict, Counter
from tqdm import tqdm

# Add src to path so pickle can find the Sentence and Word classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================================
# LOAD DATA
# ============================================================================

INPUT_FILE = "./data/processed/valid_sentences.pkl"
OUTPUT_DIR = "./data/processed"

print("\n" + "="*70)
print(" VARIANT GENERATION - STANDALONE")
print("="*70 + "\n")

if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: {INPUT_FILE} not found!")
    print("   Run scripts/01_parse_hutb.py first")
    exit(1)

print(f"Loading sentences from: {INPUT_FILE}")

# Import Sentence and Word classes so pickle can deserialize them
try:
    from parsers.ud_parser import Sentence, Word
    print("  ✅ Imported parser classes")
except ImportError:
    print("  ⚠️  Could not import parser classes, trying alternative...")
    from src.parsers.ud_parser import Sentence, Word
    print("  ✅ Imported parser classes (alternative path)")

with open(INPUT_FILE, 'rb') as f:
    sentences = pickle.load(f)

print(f"  → Loaded {len(sentences):,} sentences\n")

# ============================================================================
# EXTRACT DEPENDENCY PATTERNS
# ============================================================================

print("Extracting dependency patterns from corpus...")

attested_patterns = set()

for sent in sentences:
    if sent.root_idx is None:
        continue
    
    preverbal = sent.get_preverbal_constituents()
    
    if len(preverbal) < 2:
        continue
    
    # Extract dependency relation sequence
    deprel_sequence = tuple(w.deprel for w in preverbal)
    attested_patterns.add(deprel_sequence)
    attested_patterns.add(tuple(reversed(deprel_sequence)))

print(f"  → Extracted {len(attested_patterns):,} unique patterns\n")

# ============================================================================
# SELECT SUITABLE SENTENCES
# ============================================================================

print("Selecting suitable sentences...")

suitable = []

for sent in sentences:
    preverbal = sent.get_preverbal_constituents()
    num_preverbal = len(preverbal)
    
    if (2 <= num_preverbal <= 4 and
        sent.has_subject() and 
        sent.has_object() and
        sent.root_word is not None):
        suitable.append(sent)

print(f"  → Found {len(suitable):,} suitable sentences")

# Select ~2000
if len(suitable) > 2000:
    suitable.sort(key=lambda s: len(s.get_preverbal_constituents()), reverse=True)
    selected = suitable[:2000]
else:
    selected = suitable

print(f"  → Selected {len(selected):,} reference sentences\n")

# ============================================================================
# GENERATE VARIANTS
# ============================================================================

print("Generating variants...")

all_variants = []
total_variants = 0

for sent in tqdm(selected, desc="Processing"):
    preverbal = sent.get_preverbal_constituents()
    
    if len(preverbal) < 2:
        continue
    
    # Generate all permutations
    for perm in permutations(preverbal):
        deprel_seq = tuple(w.deprel for w in perm)
        
        # Check if grammatical
        if deprel_seq in attested_patterns:
            variant = {
                'sent_id': sent.sent_id,
                'reference_text': sent.text,
                'root_idx': sent.root_idx,
                'root_form': sent.root_word.form,
                'original_order': [w.idx for w in preverbal],
                'variant_order': [w.idx for w in perm],
                'deprel_sequence': deprel_seq,
                'preverbal_words': [w.form for w in perm],
                'is_reference': (perm == tuple(preverbal))
            }
            
            all_variants.append(variant)
            
            if not variant['is_reference']:
                total_variants += 1

print(f"\n{'='*70}")
print("VARIANT GENERATION STATISTICS")
print(f"{'='*70}")
print(f"Reference sentences: {len(selected):,}")
print(f"Total variants (excluding references): {total_variants:,}")
print(f"Total sentences (refs + variants): {len(all_variants):,}")
print(f"Average variants per reference: {total_variants / len(selected):.2f}")
print(f"{'='*70}\n")

# ============================================================================
# CREATE PAIRWISE DATASET
# ============================================================================

print("Creating pairwise dataset (Joachims transformation)...")

# Group by sentence ID
variants_by_sent = defaultdict(list)
for v in all_variants:
    variants_by_sent[v['sent_id']].append(v)

pairs = []

for sent_id, sent_variants in variants_by_sent.items():
    # Find reference
    reference = None
    non_refs = []
    
    for v in sent_variants:
        if v['is_reference']:
            reference = v
        else:
            non_refs.append(v)
    
    if reference is None:
        continue
    
    # Create pairs
    for variant in non_refs:
        # Pair 1: Ref-Var (label=1)
        pairs.append({
            'sent_id': sent_id,
            'sentence_a': reference,
            'sentence_b': variant,
            'label': 1,
            'pair_type': 'ref-var'
        })
        
        # Pair 2: Var-Ref (label=0)
        pairs.append({
            'sent_id': sent_id,
            'sentence_a': variant,
            'sentence_b': reference,
            'label': 0,
            'pair_type': 'var-ref'
        })

print(f"  → Created {len(pairs):,} pairwise comparisons")
print(f"  → Balance: {sum(1 for p in pairs if p['label']==1):,} positive, {sum(1 for p in pairs if p['label']==0):,} negative\n")

# ============================================================================
# SHOW EXAMPLES
# ============================================================================

print(f"{'='*70}")
print("EXAMPLE VARIANTS (first 3 reference sentences)")
print(f"{'='*70}\n")

shown = 0
for sent_id, variants in list(variants_by_sent.items())[:3]:
    reference = None
    for v in variants:
        if v['is_reference']:
            reference = v
            break
    
    if reference:
        print(f"Reference ID: {sent_id}")
        print(f"Root verb: {reference['root_form']}")
        print(f"Variants: {len(variants) - 1}\n")
        
        print(f"  [REF] {' '.join(reference['preverbal_words'])} {reference['root_form']}")
        print(f"        Deprels: {' → '.join(reference['deprel_sequence'])}\n")
        
        for i, v in enumerate(variants, 1):
            if not v['is_reference']:
                print(f"  [VAR{i}] {' '.join(v['preverbal_words'])} {v['root_form']}")
                print(f"         Deprels: {' → '.join(v['deprel_sequence'])}")
        
        print()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("Saving data...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save references
with open(os.path.join(OUTPUT_DIR, "reference_sentences.pkl"), 'wb') as f:
    pickle.dump(selected, f)
print(f"Saved {len(selected):,} references")

# Save all variants
with open(os.path.join(OUTPUT_DIR, "all_variants.pkl"), 'wb') as f:
    pickle.dump(all_variants, f)
print(f"Saved {len(all_variants):,} variants")

# Save pairs
with open(os.path.join(OUTPUT_DIR, "pairwise_dataset.pkl"), 'wb') as f:
    pickle.dump(pairs, f)
print(f" Saved {len(pairs):,} pairs")

print("\n" + "="*70)
print("  VARIANT GENERATION COMPLETE!")
print("="*70)
print(f"\nFiles created:")
print(f"  1. reference_sentences.pkl")
print(f"  2. all_variants.pkl")
print(f"  3. pairwise_dataset.pkl")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("="*70 + "\n")