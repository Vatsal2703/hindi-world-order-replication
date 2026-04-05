#!/usr/bin/env python3
"""
Step 4: Filter sentences based on the 4 replication conditions from the paper.
Conditions:
a) Well-defined subjects and objects
b) Projective trees
c) Declarative sentences
d) Finite verb root with at least two preverbal dependents
"""

import sys
import os
import pickle
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsers.ud_parser import Sentence, Word

# ============================================================================
# LINGUISTIC CONDITION CHECKERS
# ============================================================================

def is_projective(sentence: Sentence) -> bool:
    """
    Condition (b): Checks if the dependency tree is projective.
    A tree is projective if no dependency arcs cross.
    """
    words = sentence.words
    for i, w1 in enumerate(words):
        if w1.head == 0: continue # Skip root
        
        # Define the range of the first arc (from word to its head)
        start1, end1 = min(w1.idx, w1.head), max(w1.idx, w1.head)
        
        for j, w2 in enumerate(words):
            if w2.head == 0: continue
            start2, end2 = min(w2.idx, w2.head), max(w2.idx, w2.head)
            
            # Check for crossing: start1 < start2 < end1 < end2
            if (start1 < start2 < end1 < end2) or (start2 < start1 < end2 < end1):
                return False
    return True

def is_declarative(sentence: Sentence) -> bool:
    if sentence.root_word:
        misc = sentence.root_word.misc or ""
        return 'Stype=declarative' in misc
    return False

def is_finite_verb(word: Word) -> bool:
    """
    Condition (d): Checks if the root is a finite verb.

    """
    # Simply ensure the root is a verbal element
    return word.upos in ['VERB', 'AUX']

# ============================================================================
# MAIN FILTERING SCRIPT
# ============================================================================

INPUT_FILE = "./data/processed/valid_sentences.pkl"
OUTPUT_FILE = "./data/processed/replication_filtered_sentences.pkl"

print("\n" + "="*70)
print(" REPLICATION FILTERING: APPLYING 4 CONDITIONS")
print("="*70 + "\n")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    exit(1)

with open(INPUT_FILE, 'rb') as f:
    sentences = pickle.load(f)

# Define core relation sets
SUBJ_TAGS = {'nsubj', 'nsubj:pass', 'csubj', 'csubj:pass'}
OBJ_TAGS = {'obj', 'iobj'}

filtered = []
stats = {
    'total': len(sentences),
    'excluded_non_transitive': 0, # Condition (a)
    'excluded_non_projective': 0, # Condition (b)
    'excluded_non_declarative': 0, # Condition (c)
    'excluded_non_finite': 0,      # Condition (d)
    'excluded_preverbal_count': 0, # Condition (d)
    'included': 0
}

for sent in tqdm(sentences, desc="Filtering"):
    # 1. Condition (d): Finite Verb Root
    if not sent.root_word or not is_finite_verb(sent.root_word):
        stats['excluded_non_finite'] += 1
        continue

    # 2. Condition (a): Both Subject and Object present
    has_sub = any(w.deprel in SUBJ_TAGS for w in sent.words)
    has_obj = any(w.deprel in OBJ_TAGS for w in sent.words)
    if not (has_sub and has_obj):
        stats['excluded_non_transitive'] += 1
        continue

    # 3. Condition (b): Projectivity (no crossing arcs)
    if not is_projective(sent):
        stats['excluded_non_projective'] += 1
        continue

    # 4. Condition (c): Declarative only
    if not is_declarative(sent):
        stats['excluded_non_declarative'] += 1
        # --- ADD THESE LINES TO SEE THE DISREGARDED SENTENCES ---
        if stats['excluded_non_declarative'] < 10: # Print only first 10 to avoid flooding
            print(f"\n🚫 Disregarded (Non-Declarative): {sent.text}")
        # -------------------------------------------------------
        continue

    # 5. Condition (d): At least two preverbal dependents (keeping 2-4 limit)
    preverbal = sent.get_preverbal_constituents() 
    if len(preverbal) < 2: 
        stats['excluded_preverbal_count'] += 1
        continue

    # ALL CONDITIONS MET - (Subject/Object First condition removed)
    filtered.append(sent)
    stats['included'] += 1

# Summary results
print("\n" + "="*70)
print("FILTERING RESULTS SUMMARY")
print("="*70)
print(f"Total processed: {stats['total']:,}")
print(f"Included: {stats['included']:,}")
print(f"Filter Rate: {100 * stats['included'] / stats['total']:.1f}%")
print("-" * 30)
print(f"Exclusions:")
print(f" - Non-transitive: {stats['excluded_non_transitive']:,}")
print(f" - Non-projective: {stats['excluded_non_projective']:,}")
print(f" - Non-declarative: {stats['excluded_non_declarative']:,}")
print(f" - Non-finite root: {stats['excluded_non_finite']:,}")
print(f" - Preverbal count: {stats['excluded_preverbal_count']:,}")
print("="*70 + "\n")

# Save results
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(filtered, f)

print(f"Saved {len(filtered):,} sentences to {OUTPUT_FILE}")