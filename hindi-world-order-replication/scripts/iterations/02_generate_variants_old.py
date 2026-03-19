#!/usr/bin/env python3
"""
Step 4: Strict Replication Filter (4 Conditions)
a) Well-defined subjects and objects
b) Projective trees
c) Declarative sentences
d) Finite verb root with >= 2 preverbal dependents
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
# 4 REPLICATION CONDITIONS
# ============================================================================

def is_projective(sentence: Sentence) -> bool:
    """Condition (b): Checks if the dependency tree is projective (no crossing arcs)."""
    words = sentence.words
    for i, w1 in enumerate(words):
        if w1.head == 0: continue 
        start1, end1 = min(w1.idx, w1.head), max(w1.idx, w1.head)
        for j, w2 in enumerate(words):
            if w2.head == 0: continue
            start2, end2 = min(w2.idx, w2.head), max(w2.idx, w2.head)
            if (start1 < start2 < end1 < end2) or (start2 < start1 < end2 < end1):
                return False
    return True

def is_declarative(sentence: Sentence) -> bool:
    """Condition (c): Excludes interrogatives (questions)."""
    interrogative_markers = {'क्या', 'क्यों', 'कैसे', 'कब', 'कहाँ', 'किस'}
    if '?' in sentence.text or '？' in sentence.text:
        return False
    forms = {w.form for w in sentence.words}
    if any(q in forms for q in interrogative_markers):
        return False
    return True

def is_finite_verb(word: Word) -> bool:
    """Condition (d): Checks if the root is a finite verb (handles string features)."""
    if word.upos != 'VERB':
        return False
    
    # FIX: Handle feats as a string to avoid AttributeError
    feats = word.feats if hasattr(word, 'feats') else ""
    if isinstance(feats, str):
        if 'VerbForm=Inf' in feats or 'VerbForm=Part' in feats:
            return False
        return True
    elif isinstance(feats, dict):
        return feats.get('VerbForm') not in ['Inf', 'Part']
    return True

# ============================================================================
# MAIN FILTERING LOGIC
# ============================================================================

INPUT_FILE = "./data/processed/valid_sentences.pkl"
OUTPUT_FILE = "./data/processed/replication_filtered_sentences.pkl"

with open(INPUT_FILE, 'rb') as f:
    sentences = pickle.load(f)

# Relation sets for Condition (a)
SUBJ_TAGS = {'nsubj', 'nsubj:pass', 'csubj', 'csubj:pass'}
OBJ_TAGS = {'obj', 'iobj'}

filtered = []
stats = {k: 0 for k in ['total', 'non_finite', 'non_transitive', 'non_projective', 'non_declarative', 'wrong_length', 'included']}

print("\nApplying 4 Replication Conditions...")

for sent in tqdm(sentences):
    stats['total'] += 1
    
    # 1. Condition (d): Finite Verb Root
    if not sent.root_word or not is_finite_verb(sent.root_word):
        stats['non_finite'] += 1
        continue
    
    # 2. Condition (a): Both Subject AND Object present
    has_sub = any(w.deprel in SUBJ_TAGS for w in sent.words)
    has_obj = any(w.deprel in OBJ_TAGS for w in sent.words)
    if not (has_sub and has_obj):
        stats['non_transitive'] += 1
        continue
        
    # 3. Condition (b): Projectivity
    if not is_projective(sent):
        stats['non_projective'] += 1
        continue
        
    # 4. Condition (c): Declarative Mood
    if not is_declarative(sent):
        stats['non_declarative'] += 1
        continue

    # 5. Condition (d): >= 2 Preverbal Dependents
    preverbal = sent.get_preverbal_constituents()
    if len(preverbal) < 2:
        stats['wrong_length'] += 1
        continue
    

    filtered.append(sent)
    stats['included'] += 1

# Save Result
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(filtered, f)

print(f"\nReplication Complete. Final Count: {len(filtered):,}")
print(f"Exclusions: {stats}")