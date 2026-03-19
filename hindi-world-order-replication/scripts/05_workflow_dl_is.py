#!/usr/bin/env python3

import os
import sys
import pickle
from itertools import islice, permutations
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsers.ud_parser import Sentence, Word
from features.basic_features import extract_features_for_sentence

print("\n" + "="*70)
print(" COMPLETE WORKFLOW - PROFESSOR'S FILTERED DATA")
print("="*70 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "./data/processed/replication_filtered_sentences.pkl"
ALL_SENTENCES_FILE = "./data/processed/all_sentences.pkl"
OUTPUT_DIR_PROCESSED = "./data/processed"
OUTPUT_DIR_FEATURES = "./data/features"

# ============================================================================
# STEP 1: LOAD FILTERED SENTENCES
# ============================================================================
print("STEP 1: Loading filtered sentences...")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found! Run scripts/04_filter_first_core_arg.py first.")
    exit(1)

with open(INPUT_FILE, 'rb') as f:
    sentences = pickle.load(f)

print(f"  → Loaded {len(sentences):,} filtered sentences")

# Load ALL sentences to serve as a lookup for Discourse Context (Information Status)
if os.path.exists(ALL_SENTENCES_FILE):
    with open(ALL_SENTENCES_FILE, 'rb') as f:
        all_parsed_sentences = pickle.load(f)
    full_corpus_lookup = {s.sent_id: s for s in all_parsed_sentences}
    print(f"  → Loaded {len(full_corpus_lookup):,} total corpus sentences for discourse context lookup.")
else:
    print(f"WARNING: {ALL_SENTENCES_FILE} not found. Discourse context features will be 0.")
    full_corpus_lookup = {}

def get_previous_id(sent_id):
    """Helper to find the ID of the immediately preceding sentence."""
    try:
        prefix, num = sent_id.rsplit('s', 1)
        return f"{prefix}s{int(num)-1}"
    except: 
        return None

# ============================================================================
# STEP 2 & 3: PATTERNS & SUITABLE SELECTION
# ============================================================================
print("\nSTEP 2 & 3: Extracting patterns and selecting sentences...")

attested_patterns = set()

# CRITICAL FIX: Build patterns from the FULL corpus, not just the filtered sentences
if full_corpus_lookup:
    for sent in full_corpus_lookup.values():
        if sent.root_idx is None: continue
        preverbal = sent.get_preverbal_constituents()
        if len(preverbal) < 2: continue

        deprel_sequence = tuple(w.deprel for w in preverbal)
        attested_patterns.add(deprel_sequence)
        # Only add the exact sequence, don't reverse it unless the paper explicitly says to!
        # attested_patterns.add(tuple(reversed(deprel_sequence)))
else:
    print("WARNING: Cannot build complete grammar filter without full corpus.")

suitable = []
for sent in sentences:
    preverbal = sent.get_preverbal_constituents()
    if (len(preverbal) >= 2 and sent.has_subject and sent.has_object and sent.root_word is not None):
        suitable.append(sent)

if len(suitable) > 1996:
    # Taking exactly 1,996 to match the paper's N=1996 references
    suitable.sort(key=lambda s: len(s.get_preverbal_constituents()), reverse=True)
    selected = suitable[:1996]
else:
    selected = suitable

print(f"  → Selected {len(selected):,} sentences for variant generation")

# ============================================================================
# STEP 4: LOAD SUBTREE-AWARE VARIANTS (REPLACES PREVIOUS GENERATION)
# ============================================================================
print("\nSTEP 4: Loading pre-generated subtree-aware variants...")

VARIANTS_PICKLE = "./data/processed/all_variants_final.pkl"

if not os.path.exists(VARIANTS_PICKLE):
    print(f"ERROR: {VARIANTS_PICKLE} not found! Run your subtree generator script first.")
    exit(1)

with open(VARIANTS_PICKLE, 'rb') as f:
    all_variants = pickle.load(f)

print(f"  → Loaded {len(all_variants):,} subtree-consistent variants")

# ============================================================================
# STEP 5, 6, 7: PAIRWISE MAPPING (FIXED)
# ============================================================================
print("\nSTEP 5, 6, 7: Creating pairwise dataset...")

# CRITICAL: Create lookup from ALL filtered sentences, not just the 'selected' 1996
# Load the original sentence objects so the feature extractor can read the words
with open(INPUT_FILE, 'rb') as f:
    temp_sents = pickle.load(f)
sent_lookup = {s.sent_id: s for s in temp_sents} # <--- This fixes the KeyError

variants_by_sent = defaultdict(list)
for v in all_variants:
    # Only process variants if we actually have the original sentence object
    if v['sent_id'] in sent_lookup:
        variants_by_sent[v['sent_id']].append(v)

pairs = []
for sent_id, sent_variants in variants_by_sent.items():
    reference = next((v for v in sent_variants if v['is_reference']), None)
    non_refs = [v for v in sent_variants if not v['is_reference']]
    
    if reference is None or not non_refs:
        continue
    
    for variant in non_refs:
        # A vs B
        pairs.append({'sent_id': sent_id, 'sentence_a': reference, 'sentence_b': variant, 'label': 1, 'pair_type': 'ref-var'})
        # B vs A
        pairs.append({'sent_id': sent_id, 'sentence_a': variant, 'sentence_b': reference, 'label': 0, 'pair_type': 'var-ref'})

print(f"  → Created {len(pairs):,} pairwise comparisons.")

# ============================================================================
# STEP 8, 9 & 10: EXTRACT FEATURES (WITH CONTEXT AND FULL ORDERS)
# ============================================================================
print("\nSTEP 8 & 9 & 10: Extracting pairwise features (Order-Aware)...")

pair_features = []

for pair in tqdm(pairs, desc="Pairs Extraction"):
    sent_id = pair['sent_id']
    original_sent = sent_lookup[sent_id]
    
    # 1. Get Discourse Context (the sentence immediately preceding this one)
    prev_id = get_previous_id(sent_id)
    context = full_corpus_lookup.get(prev_id)

    # 2. Extract features for Sentence A (Using its specific word order)
    feat_a = extract_features_for_sentence(
        original_sent, 
        word_order=pair['sentence_a']['variant_order'], 
        context_sentence=context
    )
    
    # 3. Extract features for Sentence B (Using its specific word order)
    feat_b = extract_features_for_sentence(
        original_sent, 
        word_order=pair['sentence_b']['variant_order'], 
        context_sentence=context
    )
    
    # 4. Calculate Differences
    pair_feat = {
        'sent_id': sent_id,
        'label': pair['label'],
        'pair_type': pair['pair_type'],
        'dep_len_diff': feat_a['dep_len_temperley'] - feat_b['dep_len_temperley'],
        'info_status_diff': feat_a['info_status_score'] - feat_b['info_status_score'],
        'dep_len_a': feat_a['dep_len_temperley'],
        'dep_len_b': feat_b['dep_len_temperley'],
        'info_status_a': feat_a['info_status_score'],
        'info_status_b': feat_b['info_status_score']
    }
    
    pair_features.append(pair_feat)

# ============================================================================
# STEP 11: SAVE FINAL CSV
# ============================================================================
print("\nSTEP 11: Saving feature files...")

os.makedirs(OUTPUT_DIR_FEATURES, exist_ok=True)

df_pairs = pd.DataFrame(pair_features)
df_pairs.to_csv(os.path.join(OUTPUT_DIR_FEATURES, "pairwise_features.csv"), index=False)
with open(os.path.join(OUTPUT_DIR_FEATURES, "pairwise_features.pkl"), 'wb') as f:
    pickle.dump(df_pairs, f)

print(f"  → Saved pairwise_features.csv ({len(df_pairs):,} rows)")

print("\n" + "="*70)
print(" VERIFICATION: FIRST 5 ROWS OF PAIRWISE DIFFERENCES")
print("="*70)
print(df_pairs[['sent_id', 'pair_type', 'dep_len_diff', 'info_status_diff']].head())
print("="*70 + "\n")