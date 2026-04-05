#!/usr/bin/env python3
"""
Script 20: Merge ALL Features into Final Dataset

Merges 6 feature sources, all aligned to all_variants_final.pkl order:
  1. pairwise_features_trigram_blind.pkl  -> dep_len_diff, info_status_diff, trigram_surprisal_diff
  2. final_scored_variants.pkl            -> vanilla LSTM surprisal (per variant, zip-aligned)
  3. adaptive_lstm_scores.pkl             -> adaptive LSTM surprisal (per variant, zip-aligned)

Pairwise construction (Joachims method):
  For each sentence: (ref, var) -> label=1, (var, ref) -> label=0
  All feature values are differences: ref_value - var_value

Output: data/features/all_features_final.pkl / .csv
"""

import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("\n" + "="*70)
print(" MERGING ALL FEATURES")
print("="*70 + "\n")

# ============================================================================
# STEP 1: Load all_variants_final (the anchor - everything aligns to this)
# ============================================================================
print("STEP 1: Loading all_variants_final (anchor)...")
with open('./data/processed/all_variants_final.pkl', 'rb') as f:
    all_variants = pickle.load(f)
print(f"  {len(all_variants):,} total variants")

# ============================================================================
# STEP 2: Load vanilla LSTM scores (same order as all_variants_final)
# ============================================================================
print("\nSTEP 2: Loading vanilla LSTM scores...")
with open('./data/results/final_scored_variants.pkl', 'rb') as f:
    lstm_scores = pickle.load(f)
assert len(lstm_scores) == len(all_variants), \
    f"LSTM scores length {len(lstm_scores)} != variants {len(all_variants)}"
print(f"  {len(lstm_scores):,} LSTM scores (aligned)")

# ============================================================================
# STEP 3: Load adaptive LSTM scores (same order as all_variants_final)
# ============================================================================
ADAPTIVE_PATH  = './data/results/adaptive_lstm_scores.pkl'
LEX_REPT_PATH  = './data/results/lex_rept_scores.pkl'
has_adaptive = os.path.exists(ADAPTIVE_PATH)
if has_adaptive:
    print("\nSTEP 3: Loading adaptive LSTM scores...")
    with open(ADAPTIVE_PATH, 'rb') as f:
        adaptive_scores = pickle.load(f)
    assert len(adaptive_scores) == len(all_variants), \
        f"Adaptive scores length {len(adaptive_scores)} != variants {len(all_variants)}"
    print(f"  {len(adaptive_scores):,} adaptive LSTM scores (aligned)")
else:
    print("\nSTEP 3: adaptive_lstm_scores.pkl not found — skipping adaptive feature.")
    print("         Run script 16 to generate it.")
    adaptive_scores = None

LEX_REPT_PATH = './data/results/lex_rept_scores.pkl'
has_lex_rept = os.path.exists(LEX_REPT_PATH)
if has_lex_rept:
    print("\nSTEP 3b: Loading lexical repetition surprisal scores...")
    with open(LEX_REPT_PATH, 'rb') as f:
        lex_rept_scores = pickle.load(f)
    assert len(lex_rept_scores) == len(all_variants), \
        f"Lex-rept scores length {len(lex_rept_scores)} != variants {len(all_variants)}"
    print(f"  {len(lex_rept_scores):,} lex-rept scores (aligned)")
else:
    print("\nSTEP 3b: lex_rept_scores.pkl not found — skipping lex-rept feature.")
    print("          Run script 21 to generate it.")
    lex_rept_scores = None

PCFG_PATH = './data/results/pcfg_scores.pkl'
has_pcfg = os.path.exists(PCFG_PATH)
if has_pcfg:
    print("\nSTEP 3c: Loading PCFG surprisal scores...")
    with open(PCFG_PATH, 'rb') as f:
        pcfg_scores = pickle.load(f)
    assert len(pcfg_scores) == len(all_variants), \
        f"PCFG scores length {len(pcfg_scores)} != variants {len(all_variants)}"
    print(f"  {len(pcfg_scores):,} PCFG scores (aligned)")
else:
    print("\nSTEP 3c: pcfg_scores.pkl not found — skipping PCFG feature.")
    print("          Run script 22 to generate it.")
    pcfg_scores = None

# ============================================================================
# STEP 4: Load trigram features (pairwise, already has dep_len + IS + trigram)
# ============================================================================
print("\nSTEP 4: Loading trigram pairwise features (DL + IS + Trigram)...")
with open('./data/features/pairwise_features_trigram_blind.pkl', 'rb') as f:
    trigram_df = pickle.load(f)
if isinstance(trigram_df, list):
    trigram_df = pd.DataFrame(trigram_df)
print(f"  {len(trigram_df):,} pairwise rows")
print(f"  Columns: {list(trigram_df.columns)}")

# ============================================================================
# STEP 5: Load reference sentences for word form lookup
# ============================================================================
print("\nSTEP 5: Loading reference sentences...")
with open('./data/processed/reference_sentences.pkl', 'rb') as f:
    ref_sents = pickle.load(f)
sent_lookup = {s.sent_id: s for s in ref_sents}
print(f"  {len(sent_lookup):,} reference sentences loaded")

# ============================================================================
# STEP 6: Build per-variant lookup with ALL scores
# ============================================================================
print("\nSTEP 6: Attaching scores to variants...")

# Attach vanilla LSTM, adaptive LSTM, lex-rept, and PCFG scores to each variant
variants_with_scores = []
for i, (variant, lstm_row) in enumerate(zip(all_variants, lstm_scores)):
    entry = {
        'sent_id': variant['sent_id'],
        'variant_order': variant['variant_order'],
        'is_reference': variant['is_reference'],
        'lstm_surprisal': lstm_row['avg_surprisal'],
    }
    if adaptive_scores is not None:
        entry['adaptive_surprisal'] = adaptive_scores[i]['adaptive_surprisal']
    if lex_rept_scores is not None:
        entry['lex_rept_surprisal'] = lex_rept_scores[i]['lex_rept_surprisal']
    if pcfg_scores is not None:
        entry['pcfg_surprisal'] = pcfg_scores[i]['pcfg_surprisal']
    variants_with_scores.append(entry)

# Group by sent_id
by_sent = defaultdict(list)
for v in variants_with_scores:
    by_sent[v['sent_id']].append(v)

print(f"  {len(by_sent):,} unique sentences with scores")

# ============================================================================
# STEP 7: Build trigram pairwise lookup keyed by sent_id
# Trigram features are already pairwise — we need to match them up.
# We rebuild pairwise from scratch using the variant scores to stay consistent.
# ============================================================================

# Build trigram surprisal lookup: sent_id -> variant_order (as tuple) -> surprisal
# We need to re-score each variant_order with trigram to get the diff
# Instead: load trigram model and recompute (or reuse pairwise_features_trigram_blind
# which already has surprisal_diff, dep_len_diff, info_status_diff)

# The trigram pairwise file was built from the same all_variants_final.pkl
# so it has the same (ref, var) pairs in the same order.
# Structure: each sentence contributes 2*(num_variants-1) rows (label=1 and label=0)
# We use the trigram pairwise as the base and JOIN the LSTM diffs.

# Rebuild LSTM pairwise from scratch to guarantee alignment
print("\nSTEP 7: Building LSTM pairwise differences...")

lstm_pairs = []
for sent_id, variants in tqdm(by_sent.items(), desc="Building LSTM pairs"):
    ref = next((v for v in variants if v['is_reference']), None)
    non_refs = [v for v in variants if not v['is_reference']]
    if ref is None or not non_refs:
        continue

    ref_lstm  = ref['lstm_surprisal']
    ref_adapt = ref.get('adaptive_surprisal', 0.0)
    ref_pcfg  = ref.get('pcfg_surprisal', None)

    for var in non_refs:
        var_lstm  = var['lstm_surprisal']
        var_adapt = var.get('adaptive_surprisal', 0.0)
        var_pcfg  = var.get('pcfg_surprisal', None)

        ref_lex  = ref.get('lex_rept_surprisal', None)
        var_lex  = var.get('lex_rept_surprisal', None)

        # Label=1: ref is preferred (ref - var)
        row1 = {
            'sent_id': sent_id,
            'label': 1,
            'lstm_surprisal_diff': ref_lstm - var_lstm,
        }
        if adaptive_scores is not None:
            row1['adaptive_surprisal_diff'] = ref_adapt - var_adapt
        if lex_rept_scores is not None:
            row1['lex_rept_surprisal_diff'] = ref_lex - var_lex
        if pcfg_scores is not None:
            row1['pcfg_surprisal_diff'] = ref_pcfg - var_pcfg
        lstm_pairs.append(row1)

        # Label=0: var is preferred (var - ref)
        row0 = {
            'sent_id': sent_id,
            'label': 0,
            'lstm_surprisal_diff': var_lstm - ref_lstm,
        }
        if adaptive_scores is not None:
            row0['adaptive_surprisal_diff'] = var_adapt - ref_adapt
        if lex_rept_scores is not None:
            row0['lex_rept_surprisal_diff'] = var_lex - ref_lex
        if pcfg_scores is not None:
            row0['pcfg_surprisal_diff'] = var_pcfg - ref_pcfg
        lstm_pairs.append(row0)

df_lstm = pd.DataFrame(lstm_pairs)
print(f"  {len(df_lstm):,} LSTM pairwise rows")

# ============================================================================
# STEP 8: Merge trigram pairwise + LSTM pairwise on sent_id
# Both have same (sent_id, label=1 then label=0) structure for same sentence set
# Use an index-based merge within each sent_id group
# ============================================================================
print("\nSTEP 8: Merging trigram features with LSTM features...")

# Rename trigram surprisal column for clarity
trigram_df = trigram_df.rename(columns={'surprisal_diff': 'trigram_surprisal_diff'})

# Check if sent_id merge is clean
trig_sents = set(trigram_df['sent_id'].unique())
lstm_sents = set(df_lstm['sent_id'].unique())
common = trig_sents & lstm_sents
print(f"  Trigram sentences: {len(trig_sents):,}")
print(f"  LSTM sentences:    {len(lstm_sents):,}")
print(f"  Common:            {len(common):,}")

# Merge on (sent_id, label) — but each sent_id can have multiple pairs
# We need positional merge within each sent_id group
# Sort both by sent_id, then by label (1 first, 0 second) to align
trigram_df_sorted = trigram_df.sort_values(['sent_id', 'label'], ascending=[True, False]).reset_index(drop=True)
df_lstm_sorted = df_lstm.sort_values(['sent_id', 'label'], ascending=[True, False]).reset_index(drop=True)

# Keep only common sent_ids
trigram_filtered = trigram_df_sorted[trigram_df_sorted['sent_id'].isin(common)].reset_index(drop=True)
lstm_filtered = df_lstm_sorted[df_lstm_sorted['sent_id'].isin(common)].reset_index(drop=True)

if len(trigram_filtered) == len(lstm_filtered):
    print(f"  Row counts match: {len(trigram_filtered):,} rows — merging by position")
    lstm_cols = ['lstm_surprisal_diff']
    if 'adaptive_surprisal_diff' in lstm_filtered.columns:
        lstm_cols.append('adaptive_surprisal_diff')
    if 'lex_rept_surprisal_diff' in lstm_filtered.columns:
        lstm_cols.append('lex_rept_surprisal_diff')
    if 'pcfg_surprisal_diff' in lstm_filtered.columns:
        lstm_cols.append('pcfg_surprisal_diff')
    final = pd.concat([
        trigram_filtered.reset_index(drop=True),
        lstm_filtered[lstm_cols].reset_index(drop=True)
    ], axis=1)
else:
    print(f"  Row count mismatch: trigram={len(trigram_filtered):,}, lstm={len(lstm_filtered):,}")
    print("  Falling back to sent_id merge (may reduce rows)...")
    # Use a merge key: sent_id + within-group row number
    trigram_filtered['_pair_rank'] = trigram_filtered.groupby(['sent_id', 'label']).cumcount()
    lstm_filtered['_pair_rank'] = lstm_filtered.groupby(['sent_id', 'label']).cumcount()
    extra_cols = ['lstm_surprisal_diff']
    if 'adaptive_surprisal_diff' in lstm_filtered.columns:
        extra_cols.append('adaptive_surprisal_diff')
    if 'lex_rept_surprisal_diff' in lstm_filtered.columns:
        extra_cols.append('lex_rept_surprisal_diff')
    if 'pcfg_surprisal_diff' in lstm_filtered.columns:
        extra_cols.append('pcfg_surprisal_diff')
    final = pd.merge(
        trigram_filtered,
        lstm_filtered[['sent_id', 'label', '_pair_rank'] + extra_cols],
        on=['sent_id', 'label', '_pair_rank'],
        how='inner'
    ).drop(columns=['_pair_rank'])

print(f"  Merged: {len(final):,} rows")

# ============================================================================
# STEP 9: Validate
# ============================================================================
print("\nSTEP 9: Validating...")

feature_cols = [c for c in final.columns if 'diff' in c.lower() or 'surp' in c.lower()]
print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

missing = final[feature_cols].isnull().sum()
if missing.sum() > 0:
    print(f"  Missing values found, dropping rows:")
    print(missing[missing > 0])
    final = final.dropna(subset=feature_cols)

# Ensure label column is clean
if 'label' not in final.columns:
    print("ERROR: No label column found!")
    exit(1)

print(f"  Label distribution:")
print(f"    label=1 (ref preferred): {(final['label']==1).sum():,}")
print(f"    label=0 (var preferred): {(final['label']==0).sum():,}")

# ============================================================================
# STEP 10: Save
# ============================================================================
print("\nSTEP 10: Saving final dataset...")

os.makedirs('./data/features', exist_ok=True)
final.to_pickle('./data/features/all_features_final.pkl')
final.to_csv('./data/features/all_features_final.csv', index=False)

print(f"\n  Saved: data/features/all_features_final.pkl")
print(f"  Saved: data/features/all_features_final.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print(" FINAL DATASET SUMMARY")
print("="*70)
print(f"  Total rows:    {len(final):,}")
print(f"  Total columns: {len(final.columns)}")
print(f"\n  Feature statistics:")
for col in feature_cols:
    print(f"\n  {col}:")
    print(f"    mean={final[col].mean():.4f}  std={final[col].std():.4f}  "
          f"min={final[col].min():.4f}  max={final[col].max():.4f}")

has_adaptive = 'adaptive_surprisal_diff' in final.columns
has_pcfg     = 'pcfg_surprisal_diff' in final.columns
print(f"\n  Features present (paper Table 4 ordering):")
print(f"    a  info_status_diff:          {'✅' if 'info_status_diff' in final.columns else '❌'}")
print(f"    b  dep_len_diff:              {'✅' if 'dep_len_diff' in final.columns else '❌'}")
print(f"    c  pcfg_surprisal_diff:       {'✅' if has_pcfg else '❌ (run script 22 first)'}")
print(f"    d  lex_rept_surprisal_diff:  {'✅' if has_lex_rept else '❌ (run script 21 first)'}")
print(f"    e  trigram_surprisal_diff:   {'✅' if 'trigram_surprisal_diff' in final.columns else '❌'}")
print(f"    f  lstm_surprisal_diff:      {'✅' if 'lstm_surprisal_diff' in final.columns else '❌'}")
print(f"    g  adaptive_surprisal_diff:  {'✅' if has_adaptive else '❌ (run script 16 first)'}")

print("\n" + "="*70)
print(" Next steps:")
missing_steps = []
if not has_adaptive:
    missing_steps.append("Run scripts/16_workflow_LSTM.py  (adaptive LSTM surprisal, lr=2)")
if not has_lex_rept:
    missing_steps.append("Run scripts/21_lex_rept_surprisal.py  (lexical repetition surprisal)")
if not has_pcfg:
    missing_steps.append("Run scripts/22_pcfg_surprisal.py  (PCFG surprisal — Berkeley parser)")
if missing_steps:
    for i, step in enumerate(missing_steps, 1):
        print(f"  {i}. {step}")
    n = len(missing_steps) + 1
    print(f"  {n}. Re-run this script")
    print(f"  {n+1}. Run scripts/24_final_classification.py")
else:
    print("  All 7 features present! (matches paper Table 4 a-g)")
    print("  → Run scripts/24_final_classification.py")
print("="*70 + "\n")
