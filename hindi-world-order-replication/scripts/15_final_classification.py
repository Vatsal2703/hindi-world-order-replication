#!/usr/bin/env python3
"""
Script 24: Final Classification — Reproducing Table 4 from the Paper

Paper setup:
  - 10-fold cross-validation, logistic regression (GLM)
  - Progressive model comparison (each row adds one feature)
  - McNemar's two-tailed test for significance between consecutive models
  - Report accuracy on full dataset, DO-fronted subset, IO-fronted subset

Our setup:
  - 10-fold CV with sklearn LogisticRegression
  - Progressive models matching paper's Table 4 rows
  - McNemar's test between consecutive models
  - Reports full dataset accuracy (DO/IO subsets require extra annotation — see note)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("\n" + "="*70)
print(" FINAL CLASSIFICATION — REPRODUCING TABLE 4")
print("="*70 + "\n")

# ============================================================================
# LOAD FINAL FEATURE DATASET
# ============================================================================
FEATURES_FILE = './data/features/all_features_final.pkl'

if not os.path.exists(FEATURES_FILE):
    print(f"ERROR: {FEATURES_FILE} not found!")
    print("Run scripts/20_merge_all_feature.py first.")
    exit(1)

try:
    df = pd.read_pickle(FEATURES_FILE)
except Exception:
    csv_file = FEATURES_FILE.replace('.pkl', '.csv')
    print(f"  Pickle load failed, falling back to CSV: {csv_file}")
    df = pd.read_csv(csv_file)
print(f"Loaded {len(df):,} pairwise rows")
print(f"Columns: {list(df.columns)}\n")

# ============================================================================
# DEFINE FEATURE SETS — strictly matching paper Table 4 row order
# Table 4: a=IS, b=dep_len, c=pcfg, d=lex_rept, e=3gram, f=lstm, g=adaptive
# base1 = a+b+c+d+e+f  (with lex-rept)
# base2 = a+b+c+e+f    (without lex-rept)
# ============================================================================
available = set(df.columns)

FEATURE_SETS = {}

# ---- Individual predictors (Table 4 order a → g) ----
if 'info_status_diff' in available:
    FEATURE_SETS['a = IS score'] = ['info_status_diff']

if 'dep_len_diff' in available:
    FEATURE_SETS['b = dep length'] = ['dep_len_diff']

if 'pcfg_surprisal_diff' in available:
    FEATURE_SETS['c = pcfg surp'] = ['pcfg_surprisal_diff']

if 'lex_rept_surprisal_diff' in available:
    FEATURE_SETS['d = lex repetition surp'] = ['lex_rept_surprisal_diff']

if 'trigram_surprisal_diff' in available:
    FEATURE_SETS['e = 3-gram surp'] = ['trigram_surprisal_diff']

if 'lstm_surprisal_diff' in available:
    FEATURE_SETS['f = lstm surp'] = ['lstm_surprisal_diff']

if 'adaptive_surprisal_diff' in available:
    FEATURE_SETS['g = adaptive lstm surp'] = ['adaptive_surprisal_diff']

# ---- Collective models (paper Table 4) ----
# base1 = a + b + c + d + e + f  (all features except adaptive)
base1_ordered = ['info_status_diff', 'dep_len_diff', 'pcfg_surprisal_diff',
                 'lex_rept_surprisal_diff', 'trigram_surprisal_diff', 'lstm_surprisal_diff']
base1_cols = [c for c in base1_ordered if c in available]
if len(base1_cols) >= 4:
    FEATURE_SETS['base1 = a+b+c+d+e+f'] = base1_cols

if 'adaptive_surprisal_diff' in available and len(base1_cols) >= 4:
    FEATURE_SETS['base1 + g'] = base1_cols + ['adaptive_surprisal_diff']

# base2 = a + b + c + e + f  (without lex-rept d)
base2_ordered = ['info_status_diff', 'dep_len_diff', 'pcfg_surprisal_diff',
                 'trigram_surprisal_diff', 'lstm_surprisal_diff']
base2_cols = [c for c in base2_ordered if c in available]
if len(base2_cols) >= 3:
    FEATURE_SETS['base2 = a+b+c+e+f'] = base2_cols

if 'adaptive_surprisal_diff' in available and len(base2_cols) >= 3:
    FEATURE_SETS['base2 + g'] = base2_cols + ['adaptive_surprisal_diff']

print(f"Feature sets to evaluate ({len(FEATURE_SETS)}):")
for name, cols in FEATURE_SETS.items():
    print(f"  {name}: {cols}")
print()

# ============================================================================
# 10-FOLD CROSS-VALIDATION
# ============================================================================
X_all = df[list(set(col for cols in FEATURE_SETS.values() for col in cols))].values
y = df['label'].values

# Drop rows with NaN in any feature
valid_mask = ~pd.DataFrame(
    df[[col for cols in FEATURE_SETS.values() for col in cols]].values
).isnull().any(axis=1).values

df_clean = df[valid_mask].reset_index(drop=True)
y_clean = df_clean['label'].values
print(f"Clean rows (no NaN): {len(df_clean):,}")

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}   # model_name -> list of per-sample predictions (in original order)
fold_accs = {} # model_name -> list of fold accuracies

# Pre-allocate prediction arrays
for name in FEATURE_SETS:
    results[name] = np.full(len(df_clean), -1, dtype=int)
    fold_accs[name] = []

print("Running 10-fold cross-validation...")
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_clean, y_clean)):
    print(f"  Fold {fold_idx+1}/10 ...", end=' ', flush=True)
    y_train = y_clean[train_idx]
    y_test  = y_clean[test_idx]

    for name, cols in FEATURE_SETS.items():
        X_train = df_clean.iloc[train_idx][cols].values
        X_test  = df_clean.iloc[test_idx][cols].values

        clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        results[name][test_idx] = preds
        fold_accs[name].append(accuracy_score(y_test, preds))

    print("done")

# ============================================================================
# ACCURACY TABLE
# ============================================================================
print("\n" + "="*70)
print(" TABLE 4 REPRODUCTION — PREDICTION ACCURACY")
print("="*70)
print(f"\n{'Model':<30} {'Accuracy':>10}  {'Paper Target':>14}")
print("-" * 60)

# Paper Table 4: Full dataset accuracy targets
paper_targets = {
    'a = IS score':            51.84,
    'b = dep length':          62.31,
    'c = pcfg surp':           86.86,
    'd = lex repetition surp': 90.07,
    'e = 3-gram surp':         91.18,
    'f = lstm surp':           94.01,
    'g = adaptive lstm surp':  94.06,
    'base1 = a+b+c+d+e+f':     95.05,
    'base1 + g':               95.06,
    'base2 = a+b+c+e+f':       95.06,
    'base2 + g':               95.09,
}

accuracy_scores = {}
for name, preds in results.items():
    acc = accuracy_score(y_clean, preds) * 100
    accuracy_scores[name] = acc
    target = paper_targets.get(name, None)
    target_str = f"{target:.2f}%" if target else "  —"
    print(f"  {name:<28} {acc:>8.2f}%  {target_str:>14}")

# ============================================================================
# McNEMAR'S SIGNIFICANCE TESTS (consecutive models)
# ============================================================================
print("\n" + "="*70)
print(" McNEMAR'S TWO-TAILED SIGNIFICANCE TESTS")
print("="*70)

model_names = list(FEATURE_SETS.keys())
for i in range(1, len(model_names)):
    prev_name = model_names[i-1]
    curr_name = model_names[i]

    prev_preds = results[prev_name]
    curr_preds = results[curr_name]

    # Contingency table for McNemar's test
    # n00: both wrong, n01: prev wrong curr right, n10: prev right curr wrong, n11: both right
    n01 = np.sum((prev_preds != y_clean) & (curr_preds == y_clean))
    n10 = np.sum((prev_preds == y_clean) & (curr_preds != y_clean))

    if n01 + n10 == 0:
        print(f"  {prev_name} vs {curr_name}: no disagreement")
        continue

    table = [[0, n01], [n10, 0]]  # McNemar only uses off-diagonal
    result = mcnemar([[n01+n10-n01, n01], [n10, 0]], exact=False, correction=True)
    p = result.pvalue
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    acc_diff = accuracy_scores[curr_name] - accuracy_scores[prev_name]
    print(f"  {prev_name} → {curr_name}:")
    print(f"    Δacc={acc_diff:+.3f}%  p={p:.4f}  {sig}")

# ============================================================================
# FEATURE COEFFICIENTS 
# ============================================================================
full_model_name = 'base2 + g' if 'base2 + g' in FEATURE_SETS else (
    'base1 + g' if 'base1 + g' in FEATURE_SETS else list(FEATURE_SETS.keys())[-1]
)
full_cols = FEATURE_SETS[full_model_name]

print(f"\n" + "="*70)
print(f" FEATURE COEFFICIENTS — {full_model_name}")
print("="*70)

X_full = df_clean[full_cols].values
clf_full = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
clf_full.fit(X_full, y_clean)

print(f"\n{'Feature':<35} {'Coefficient':>12}  {'Direction'}")
print("-" * 60)
for col, coef in zip(full_cols, clf_full.coef_[0]):
    direction = "↓ lower in ref" if coef < 0 else "↑ higher in ref"
    print(f"  {col:<33} {coef:>12.4f}  {direction}")

print(f"\n  Interpretation:")
print(f"  Negative coef → feature tends to be LOWER in reference sentences")
print(f"  (surprisal: reference is more predictable; DL: reference may be longer)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
os.makedirs('./data/results', exist_ok=True)

results_df = pd.DataFrame({
    'model': list(accuracy_scores.keys()),
    'accuracy': list(accuracy_scores.values()),
    'paper_target': [paper_targets.get(k, None) for k in accuracy_scores.keys()]
})
results_df.to_csv('./data/results/classification_results.csv', index=False)
print(f"\nResults saved to: data/results/classification_results.csv")

print("\n" + "="*70)
print(" DONE")
print("="*70 + "\n")
