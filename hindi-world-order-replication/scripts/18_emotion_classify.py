#!/usr/bin/env python3
"""
Script 18: Classification with Emotion Features

MTP-II: Tests whether emotion (valence, arousal) of the preceding sentence
improves prediction of Hindi word order, using nested logistic regression
models compared with McNemar's test -- the same methodology as MTP-I.

Models:
  A (baseline) : 7 MTP-I cognitive features              -> reference 94.62%
  B            : A + prev_valence + prev_arousal          -> tests H1
  C            : B + valence_x_adaptive + arousal_x_adaptive -> tests H2/H3

Also runs the DO/IO subset analysis: splits IO-fronted sentences by high vs
low arousal of the preceding context and compares the adaptive LSTM effect.

Input:
  ./data/features/all_features_with_emotion.pkl   (Script 17 output)

Output:
  ./data/results/emotion_model_comparison.csv
  Console report with accuracies, McNemar p-values, and coefficients.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))
from utils.paths import find_base

BASE = find_base(__file__)
FEATURES_IN = os.path.join(BASE, "data", "features", "all_features_with_emotion.pkl")
RESULTS_OUT = os.path.join(BASE, "data", "results", "emotion_model_comparison.csv")

# The 7 MTP-I baseline features (difference features).
# Adjust names here if your columns differ.
BASE_FEATURES = [
    "dep_len_diff",
    "info_status_diff",
    "trigram_surprisal_diff",
    "lstm_surprisal_diff",
    "adaptive_surprisal_diff",
    "lex_rept_surprisal_diff",
    "pcfg_surprisal_diff",
]

EMOTION_MAIN = ["prev_valence", "prev_arousal"]
EMOTION_INTERACT = ["valence_x_adaptive", "arousal_x_adaptive"]

LABEL_COL = "label"            # 1 = reference preferred, 0 = variant
CONSTRUCTION_COL = "construction_type"  # SOV / DOSV / IOSV
SENT_ID_COL = "sent_id"


def resolve_features(df, wanted):
    """Keep only feature names that exist in the dataframe; warn about missing."""
    present = [f for f in wanted if f in df.columns]
    missing = [f for f in wanted if f not in df.columns]
    if missing:
        print(f"  NOTE: missing columns ignored: {missing}")
    return present


def cv_predictions(X, y, n_splits=10, seed=42):
    """Return out-of-fold predictions for honest accuracy + McNemar."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    return preds


def mcnemar_test(y_true, pred1, pred2):
    """McNemar's test between two models' predictions."""
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)
    # contingency: model1 correct vs model2 correct
    n01 = np.sum(~correct1 & correct2)  # m1 wrong, m2 right
    n10 = np.sum(correct1 & ~correct2)  # m1 right, m2 wrong
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue, n01, n10


def fit_statsmodels(X, y, names):
    """Fit logistic regression with statsmodels to get coefficients + t/z-values."""
    Xc = sm.add_constant(X)
    model = sm.Logit(y, Xc).fit(disp=0)
    coefs = {}
    for i, name in enumerate(["const"] + names):
        coefs[name] = (model.params[i], model.tvalues[i], model.pvalues[i])
    return coefs


def main():
    print("\n" + "=" * 70)
    print(" SCRIPT 18: CLASSIFICATION WITH EMOTION FEATURES")
    print("=" * 70 + "\n")

    if not os.path.exists(FEATURES_IN):
        print(f"ERROR: {FEATURES_IN} not found. Run Script 17 first.")
        sys.exit(1)

    df = pd.read_pickle(FEATURES_IN)
    print(f"Loaded {len(df):,} rows")

    y = df[LABEL_COL].values.astype(int)

    feats_A = resolve_features(df, BASE_FEATURES)
    feats_B = feats_A + resolve_features(df, EMOTION_MAIN)
    feats_C = feats_B + resolve_features(df, EMOTION_INTERACT)

    print(f"\nModel A: {len(feats_A)} features")
    print(f"Model B: {len(feats_B)} features (+ emotion main effects)")
    print(f"Model C: {len(feats_C)} features (+ interactions)")

    # --- Out-of-fold predictions for each model ---
    print("\nRunning 10-fold cross-validation...")
    XA = df[feats_A].values
    XB = df[feats_B].values
    XC = df[feats_C].values

    predA = cv_predictions(XA, y)
    predB = cv_predictions(XB, y)
    predC = cv_predictions(XC, y)

    accA = (predA == y).mean()
    accB = (predB == y).mean()
    accC = (predC == y).mean()

    print("\n" + "-" * 70)
    print(" ACCURACY")
    print("-" * 70)
    print(f"  Model A (baseline)          : {accA:.4%}")
    print(f"  Model B (+ emotion)         : {accB:.4%}  (\u0394 = {accB-accA:+.4%})")
    print(f"  Model C (+ interactions)    : {accC:.4%}  (\u0394 = {accC-accB:+.4%})")

    # --- McNemar tests ---
    print("\n" + "-" * 70)
    print(" McNEMAR'S SIGNIFICANCE TESTS")
    print("-" * 70)
    pBA, n01_BA, n10_BA = mcnemar_test(y, predA, predB)
    pCB, n01_CB, n10_CB = mcnemar_test(y, predB, predC)
    print(f"  B vs A (H1: emotion adds power)     : p = {pBA:.4g}")
    print(f"  C vs B (H2/H3: interactions help)   : p = {pCB:.4g}")

    # --- Coefficients (statsmodels) for the full model C ---
    print("\n" + "-" * 70)
    print(" MODEL C COEFFICIENTS (valence, arousal, interactions)")
    print("-" * 70)
    try:
        coefs = fit_statsmodels(XC, y, feats_C)
        for name in EMOTION_MAIN + EMOTION_INTERACT:
            if name in coefs:
                b, t, pv = coefs[name]
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"  {name:>22s}: \u03B2={b:+.4f}  t={t:+.2f}  p={pv:.4g} {sig}")
    except Exception as e:
        print(f"  Could not fit statsmodels coefficients: {e}")

    # --- DO / IO subset: arousal split (H2, H3) ---
    if CONSTRUCTION_COL in df.columns and "prev_arousal" in df.columns:
        print("\n" + "-" * 70)
        print(" IO-FRONTED: HIGH vs LOW AROUSAL (H2, H3)")
        print("-" * 70)
        io = df[df[CONSTRUCTION_COL] == "IOSV"]
        if len(io) > 0 and "adaptive_surprisal_diff" in io.columns:
            for label, subset in [("High arousal (>0.6)", io[io["prev_arousal"] > 0.6]),
                                  ("Low arousal (<0.4)", io[io["prev_arousal"] < 0.4])]:
                if len(subset) > 20:
                    Xs = sm.add_constant(subset[["adaptive_surprisal_diff"]].values)
                    ys = subset[LABEL_COL].values.astype(int)
                    try:
                        m = sm.Logit(ys, Xs).fit(disp=0)
                        print(f"  {label:>22s}: n={len(subset):>4}, "
                              f"adaptive t={m.tvalues[1]:+.2f}, p={m.pvalues[1]:.4g}")
                    except Exception:
                        print(f"  {label:>22s}: n={len(subset):>4} (fit failed)")
                else:
                    print(f"  {label:>22s}: n={len(subset):>4} (too few for analysis)")

    # --- Save summary ---
    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    summary = pd.DataFrame([
        {"model": "A (baseline)", "n_features": len(feats_A), "accuracy": accA, "mcnemar_vs_prev": None},
        {"model": "B (+emotion)", "n_features": len(feats_B), "accuracy": accB, "mcnemar_vs_prev": pBA},
        {"model": "C (+interact)", "n_features": len(feats_C), "accuracy": accC, "mcnemar_vs_prev": pCB},
    ])
    summary.to_csv(RESULTS_OUT, index=False)

    print("\n" + "=" * 70)
    print(f"  Saved -> {RESULTS_OUT}")
    print("  Next: Run Script 19 (emotion_validation.py)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()