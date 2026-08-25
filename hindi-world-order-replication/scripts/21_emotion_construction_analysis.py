#!/usr/bin/env python3
"""
Script 21: Construction-Type Reconstruction & DO/IO Emotion Analysis (H3)

MTP-II: Tests whether the DO/IO construction asymmetry found in MTP-I interacts
with emotion. In MTP-I, adaptive LSTM surprisal (discourse priming) was
significant for IO-fronted but NOT DO-fronted constructions. Here we ask:
does the EMOTION x priming interaction also differ by construction type?

The features file lacks a construction_type column, so we first reconstruct it
from the reference sentences' dependency parse, then merge by sent_id.

Construction type (from preverbal order of subject vs objects in the reference):
  DOSV : direct object (obj)  appears before the subject (DO-fronted)
  IOSV : indirect object (iobj) appears before the subject (IO-fronted)
  SOV  : subject first (canonical)

For each construction subset we fit a focused logistic model with mean-centered
interactions:
    label ~ adaptive + valence*adaptive + arousal*adaptive
and report whether emotion modulates priming differently across constructions.

RUN IN THE `digit` CONDA ENV (Pandas 3.0.1) so the pickles load.

Input:
  data/processed/replication_filtered_sentences.pkl
  data/features/all_features_with_emotion.pkl
Output:
  data/features/all_features_with_emotion.pkl   (updated, + construction_type)
  data/features/all_features_with_emotion.csv
  data/results/construction_emotion_analysis.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from utils.paths import find_base
from utils.emotion import get_construction_type, sig_mark

# ============================================================================
# PATHS
# ============================================================================

BASE = find_base(__file__)
SENTENCES_IN = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
FEATURES_IO = os.path.join(BASE, "data", "features", "all_features_with_emotion.pkl")
RESULTS_OUT = os.path.join(BASE, "data", "results", "construction_emotion_analysis.csv")

try:
    from parsers.ud_parser import Sentence, Word
except ImportError:
    pass


def fit_focused(subset, label_col="label"):
    """
    Fit label ~ adaptive + val_x_adp_c + aro_x_adp_c (centered) on a subset.
    Returns dict of coefficient stats, or None if too small / fails.
    """
    needed = ["adaptive_surprisal_diff", "prev_valence", "prev_arousal", label_col]
    if any(c not in subset.columns for c in needed):
        return None
    if len(subset) < 50:
        return None

    adp = subset["adaptive_surprisal_diff"].astype(float)
    val = subset["prev_valence"].astype(float)
    aro = subset["prev_arousal"].astype(float)

    adp_c = adp - adp.mean()
    val_c = val - val.mean()
    aro_c = aro - aro.mean()

    X = pd.DataFrame({
        "adaptive": adp_c,
        "val_x_adp": val_c * adp_c,
        "aro_x_adp": aro_c * adp_c,
    })
    y = subset[label_col].astype(int).values

    # Need both label classes present
    if len(np.unique(y)) < 2:
        return None

    try:
        model = sm.Logit(y, sm.add_constant(X.values)).fit(disp=0)
    except Exception:
        return None

    names = ["const", "adaptive", "val_x_adp", "aro_x_adp"]
    out = {}
    for i, nm in enumerate(names):
        out[nm] = (model.params[i], model.tvalues[i], model.pvalues[i])
    return out


def main():
    print("\n" + "=" * 70)
    print(" SCRIPT 21: CONSTRUCTION-TYPE + DO/IO EMOTION ANALYSIS (H3)")
    print("=" * 70)

    # ---- Load reference sentences ----
    if not os.path.exists(SENTENCES_IN):
        print(f"\nERROR: {SENTENCES_IN} not found.")
        sys.exit(1)
    with open(SENTENCES_IN, "rb") as f:
        sentences = pickle.load(f)
    print(f"\nLoaded {len(sentences):,} reference sentences")

    # ---- Compute construction type per sent_id ----
    print("Reconstructing construction types...")
    construction = {}
    for s in sentences:
        construction[s.sent_id] = get_construction_type(s)

    dist = pd.Series(list(construction.values())).value_counts()
    total = len(construction)
    print("\n  Construction distribution (reference sentences):")
    for ctype in ["SOV", "DOSV", "IOSV", "UNKNOWN"]:
        n = dist.get(ctype, 0)
        print(f"    {ctype:>8s}: {n:>5,} ({100*n/total:.1f}%)")
    print("    (MTP-I reference: SOV 91.8%, DOSV 5.3%, IOSV 3.0%)")

    # ---- Load features and merge construction type ----
    if not os.path.exists(FEATURES_IO):
        print(f"\nERROR: {FEATURES_IO} not found. Run Script 17 first.")
        sys.exit(1)
    df = pd.read_pickle(FEATURES_IO)
    print(f"\nLoaded features: {len(df):,} rows")

    df["construction_type"] = df["sent_id"].map(construction).fillna("UNKNOWN")

    row_dist = df["construction_type"].value_counts()
    print("\n  Pairwise instances per construction:")
    for ctype in ["SOV", "DOSV", "IOSV", "UNKNOWN"]:
        n = row_dist.get(ctype, 0)
        print(f"    {ctype:>8s}: {n:>7,} rows")

    # Save the updated features (now with construction_type)
    df.to_pickle(FEATURES_IO)
    df.to_csv(FEATURES_IO.replace(".pkl", ".csv"), index=False)
    print(f"\n  Updated features saved (now includes construction_type)")

    # ---- Per-construction emotion-modulation analysis ----
    print("\n" + "=" * 70)
    print(" EMOTION x PRIMING BY CONSTRUCTION (centered interactions)")
    print("=" * 70)

    results = []
    for ctype in ["SOV", "DOSV", "IOSV"]:
        subset = df[df["construction_type"] == ctype]
        print("\n" + "-" * 70)
        print(f" {ctype}  (n = {len(subset):,} pairwise rows)")
        print("-" * 70)

        fit = fit_focused(subset)
        if fit is None:
            print("  (insufficient data or fit failed)")
            results.append({"construction": ctype, "n": len(subset),
                            "adaptive_t": None, "val_x_adp_t": None,
                            "aro_x_adp_t": None})
            continue

        for term in ["adaptive", "val_x_adp", "aro_x_adp"]:
            b, t, p = fit[term]
            print(f"  {term:>12s}: beta={b:+.4f}  t={t:+.2f}  p={p:.4g} {sig_mark(p)}")

        results.append({
            "construction": ctype,
            "n": len(subset),
            "adaptive_beta": round(fit["adaptive"][0], 4),
            "adaptive_t": round(fit["adaptive"][1], 2),
            "adaptive_p": fit["adaptive"][2],
            "val_x_adp_t": round(fit["val_x_adp"][1], 2),
            "val_x_adp_p": fit["val_x_adp"][2],
            "aro_x_adp_t": round(fit["aro_x_adp"][1], 2),
            "aro_x_adp_p": fit["aro_x_adp"][2],
        })

    pd.DataFrame(results).to_csv(RESULTS_OUT, index=False)

    print("\n" + "=" * 70)
    print(" INTERPRETATION")
    print("=" * 70)
    print("""
  Compare the emotion x adaptive interactions across constructions:

  - If val_x_adp / aro_x_adp are significant for IO-fronted (IOSV) but not
    DO-fronted (DOSV), the emotion modulation MIRRORS the MTP-I asymmetry
    (where adaptive priming mattered for IO, not DO). That would mean emotion
    tunes priming specifically where priming already operates.

  - Note the small IO/DO sample sizes -- treat these subset results as
    suggestive, with the full-dataset interaction (Script 18/20) as the
    primary, well-powered finding.
""")
    print(f"  Saved -> {RESULTS_OUT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()