#!/usr/bin/env python3
"""
Script 20: Multicollinearity Check & Centered Interaction Re-analysis

MTP-II: The raw interaction terms (valence_x_adaptive, arousal_x_adaptive) can be
highly collinear with the adaptive LSTM feature, inflating their VIF and making
coefficients hard to trust. This script:

  1. Reports VIF for all Model C features (raw interactions).
  2. Re-builds the interactions using MEAN-CENTERED components, which removes
     the structural collinearity between an interaction and its parent terms.
  3. Re-fits Model C with centered interactions and reports whether the
     valence/arousal modulation of discourse priming survives.

Centering is the standard remedy (Aiken & West, 1991): X*Z is correlated with X
and Z, but (X - mean(X)) * (Z - mean(Z)) usually is not.

IMPORTANT: Run this in the `digit` conda env (Pandas 3.0.1) so the pickle loads.

Input:
  data/features/all_features_with_emotion.pkl
Output:
  data/results/vif_emotion_analysis.csv
  Console report.
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================================
# DYNAMIC PATH RESOLUTION
# ============================================================================

def find_base():
    d = os.path.dirname(os.path.abspath(__file__))
    while d and d != os.path.dirname(d):
        if os.path.basename(d) == "hindi-world-order-replication":
            return d
        if os.path.isdir(os.path.join(d, "hindi-world-order-replication")):
            return os.path.join(d, "hindi-world-order-replication")
        d = os.path.dirname(d)
    return "."

BASE = find_base()
FEATURES_IN = os.path.join(BASE, "data", "features", "all_features_with_emotion.pkl")
RESULTS_OUT = os.path.join(BASE, "data", "results", "vif_emotion_analysis.csv")

BASE_FEATURES = [
    "dep_len_diff", "info_status_diff", "trigram_surprisal_diff",
    "lstm_surprisal_diff", "adaptive_surprisal_diff",
    "lex_rept_surprisal_diff", "pcfg_surprisal_diff",
]


def report_vif(df, feats, title):
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)
    print(" VIF > 10 indicates multicollinearity\n")
    X = df[feats].values
    rows = []
    for i, name in enumerate(feats):
        vif = variance_inflation_factor(X, i)
        flag = "  <-- HIGH" if vif > 10 else ""
        print(f"  {name:>24s}: {vif:8.2f}{flag}")
        rows.append({"feature": name, "vif": round(vif, 2), "high": vif > 10})
    return rows


def fit_and_report(df, feats, interaction_names, title):
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)
    y = df["label"].values.astype(int)
    X = sm.add_constant(df[feats].values)
    model = sm.Logit(y, X).fit(disp=0)
    names = ["const"] + feats
    out = []
    for i, nm in enumerate(names):
        if nm in interaction_names:
            b, t, pv = model.params[i], model.tvalues[i], model.pvalues[i]
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"  {nm:>16s}: beta={b:+.4f}  t={t:+.2f}  p={pv:.4g} {sig}")
            out.append({"term": nm, "beta": round(b, 4),
                        "t": round(t, 2), "p": pv, "sig": sig})
    return out


def main():
    print("\n" + "=" * 70)
    print(" SCRIPT 20: MULTICOLLINEARITY & CENTERED INTERACTION ANALYSIS")
    print("=" * 70)

    if not os.path.exists(FEATURES_IN):
        print(f"\nERROR: {FEATURES_IN} not found. Run Script 17 first.")
        sys.exit(1)

    df = pd.read_pickle(FEATURES_IN)
    print(f"\nLoaded {len(df):,} rows")

    # ---- 1. VIF with RAW interactions ----
    raw_feats = BASE_FEATURES + ["prev_valence", "prev_arousal",
                                 "valence_x_adaptive", "arousal_x_adaptive"]
    raw_vif = report_vif(df, raw_feats, "VIF WITH RAW INTERACTIONS")

    # ---- 2. Build CENTERED interactions ----
    print("\n" + "-" * 70)
    print(" Building mean-centered interaction terms...")
    print("-" * 70)
    adaptive_c = df["adaptive_surprisal_diff"] - df["adaptive_surprisal_diff"].mean()
    valence_c = df["prev_valence"] - df["prev_valence"].mean()
    arousal_c = df["prev_arousal"] - df["prev_arousal"].mean()

    df["val_x_adp_c"] = valence_c * adaptive_c
    df["aro_x_adp_c"] = arousal_c * adaptive_c
    print("  Created: val_x_adp_c, aro_x_adp_c")

    # ---- 3. VIF with CENTERED interactions ----
    cen_feats = BASE_FEATURES + ["prev_valence", "prev_arousal",
                                 "val_x_adp_c", "aro_x_adp_c"]
    cen_vif = report_vif(df, cen_feats, "VIF WITH CENTERED INTERACTIONS")

    # ---- 4. Re-fit Model C with centered interactions ----
    coefs = fit_and_report(
        df, cen_feats,
        ["prev_valence", "prev_arousal", "val_x_adp_c", "aro_x_adp_c"],
        "MODEL C COEFFICIENTS (centered interactions)")

    # ---- 5. Save VIF table ----
    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    vdf = pd.DataFrame([
        {"feature": r["feature"], "vif_raw": r["vif"]} for r in raw_vif
    ])
    cdf = pd.DataFrame([
        {"feature": r["feature"], "vif_centered": r["vif"]} for r in cen_vif
    ])
    merged = pd.merge(vdf, cdf, on="feature", how="outer")
    merged.to_csv(RESULTS_OUT, index=False)

    # ---- 6. Interpretation ----
    print("\n" + "=" * 70)
    print(" INTERPRETATION")
    print("=" * 70)
    print("""
  - Emotion MAIN effects (prev_valence, prev_arousal) are expected to be ~0:
    they are constant across a sentence's variants, so they cannot by
    themselves distinguish reference from variant in the pairwise setup.

  - The INTERACTION terms test whether emotion MODULATES discourse priming
    (adaptive LSTM surprisal). After centering, their VIF should drop sharply,
    making the coefficients trustworthy.

  - A surviving significant valence x adaptive (or arousal x adaptive) term
    means: the strength of discourse priming on Hindi word order depends on
    the emotional content of the preceding sentence. This is the core MTP-II
    finding, and connects directly to the project title.
""")
    print(f"  Saved VIF table -> {RESULTS_OUT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()