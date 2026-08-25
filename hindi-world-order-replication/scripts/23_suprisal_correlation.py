#!/usr/bin/env python3
"""
Script 22: Surprisal x Emotion Correlation Analysis  — v1.0

MTP-II follow-up (requested by supervisor): before asking whether emotion
MODULATES the surprisal effect on word order (the interaction result), ask the
simpler, more basic question:

    Are surprisal and the emotion scores directly related at all?

This script answers that at the SENTENCE level (one row per reference
sentence, not per pairwise variant), computing:

  1. Correlations between each surprisal variant and valence / arousal
  2. Valence vs arousal (a sanity check: Russell predicts near-independence)
  3. The same correlations split by construction type (SOV / DOSV / IOSV)
  4. Scatter plots with fitted trend lines

Both Pearson (linear) and Spearman (monotonic, rank-based) are reported:
if they disagree markedly the relationship is non-linear.

RUN IN `digit` CONDA ENV (pandas 3.0.1 — the version that wrote the pickles).

Input : data/features/all_features_with_emotion.pkl
Output: data/results/surprisal_emotion_correlations.csv
        data/results/surprisal_emotion_scatter.png
"""

VERSION = "1.1"

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from utils.paths import find_base

# ============================================================================
# PATHS
# ============================================================================

BASE        = find_base(__file__)
FEATURES_IN = os.path.join(BASE, "data", "features", "all_features_with_emotion.pkl")
CSV_OUT     = os.path.join(BASE, "data", "results", "surprisal_emotion_correlations.csv")
PLOT_OUT    = os.path.join(BASE, "data", "results", "surprisal_emotion_scatter.png")

# Surprisal variants to test (only those present in the file are used)
SURPRISAL_COLS = [
    ("adaptive_surprisal_diff", "Adaptive LSTM"),
    ("lstm_surprisal_diff",     "Vanilla LSTM"),
    ("trigram_surprisal_diff",  "Trigram"),
]
EMOTION_COLS = [("prev_valence", "Valence"), ("prev_arousal", "Arousal")]

# ============================================================================
# HELPERS
# ============================================================================

def corr_pair(x, y):
    """Return Pearson r/p and Spearman rho/p for two aligned series."""
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(float), y[mask].astype(float)
    if len(x) < 10 or x.nunique() < 2 or y.nunique() < 2:
        return None
    r,   p_r   = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return {"n": len(x), "pearson_r": r, "pearson_p": p_r,
            "spearman_rho": rho, "spearman_p": p_rho}


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def strength(r):
    a = abs(r)
    if a < 0.10: return "negligible"
    if a < 0.30: return "weak"
    if a < 0.50: return "moderate"
    return "strong"


def report(label, res, rows, group="ALL"):
    if res is None:
        print(f"  {label:<34s}  (insufficient data)")
        return
    d = "+" if res["pearson_r"] >= 0 else "-"
    print(f"  {label:<34s}  r={res['pearson_r']:+.3f} {sig(res['pearson_p']):<4s} "
          f"rho={res['spearman_rho']:+.3f} {sig(res['spearman_p']):<4s} "
          f"n={res['n']:,}  [{strength(res['pearson_r'])}, {d}]")
    rows.append({"group": group, "pair": label, **res,
                 "direction": d, "strength": strength(res["pearson_r"])})

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 78)
    print(f" SCRIPT 22: SURPRISAL x EMOTION CORRELATIONS  (v{VERSION})")
    print("=" * 78)

    if not os.path.exists(FEATURES_IN):
        sys.exit(f"ERROR: {FEATURES_IN} not found. Run Scripts 17 and 21 first.")
    df_all = pd.read_pickle(FEATURES_IN)
    print(f"\nLoaded {len(df_all):,} pairwise rows")

    # ---- Collapse to ONE ROW PER SENTENCE -----------------------------------
    # The pairwise file repeats each sentence many times (once per variant and
    # per label). Emotion is constant within a sentence; surprisal is not, so we
    # take the mean surprisal difference per sentence as its representative value.
    have_surp = [(c, n) for c, n in SURPRISAL_COLS if c in df_all.columns]
    if not have_surp:
        sys.exit("ERROR: no surprisal columns found.")
    missing_emo = [c for c, _ in EMOTION_COLS if c not in df_all.columns]
    if missing_emo:
        sys.exit(f"ERROR: missing emotion columns: {missing_emo}")

    # CRITICAL (v1.1 fix): the file stores each comparison TWICE - once as
    # label=1 with +(ref-var) and once as label=0 with -(ref-var). Averaging
    # over both makes every sentence's surprisal exactly 0 (a constant column).
    # Keep only the label=1 rows so the differences retain their sign.
    if "label" in df_all.columns:
        before = len(df_all)
        df_all = df_all[df_all["label"] == 1]
        print(f"Kept label=1 rows only: {before:,} -> {len(df_all):,} "
              f"(avoids +/- cancellation)")

    agg = {c: "mean" for c, _ in have_surp}
    agg.update({"prev_valence": "first", "prev_arousal": "first"})
    if "construction_type" in df_all.columns:
        agg["construction_type"] = "first"

    df = df_all.groupby("sent_id", as_index=False).agg(agg)
    print(f"Collapsed to {len(df):,} unique sentences")
    print("(emotion is constant per sentence; surprisal averaged across its variants)")

    # Guard: warn if any surprisal column is still (near-)constant
    for c, n in have_surp:
        if df[c].std() < 1e-12:
            print(f"  WARNING: '{n}' is constant after collapsing - check the data.")

    print(f"\nSurprisal variants available: {', '.join(n for _, n in have_surp)}")

    rows = []

    # ---- 1. Sanity check: valence vs arousal --------------------------------
    print("\n" + "-" * 78)
    print(" 1. SANITY CHECK — valence vs arousal")
    print("-" * 78)
    print(" Russell (1980) reports these near-independent (he measured r = .03)\n")
    res = corr_pair(df["prev_valence"], df["prev_arousal"])
    report("Valence  vs  Arousal", res, rows)
    if res is not None and abs(res["pearson_r"]) < 0.20:
        print("  -> consistent with Russell: the two dimensions are largely independent")
    elif res is not None:
        d = "negative" if res["pearson_r"] < 0 else "positive"
        print(f"  -> NOTE: a {d} correlation of this size is NOT what Russell reports")
        print("     (he measured r = .03 on English affect words). A negative r here")
        print("     means unpleasant sentences tend to score HIGHER on arousal.")
        print("     Plausible causes: (a) the lexicon's known upward arousal bias for")
        print("     calm affect, (b) news-corpus register - negative events are also")
        print("     high-energy, (c) sentence-level averaging. Worth reporting.")

    # ---- 2. Main question: surprisal vs emotion -----------------------------
    print("\n" + "-" * 78)
    print(" 2. MAIN QUESTION — surprisal vs emotion scores")
    print("-" * 78 + "\n")
    for scol, sname in have_surp:
        for ecol, ename in EMOTION_COLS:
            report(f"{sname}  vs  {ename}", corr_pair(df[scol], df[ecol]), rows)
        print()

    # ---- 3. Split by construction type --------------------------------------
    if "construction_type" in df.columns:
        print("-" * 78)
        print(" 3. BY CONSTRUCTION TYPE")
        print("-" * 78)
        primary_col, primary_name = have_surp[0]
        for ctype in ["SOV", "DOSV", "IOSV"]:
            sub = df[df["construction_type"] == ctype]
            print(f"\n  {ctype}  (n = {len(sub):,} sentences)")
            if len(sub) < 10:
                print("    (too few sentences)")
                continue
            for ecol, ename in EMOTION_COLS:
                report(f"  {primary_name} vs {ename}",
                       corr_pair(sub[primary_col], sub[ecol]), rows, group=ctype)

    # ---- 4. Scatter plots ---------------------------------------------------
    primary_col, primary_name = have_surp[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (ecol, ename) in zip(axes, EMOTION_COLS):
        x = df[ecol].astype(float)
        y = df[primary_col].astype(float)
        m = x.notna() & y.notna()
        x, y = x[m], y[m]
        ax.scatter(x, y, s=9, alpha=0.28, color="#4A72C4", edgecolors="none")
        if len(x) > 2:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, color="#E0A458", lw=2.2)
            r, p = stats.pearsonr(x, y)
            ax.set_title(f"{primary_name} vs {ename}\nr = {r:+.3f}  ({sig(p)})",
                         fontsize=12, color="#1E2761")
        ax.set_xlabel(ename, fontsize=11)
        ax.set_ylabel(f"{primary_name} surprisal", fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"\n  Scatter plots saved -> {PLOT_OUT}")

    # ---- 5. Save table ------------------------------------------------------
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"  Correlation table saved -> {CSV_OUT}")

    # ---- 6. Interpretation guide -------------------------------------------
    print("\n" + "=" * 78)
    print(" HOW TO READ THIS")
    print("=" * 78)
    print("""
  |r| < 0.10 negligible | 0.10-0.30 weak | 0.30-0.50 moderate | > 0.50 strong

  A NEGLIGIBLE correlation is an informative result, not a failure: it would
  mean surprisal and emotion are largely INDEPENDENT signals. That strengthens
  the MTP-II story - the interaction effect cannot be dismissed as the two
  measures simply tracking each other.

  A CLEAR correlation would be equally interesting: it would suggest emotional
  context systematically co-varies with predictability, which could help
  explain WHY the interaction arises.

  Pearson measures linear association; Spearman measures monotonic association.
  If they diverge substantially, inspect the scatter plot for non-linearity.
""")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
