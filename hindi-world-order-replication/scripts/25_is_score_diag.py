#!/usr/bin/env python3
"""
Script 25: Feature Diagnostics  — v1.0

MTP-II sanity check. Reading the master sheet by eye, `info_status_mean` looked
like it was 0.0 for most sentences. A feature that is (near-)constant carries no
information and cannot contribute to the classifier — so before anyone else
spots it, this script checks the distribution of EVERY feature and flags any
that look degenerate.

For each feature it reports:
  - % of sentences that are exactly zero
  - min / max / mean / std, and the 25th, 50th, 75th percentiles
  - number of distinct values
  - a verdict: OK / SPARSE / DEGENERATE

It also checks the same features on the raw pairwise file, so we can tell
whether a feature is genuinely flat or is only being flattened by the
per-sentence averaging step in the master sheet.

RUN IN `digit` CONDA ENV.

Input : data/features/all_features_with_emotion.pkl
Output: data/results/feature_diagnostics.csv
        console report
"""

VERSION = "1.0"

import sys
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from utils.paths import find_base

# ============================================================================
# PATHS
# ============================================================================

BASE        = find_base(__file__)
FEATURES_IN = os.path.join(BASE, "data", "features", "all_features_with_emotion.pkl")
CSV_OUT     = os.path.join(BASE, "data", "results", "feature_diagnostics.csv")

FEATURE_COLS = [
    "dep_len_diff",
    "info_status_diff",
    "trigram_surprisal_diff",
    "lstm_surprisal_diff",
    "adaptive_surprisal_diff",
    "lex_rept_surprisal_diff",
    "pcfg_surprisal_diff",
]
EMOTION_COLS = ["prev_valence", "prev_arousal"]


def verdict(pct_zero, n_unique, std):
    if std < 1e-12 or n_unique <= 1:
        return "DEGENERATE"
    if pct_zero > 90:
        return "DEGENERATE"
    if pct_zero > 50:
        return "SPARSE"
    return "OK"


def describe(series, name, level):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"level": level, "feature": name, "n": 0, "verdict": "EMPTY"}
    pct_zero = 100.0 * (s == 0).sum() / len(s)
    row = {
        "level": level,
        "feature": name,
        "n": len(s),
        "pct_zero": round(pct_zero, 2),
        "n_unique": int(s.nunique()),
        "min": round(s.min(), 4),
        "p25": round(s.quantile(0.25), 4),
        "median": round(s.median(), 4),
        "p75": round(s.quantile(0.75), 4),
        "max": round(s.max(), 4),
        "mean": round(s.mean(), 4),
        "std": round(s.std(), 4),
    }
    row["verdict"] = verdict(pct_zero, row["n_unique"], row["std"])
    return row


def print_table(rows, title):
    print("\n" + "-" * 100)
    print(f" {title}")
    print("-" * 100)
    hdr = (f"  {'feature':<26s} {'%zero':>7s} {'uniq':>7s} {'min':>10s} "
           f"{'median':>10s} {'max':>10s} {'std':>9s}  verdict")
    print(hdr)
    print("  " + "-" * 96)
    for r in rows:
        if r.get("n", 0) == 0:
            print(f"  {r['feature']:<26s}  (no data)")
            continue
        flag = "" if r["verdict"] == "OK" else "   <-- " + r["verdict"]
        print(f"  {r['feature']:<26s} {r['pct_zero']:>6.1f}% {r['n_unique']:>7,} "
              f"{r['min']:>10.3f} {r['median']:>10.3f} {r['max']:>10.3f} "
              f"{r['std']:>9.3f}  {r['verdict']}{flag}")


def main():
    print("\n" + "=" * 100)
    print(f" SCRIPT 25: FEATURE DIAGNOSTICS  (v{VERSION})")
    print("=" * 100)

    if not os.path.exists(FEATURES_IN):
        sys.exit(f"ERROR: {FEATURES_IN} not found.")
    df_all = pd.read_pickle(FEATURES_IN)
    print(f"\nLoaded {len(df_all):,} pairwise rows")

    have = [c for c in FEATURE_COLS if c in df_all.columns]
    missing = [c for c in FEATURE_COLS if c not in df_all.columns]
    if missing:
        print(f"  NOTE: not in file: {missing}")

    rows = []

    # ---- LEVEL 1: raw pairwise rows (label=1 only, to keep the sign) --------
    pw = df_all[df_all["label"] == 1] if "label" in df_all.columns else df_all
    print(f"Using {len(pw):,} label=1 pairwise rows for the raw check")
    for c in have:
        rows.append(describe(pw[c], c, "pairwise"))
    print_table([r for r in rows if r["level"] == "pairwise"],
                "LEVEL 1 — RAW PAIRWISE ROWS (one row per reference-vs-variant comparison)")

    # ---- LEVEL 2: per-sentence means (what the master sheet shows) ---------
    agg = {c: "mean" for c in have}
    per_sent = pw.groupby("sent_id", as_index=False).agg(agg)
    print(f"\nCollapsed to {len(per_sent):,} sentences for the master-sheet check")
    sent_rows = [describe(per_sent[c], c, "per_sentence") for c in have]
    rows.extend(sent_rows)
    print_table(sent_rows, "LEVEL 2 — PER-SENTENCE MEANS (as they appear in the master sheet)")

    # ---- Emotion columns, for reference ------------------------------------
    emo_have = [c for c in EMOTION_COLS if c in pw.columns]
    if emo_have:
        emo_sent = pw.groupby("sent_id", as_index=False).agg({c: "first" for c in emo_have})
        emo_rows = [describe(emo_sent[c], c, "per_sentence") for c in emo_have]
        rows.extend(emo_rows)
        print_table(emo_rows, "EMOTION COLUMNS (per sentence, for reference)")

    # ---- Focused look at info_status ---------------------------------------
    if "info_status_diff" in have:
        print("\n" + "=" * 100)
        print(" FOCUS: info_status_diff")
        print("=" * 100)
        s_pw = pd.to_numeric(pw["info_status_diff"], errors="coerce").dropna()
        s_ps = pd.to_numeric(per_sent["info_status_diff"], errors="coerce").dropna()
        print(f"\n  Pairwise rows exactly 0 : {100*(s_pw==0).sum()/len(s_pw):.1f}%  "
              f"({(s_pw==0).sum():,} of {len(s_pw):,})")
        print(f"  Sentences with mean 0   : {100*(s_ps==0).sum()/len(s_ps):.1f}%  "
              f"({(s_ps==0).sum():,} of {len(s_ps):,})")
        nz = s_pw[s_pw != 0]
        if len(nz):
            print(f"\n  Among NON-zero pairwise values (n={len(nz):,}):")
            print(f"    range  : [{nz.min():.3f}, {nz.max():.3f}]")
            print(f"    mean   : {nz.mean():.3f}   median: {nz.median():.3f}")
        print("""
  How to read this:
    A high %zero at the PAIRWISE level means the feature genuinely does not
    distinguish most reference-variant pairs -> it is weak/sparse by nature.
    A low %zero at pairwise but high at per-sentence level would instead mean
    positive and negative values are cancelling during averaging -> the feature
    is fine, and only the master-sheet summary hides it.
""")

    # ---- Save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    # ---- Summary ------------------------------------------------------------
    print("=" * 100)
    print(" VERDICT SUMMARY")
    print("=" * 100)
    bad = [r for r in rows if r.get("verdict") in ("DEGENERATE", "SPARSE", "EMPTY")]
    if not bad:
        print("\n  All features look healthy — no degenerate or sparse columns.")
    else:
        for r in bad:
            print(f"  [{r['verdict']:<11s}] {r['feature']:<26s} ({r['level']})  "
                  f"%zero={r.get('pct_zero','?')}  unique={r.get('n_unique','?')}")
        print("""
  DEGENERATE = effectively constant; contributes nothing to the model.
  SPARSE     = zero for most rows; contributes only for a minority of cases.
               Not necessarily a bug — some linguistic features only fire when
               a specific configuration is present — but worth reporting.
""")
    print(f"  Saved -> {CSV_OUT}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()