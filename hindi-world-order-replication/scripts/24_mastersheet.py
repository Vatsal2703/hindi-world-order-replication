#!/usr/bin/env python3
"""
Script 24: Master Sheet — all per-sentence scores in one table  — v1.0

MTP-II (requested by supervisor): consolidate everything computed per sentence
into a single spreadsheet for inspection and further analysis.

Columns produced (one row per reference sentence):
    sent_id            unique sentence identifier
    sentence           the reference sentence text (Devanagari)
    construction_type  SOV / DOSV / IOSV  (from Script 21)
    n_words            sentence length in tokens
    n_variants         how many reordered variants exist for this sentence
    prev_valence       valence of the PRECEDING sentence   [-1, +1]
    prev_arousal       arousal of the PRECEDING sentence   [ 0,  1]
    prev_coverage      fraction of preceding-sentence words found in the lexicon
    prev_sentence      the preceding sentence text (the emotion source)
    <feature>_mean     mean reference-vs-variant difference for each feature
    label              class label (1 = reference / attested ordering)

IMPORTANT — how the feature scores are aggregated:
The feature file stores every comparison TWICE: once as label=1 carrying
+(reference - variant) and once as label=0 carrying the negated value. Only the
label=1 rows are kept, so the differences retain their sign; they are then
averaged over that sentence's variants. A POSITIVE mean means the attested
ordering scored higher than its variants on that feature.

RUN IN `digit` CONDA ENV (pandas 3.0.1 — the version that wrote the pickles).
Run Script 21 first if construction_type is missing.

Input : data/features/all_features_with_emotion.pkl
        data/processed/replication_filtered_sentences.pkl   (for the text)
        data/processed/preceding_emotions.pkl               (for coverage)
Output: data/results/MTP2_master_sheet.csv
        data/results/MTP2_master_sheet.xlsx   (if openpyxl is available)
"""

VERSION = "1.0"

import sys
import os
import pickle
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from utils.paths import find_base
from parsers.ud_parser import Sentence, Word  # noqa: F401  (needed to unpickle)

# ============================================================================
# PATHS
# ============================================================================

BASE        = find_base(__file__)
FEATURES_IN = os.path.join(BASE, "data", "features", "all_features_with_emotion.pkl")
SENTS_IN    = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
EMOTIONS_IN = os.path.join(BASE, "data", "processed", "preceding_emotions.pkl")
CSV_OUT     = os.path.join(BASE, "data", "results", "MTP2_master_sheet.csv")
XLSX_OUT    = os.path.join(BASE, "data", "results", "MTP2_master_sheet.xlsx")

# Feature columns to aggregate, in a sensible reading order
FEATURE_COLS = [
    "dep_len_diff",
    "info_status_diff",
    "trigram_surprisal_diff",
    "lstm_surprisal_diff",
    "adaptive_surprisal_diff",
    "lex_rept_surprisal_diff",
    "pcfg_surprisal_diff",
]

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 78)
    print(f" SCRIPT 24: MASTER SHEET  (v{VERSION})")
    print("=" * 78)

    # ---- Load the pairwise feature file ------------------------------------
    if not os.path.exists(FEATURES_IN):
        sys.exit(f"ERROR: {FEATURES_IN} not found. Run Scripts 17 and 21 first.")
    df = pd.read_pickle(FEATURES_IN)
    print(f"\nLoaded {len(df):,} pairwise rows, {len(df.columns)} columns")

    if "construction_type" not in df.columns:
        print("\n  WARNING: 'construction_type' is missing.")
        print("  Run scripts/21_emotion_construction_analysis.py first, then re-run this.")

    # ---- Keep label=1 rows so the signed differences survive ---------------
    if "label" in df.columns:
        before = len(df)
        df = df[df["label"] == 1]
        print(f"Kept label=1 rows only: {before:,} -> {len(df):,} (preserves the sign)")

    have_feats = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  NOTE: features not in file (skipped): {missing}")
    print(f"Aggregating {len(have_feats)} feature columns")

    # ---- Collapse to one row per sentence ----------------------------------
    agg = {c: "mean" for c in have_feats}
    agg["prev_valence"] = "first"
    agg["prev_arousal"] = "first"
    if "construction_type" in df.columns:
        agg["construction_type"] = "first"

    master = df.groupby("sent_id", as_index=False).agg(agg)
    master = master.rename(columns={c: c.replace("_diff", "_mean") for c in have_feats})

    # variant counts
    counts = df.groupby("sent_id").size().reset_index(name="n_variants")
    master = master.merge(counts, on="sent_id", how="left")
    print(f"Collapsed to {len(master):,} unique sentences")

    # ---- Attach sentence text, length, and the preceding sentence ----------
    if os.path.exists(SENTS_IN):
        with open(SENTS_IN, "rb") as f:
            sentences = pickle.load(f)

        ordered = [s.sent_id for s in sentences]
        lookup  = {s.sent_id: s for s in sentences}
        pos     = {sid: i for i, sid in enumerate(ordered)}

        def text_of(s):
            t = getattr(s, "text", None)
            if t:
                return t
            return " ".join(w.form for w in s.words)

        texts, lengths, prev_texts = {}, {}, {}
        for sid, s in lookup.items():
            texts[sid]   = text_of(s)
            lengths[sid] = len(s.words)
            i = pos[sid]
            prev_texts[sid] = text_of(lookup[ordered[i - 1]]) if i > 0 else ""

        master["sentence"]      = master["sent_id"].map(texts)
        master["n_words"]       = master["sent_id"].map(lengths)
        master["prev_sentence"] = master["sent_id"].map(prev_texts)
        print(f"Attached sentence text for {master['sentence'].notna().sum():,} sentences")
    else:
        print(f"  NOTE: {SENTS_IN} not found — sentence text will be blank")
        master["sentence"] = ""
        master["n_words"] = None
        master["prev_sentence"] = ""

    # ---- Attach lexicon coverage from the emotion annotations --------------
    if os.path.exists(EMOTIONS_IN):
        with open(EMOTIONS_IN, "rb") as f:
            emotions = pickle.load(f)
        master["prev_coverage"] = master["sent_id"].map(
            lambda s: emotions.get(s, {}).get("coverage"))
        master["prev_n_matched"] = master["sent_id"].map(
            lambda s: emotions.get(s, {}).get("n_matched"))
        print("Attached lexicon coverage")
    else:
        master["prev_coverage"] = None
        master["prev_n_matched"] = None

    # ---- Class label -------------------------------------------------------
    # Every retained row is the attested (reference) ordering.
    master["label"] = 1

    # ---- Order the columns for readability ---------------------------------
    lead = ["sent_id", "sentence", "construction_type", "n_words", "n_variants"]
    emo  = ["prev_valence", "prev_arousal", "prev_coverage", "prev_n_matched", "prev_sentence"]
    feat = [c.replace("_diff", "_mean") for c in have_feats]
    order = [c for c in lead + emo + feat + ["label"] if c in master.columns]
    master = master[order + [c for c in master.columns if c not in order]]

    master = master.sort_values("sent_id").reset_index(drop=True)

    # ---- Round the numeric columns for readability -------------------------
    for c in master.select_dtypes(include="number").columns:
        if c not in ("n_words", "n_variants", "label", "prev_n_matched"):
            master[c] = master[c].round(4)

    # ---- Save --------------------------------------------------------------
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    master.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")  # BOM = Excel-safe Devanagari
    print(f"\n  Saved -> {CSV_OUT}")

    try:
        master.to_excel(XLSX_OUT, index=False)
        print(f"  Saved -> {XLSX_OUT}")
    except Exception as e:
        print(f"  (xlsx skipped: {e})")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "-" * 78)
    print(" MASTER SHEET SUMMARY")
    print("-" * 78)
    print(f"  Rows (sentences) : {len(master):,}")
    print(f"  Columns          : {len(master.columns)}")
    print(f"\n  Columns: {list(master.columns)}")

    if "construction_type" in master.columns:
        print("\n  Construction distribution:")
        vc = master["construction_type"].value_counts()
        for ct in ["SOV", "DOSV", "IOSV", "UNKNOWN"]:
            n = vc.get(ct, 0)
            if n:
                print(f"    {ct:>8s}: {n:>5,} ({100*n/len(master):.1f}%)")

    print(f"\n  Valence : [{master['prev_valence'].min():.3f}, {master['prev_valence'].max():.3f}]")
    print(f"  Arousal : [{master['prev_arousal'].min():.3f}, {master['prev_arousal'].max():.3f}]")

    print("\n  First rows (key columns):")
    show = [c for c in ["sent_id", "construction_type", "prev_valence",
                        "prev_arousal", "adaptive_surprisal_mean"] if c in master.columns]
    print(master[show].head(5).to_string(index=False))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()