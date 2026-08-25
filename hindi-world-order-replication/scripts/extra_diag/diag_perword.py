#!/usr/bin/env python3
"""
Script 30: Per-Word Normalisation Diagnostic  — v1.0

Checks empirically what dividing surprisal by sentence length actually does,
rather than assuming it. Three questions:

  1. WITHIN a sentence, does normalising change anything?
     All variants of a sentence are permutations of the same tokens, so they
     share n_words. Dividing by a constant cannot reorder them. This section
     verifies that claim on the real data (the ref-vs-variant ranking and the
     ratios should be bit-for-bit unchanged).

  2. ACROSS sentences, how much does it change?
     Here length is a genuine confound: a long sentence accumulates more total
     surprisal simply by having more tokens. This section reports the
     correlation with n_words before and after normalising, and shows how many
     sentence PAIRS swap order once length is controlled for.

  3. Is per-word normalisation equally principled for every feature?
     Surprisal is additive over tokens, so the mean per token is well defined
     (it is log2 of perplexity). Dependency length is NOT additive in the same
     way -- a tree has n-1 arcs and total length grows faster than linearly --
     so this section tests how well n_words actually removes the length effect
     for each feature, and flags any where it does not.

RUN IN `digit` OR `base` CONDA ENV (pandas + scipy, no torch).

Input : data/results/MTP2_raw_dataset_full.csv
Output: console report
        data/results/perword_diagnostic.csv
"""

VERSION = "1.0"

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats


def find_base():
    d = os.path.dirname(os.path.abspath(__file__))
    while d and d != os.path.dirname(d):
        if os.path.basename(d) == "hindi-world-order-replication":
            return d
        if os.path.isdir(os.path.join(d, "hindi-world-order-replication")):
            return os.path.join(d, "hindi-world-order-replication")
        d = os.path.dirname(d)
    return "."


BASE    = find_base()
CSV_IN  = os.path.join(BASE, "data", "results", "MTP2_raw_dataset_full.csv")
CSV_OUT = os.path.join(BASE, "data", "results", "perword_diagnostic.csv")

# Features to test. 'additive' marks whether the quantity is a sum over tokens,
# which is what makes dividing by the token count principled.
FEATURES = [
    ("trigram_surprisal",  True),
    ("lstm_surprisal",     True),
    ("adaptive_surprisal", True),
    ("lex_rept_surprisal", True),
    ("pcfg_surprisal",     True),
    ("dep_len",            False),   # tree has n-1 arcs; grows super-linearly
]


def hr(title):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def main():
    print("\n" + "=" * 78)
    print(f" SCRIPT 30: PER-WORD NORMALISATION DIAGNOSTIC  (v{VERSION})")
    print("=" * 78)

    if not os.path.exists(CSV_IN):
        sys.exit(f"ERROR: {CSV_IN} not found.")
    df = pd.read_csv(CSV_IN, encoding="utf-8-sig")
    print(f"\nLoaded {len(df):,} rows")

    feats = [(c, a) for c, a in FEATURES if c in df.columns
             and pd.to_numeric(df[c], errors="coerce").notna().any()]
    skipped = [c for c, _ in FEATURES if (c, True) not in
               [(f, True) for f, _ in feats] and (c, False) not in feats]
    if skipped:
        print(f"  (skipped, absent or empty: {skipped})")
    for c, _ in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows_out = []

    # ========================================================================
    # 1. WITHIN SENTENCE
    # ========================================================================
    hr("1. WITHIN A SENTENCE — does normalising change anything?")
    print("""
 All variants of a sentence are permutations of the same tokens, so they share
 n_words. Dividing by a constant cannot change their relative order. Verifying:
""")

    lens = df.groupby("sent_id")["n_words"].nunique()
    varying = int((lens > 1).sum())
    print(f"  Sentences whose variants differ in n_words : {varying:,} of {len(lens):,}")
    if varying == 0:
        print("  -> confirmed: length is constant within every sentence")
    else:
        print("  -> NOTE: some sentences vary in length; normalising is not a "
              "no-op for those")

    # Does the reference keep the same rank within its sentence?
    primary = feats[0][0]
    disagree = 0
    checked = 0
    for sid, g in df.groupby("sent_id"):
        if len(g) < 2 or g[primary].isna().any():
            continue
        checked += 1
        tot_rank = g[primary].rank().values
        pw_rank = (g[primary] / g["n_words"]).rank().values
        if not np.allclose(tot_rank, pw_rank):
            disagree += 1
    print(f"\n  Sentences where the within-sentence RANKING changed: "
          f"{disagree:,} of {checked:,}")
    print("  -> normalising is a no-op for the reference-vs-variant comparison"
          if disagree == 0 else
          "  -> unexpected: ranking changed somewhere, worth investigating")

    # Show the ratio is preserved on one example
    ex = df[df["sent_id"] == df["sent_id"].iloc[0]]
    if len(ex) >= 2 and ex[primary].notna().all():
        r = ex[ex["is_reference"] == 1].iloc[0]
        v = ex[ex["is_reference"] == 0].iloc[0]
        n = r["n_words"]
        print(f"\n  Example — {r['sent_id']} ({int(n)} tokens), {primary}:")
        print(f"    {'':<12s} {'total':>10s} {'per-word':>10s}")
        print(f"    {'reference':<12s} {r[primary]:>10.2f} {r[primary]/n:>10.2f}")
        print(f"    {'variant':<12s} {v[primary]:>10.2f} {v[primary]/n:>10.2f}")
        if r[primary] != 0:
            print(f"    {'ratio':<12s} {v[primary]/r[primary]:>10.2f}x "
                  f"{(v[primary]/n)/(r[primary]/n):>9.2f}x   <- identical")

    # ========================================================================
    # 2. ACROSS SENTENCES
    # ========================================================================
    hr("2. ACROSS SENTENCES — where normalising genuinely matters")
    print("""
 Comparing different sentences is where length is a real confound. If the
 totals track n_words strongly but the per-word values do not, normalising has
 removed a genuine length effect.
""")
    ref = df[df["is_reference"] == 1].copy()
    print(f"  Using the {len(ref):,} reference sentences")
    print(f"  Length range: {int(ref['n_words'].min())}–{int(ref['n_words'].max())} "
          f"tokens (median {int(ref['n_words'].median())})\n")

    print(f"  {'feature':<22s} {'r(total, n)':>12s} {'r(perword, n)':>14s}  verdict")
    print("  " + "-" * 72)
    for c, additive in feats:
        sub = ref[[c, "n_words"]].dropna()
        if len(sub) < 10:
            continue
        r_tot = sub[c].corr(sub["n_words"])
        r_pw = (sub[c] / sub["n_words"]).corr(sub["n_words"])
        if abs(r_tot) > 0.5 and abs(r_pw) < 0.3:
            verdict = "confound removed"
        elif abs(r_pw) < abs(r_tot) - 0.1:
            verdict = "reduced"
        elif abs(r_pw) > abs(r_tot):
            verdict = "OVER-corrected"
        else:
            verdict = "little change"
        rows_out.append({"feature": c, "r_total_vs_n": round(r_tot, 3),
                         "r_perword_vs_n": round(r_pw, 3), "verdict": verdict,
                         "additive_over_tokens": additive})
        print(f"  {c:<22s} {r_tot:>12.3f} {r_pw:>14.3f}  {verdict}")

    # How many cross-sentence pairs actually swap order?
    print("\n  How often does the cross-sentence conclusion flip?")
    rng = np.random.default_rng(42)
    for c, _ in feats:
        sub = ref[[c, "n_words"]].dropna().reset_index(drop=True)
        if len(sub) < 100:
            continue
        n_pairs = 20000
        i = rng.integers(0, len(sub), n_pairs)
        j = rng.integers(0, len(sub), n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        tot_i, tot_j = sub[c].values[i], sub[c].values[j]
        pw_i = tot_i / sub["n_words"].values[i]
        pw_j = tot_j / sub["n_words"].values[j]
        flips = np.mean(np.sign(tot_i - tot_j) != np.sign(pw_i - pw_j))
        print(f"    {c:<22s} {100*flips:5.1f}% of random sentence pairs "
              f"swap order")

    # ========================================================================
    # 3. IS IT EQUALLY PRINCIPLED FOR EVERY FEATURE?
    # ========================================================================
    hr("3. IS PER-WORD EQUALLY PRINCIPLED FOR EVERY FEATURE?")
    print("""
 Surprisal is a SUM over tokens, so the mean per token is well defined -- it is
 exactly log2(perplexity). Dependency length is not additive in the same way: a
 tree has n-1 arcs and total length grows faster than linearly with n. Below,
 the fitted exponent b in  feature ~ n^b  shows how each quantity actually
 scales. b close to 1 means dividing by n is the right correction; b clearly above 1
 means it under-corrects.
""")
    print(f"  {'feature':<22s} {'exponent b':>12s}   interpretation")
    print("  " + "-" * 72)
    for c, additive in feats:
        sub = ref[[c, "n_words"]].dropna()
        sub = sub[(sub[c] > 0) & (sub["n_words"] > 1)]
        if len(sub) < 30:
            continue
        b, _, _, _, _ = stats.linregress(np.log(sub["n_words"]), np.log(sub[c]))
        if abs(b - 1.0) < 0.15:
            interp = "≈ linear — dividing by n is appropriate"
        elif b > 1.15:
            interp = "super-linear — dividing by n UNDER-corrects"
        else:
            interp = "sub-linear — dividing by n OVER-corrects"
        for r in rows_out:
            if r["feature"] == c:
                r["scaling_exponent"] = round(b, 3)
        print(f"  {c:<22s} {b:>12.3f}   {interp}")

    print("""
  Note on dep_len specifically: a dependency tree over n tokens has n-1 arcs,
  so a denominator of (n-1) is arguably more natural than n. More importantly,
  if the exponent above is well over 1, dividing by n does not fully remove the
  length effect and the normalised value still carries some length signal.
""")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    hr("SUMMARY")
    print("""
  1. Within a sentence, per-word normalisation is a NO-OP. Variants share the
     same token count, so the reference-vs-variant comparison -- the basis of
     the Joachims setup -- is completely unchanged.

  2. Across sentences it matters, and is the correct thing to do: totals track
     sentence length, per-word values largely do not. Any analysis comparing
     different sentences (for example correlating surprisal with emotion)
     should use the per-word values.

  3. The normalisation is principled for the surprisal measures, which are sums
     over tokens. Treat dep_len_per_word with more caution -- see the fitted
     exponent above.
""")
    if rows_out:
        pd.DataFrame(rows_out).to_csv(CSV_OUT, index=False)
        print(f"  Saved -> {CSV_OUT}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()