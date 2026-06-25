#!/usr/bin/env python3
"""
Per-constituent trigram surprisal runner.
Bounded Rationality project — 2023 paper conditions.

Pipeline position: step 4 of 5
  02_data_preparation -> variant_generation -> train_trigram_br -> [THIS] -> analyse_per_constituent

Inputs:
  data/models/br_trigram_model_blind.pkl
  data/processed/reference_sentences.pkl
  data/processed/bounded_rationality_all_variants_final.pkl

Output:
  data/features/per_constituent_surprisal.csv
    One row per (sentence, variant). Key columns:
      sent_id, is_reference, k, sentence_total_surprisal,
      last_surp_sum, last_surp_mean, last_length,
      second_last_surp_sum, second_last_surp_mean, second_last_length,
      const1_surp_sum, const1_surp_mean, const1_length, ... (up to max_k)

Run from project root:
  python scripts/bounded_rationality/per_contituent.py/per_conti.py
"""
from __future__ import annotations
import os, sys, pickle, csv, math
from pathlib import Path
from tqdm import tqdm

# --- path setup ---
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate import aggregate  # type: ignore

# ============================================================================
# CONFIG
# ============================================================================
TRIGRAM_MODEL = "./data/models/br_trigram_model_blind.pkl"
REF_SENTENCES = "./data/processed/reference_sentences.pkl"
VARIANTS      = "./data/processed/bounded_rationality_all_variants_final.pkl"
OUTPUT_CSV    = "./data/features/per_constituent_surprisal.csv"

# ============================================================================
# KN TRIGRAM SCORER — returns per-token surprisal list (not sentence total)
# ============================================================================
def _kn_prob(m, w1, w2, w3):
    D        = m["D"]
    total_bi = m["total_bigram_types"]

    # unigram KN
    p_uni = m["unigram_continuation_count"].get(w3, 0) / total_bi if total_bi > 0 else 1e-10

    # bigram KN
    bi_cont_total = m["bigram_cont_total"].get(w2, 0)
    if bi_cont_total > 0:
        bi_disc = max(m["bigram_continuation_count"].get((w2, w3), 0) - D, 0) / bi_cont_total
        lam_bi  = D * m["bigram_unique_followers"].get(w2, 0) / bi_cont_total
    else:
        bi_disc, lam_bi = 0.0, 1.0
    p_bi = bi_disc + lam_bi * p_uni

    # trigram KN
    bi_count = m["bigrams"].get((w1, w2), 0)
    if bi_count > 0:
        tri_disc = max(m["trigrams"].get((w1, w2, w3), 0) - D, 0) / bi_count
        lam_tri  = D * m["bigram_followers_count"].get((w1, w2), 0) / bi_count
    else:
        tri_disc, lam_tri = 0.0, 1.0

    return max(tri_disc + lam_tri * p_bi, 1e-10)


def per_token_surprisal(model, tokens):
    """Return per-token surprisal (bits). sum() reproduces sentence total."""
    padded = ["<s>", "<s>"] + list(tokens)
    return [
        -math.log2(_kn_prob(model, padded[i-2], padded[i-1], padded[i]))
        for i in range(2, len(padded))
    ]

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" PER-CONSTITUENT TRIGRAM SURPRISAL")
    print("=" * 70)

    for p in (TRIGRAM_MODEL, REF_SENTENCES, VARIANTS):
        if not os.path.exists(p):
            print(f"\nERROR: not found: {p}"); return 1

    print("\nLoading trigram model...")
    with open(TRIGRAM_MODEL, "rb") as f: model = pickle.load(f)
    print(f"  vocab={model['vocab_size']:,}  trigrams={len(model['trigrams']):,}")

    print("Loading reference sentences...")
    with open(REF_SENTENCES, "rb") as f: ref_sents = pickle.load(f)
    sent_lookup = {s.sent_id: s for s in ref_sents}
    print(f"  {len(ref_sents):,} reference sentences")

    print("Loading variants...")
    with open(VARIANTS, "rb") as f: all_variants = pickle.load(f)
    print(f"  {len(all_variants):,} variant entries (refs + non-refs)")

    rows, skipped, max_k = [], 0, 0

    for entry in tqdm(all_variants, desc="Scoring + aggregating"):
        sent_obj = sent_lookup.get(entry["sent_id"])
        if sent_obj is None:
            skipped += 1; continue

        word_map = {w.idx: w.form for w in sent_obj.words}
        tokens   = [word_map.get(idx, "<UNK>") for idx in entry["variant_order"]]
        per_tok  = per_token_surprisal(model, tokens)

        agg = aggregate(
            sent_obj, entry["variant_order"], per_tok,
            sent_id=entry["sent_id"], is_reference=entry["is_reference"],
        )
        max_k = max(max_k, agg.k)
        rows.append(agg.to_flat_row())

    print(f"\n  Aggregated: {len(rows):,} rows")
    print(f"  Skipped (no sent_obj): {skipped}")
    print(f"  Max k observed: {max_k}")

    fixed_cols = [
        "sent_id", "is_reference", "k", "sentence_total_surprisal",
        "last_surp_sum", "last_surp_mean", "last_length",
        "second_last_surp_sum", "second_last_surp_mean", "second_last_length",
    ]
    per_const_cols = [
        f"const{i}_{m}" for i in range(1, max_k + 1)
        for m in ("surp_sum", "surp_mean", "length")
    ]

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fixed_cols + per_const_cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows: writer.writerow(r)

    print(f"\n  Saved -> {OUTPUT_CSV}")

    refs  = [r for r in rows if r["is_reference"]]
    vars_ = [r for r in rows if not r["is_reference"]]
    if refs:
        print(f"\n  Sanity: avg verb-adjacent length — refs {sum(r.get('last_length',0) for r in refs)/len(refs):.2f}  "
              f"vars {sum(r.get('last_length',0) for r in vars_)/len(vars_):.2f}")
        print("  (refs should be lower if least-effort holds)")

    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())