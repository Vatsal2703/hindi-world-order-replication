#!/usr/bin/env python3
"""
Lexical Repetition Surprisal (Cache Language Model)

Paper Section 3.1 Feature 5:
  Interpolate the trigram LM with a unigram cache LM built from
  the immediately preceding sentence (history H=100 words, weight mu=0.05).

  P(w | context) = mu * P_cache(w) + (1 - mu) * P_trigram(w | w-2, w-1)

  where P_cache(w) = count(w in cache) / H

The cache captures lexical priming: words recently seen are more likely to
re-occur, boosting their probability and lowering their surprisal.

Output: lex_rept_scores.pkl
  - Same length and order as all_variants_final.pkl
  - Each entry: {sent_id, lex_rept_surprisal, is_reference}
  - Script 20 will zip this with other feature files.
"""

import sys
import os
import pickle
import math
from tqdm import tqdm
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("\n" + "="*70)
print(" LEXICAL REPETITION SURPRISAL (CACHE LM)")
print("="*70 + "\n")

# ============================================================================
# CONFIG (matches paper)
# ============================================================================
MU = 0.05        # Interpolation weight for cache (paper default)
H = 100          # Cache history size (preceding sentence words)

TRIGRAM_MODEL = "./data/models/trigram_model_blind.pkl"
VARIANTS_FILE = "./data/processed/all_variants_final.pkl"
REF_SENTS_FILE = "./data/processed/replication_filtered_sentences.pkl"
OUTPUT_FILE = "./data/results/lex_rept_scores.pkl"

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading trigram model...")
with open(TRIGRAM_MODEL, 'rb') as f:
    trigram_model = pickle.load(f)

unigrams = trigram_model['unigrams']
bigrams = trigram_model['bigrams']
trigrams = trigram_model['trigrams']
print(f"  Vocab size: {trigram_model['vocab_size']:,}")

print("Loading variants...")
with open(VARIANTS_FILE, 'rb') as f:
    all_variants = pickle.load(f)
print(f"  {len(all_variants):,} variants")

print("Loading reference sentences (discourse order)...")
with open(REF_SENTS_FILE, 'rb') as f:
    ref_sents = pickle.load(f)

sent_lookup = {s.sent_id: s for s in ref_sents}
ordered_sent_ids = [s.sent_id for s in ref_sents]
sent_id_to_disc_idx = {sid: i for i, sid in enumerate(ordered_sent_ids)}
print(f"  {len(ref_sents):,} reference sentences")

# ============================================================================
# KNESER-NEY TRIGRAM PROBABILITY
# ============================================================================
def kn_prob(w1, w2, w3):
    """Interpolated Kneser-Ney trigram probability."""
    D = trigram_model['D']

    # Unigram KN
    uni_cont  = trigram_model['unigram_continuation_count'].get(w3, 0)
    total_bi  = trigram_model['total_bigram_types']
    p_uni     = uni_cont / total_bi if total_bi > 0 else 1e-10

    # Bigram KN
    bi_cont       = trigram_model['bigram_continuation_count'].get((w2, w3), 0)
    bi_cont_total = trigram_model['bigram_cont_total'].get(w2, 0)
    if bi_cont_total > 0:
        bi_disc = max(bi_cont - D, 0) / bi_cont_total
        lam_bi  = D * trigram_model['bigram_unique_followers'].get(w2, 0) / bi_cont_total
    else:
        bi_disc, lam_bi = 0.0, 1.0
    p_bi = bi_disc + lam_bi * p_uni

    # Trigram KN
    tri_count = trigrams.get((w1, w2, w3), 0)
    bi_count  = bigrams.get((w1, w2), 0)
    if bi_count > 0:
        tri_disc = max(tri_count - D, 0) / bi_count
        lam_tri  = D * trigram_model['bigram_followers_count'].get((w1, w2), 0) / bi_count
    else:
        tri_disc, lam_tri = 0.0, 1.0
    p_tri = tri_disc + lam_tri * p_bi

    return max(p_tri, 1e-10)


# ============================================================================
# CACHE LM INTERPOLATION
# ============================================================================
def cache_prob(w, cache_counter, cache_total):
    """Unigram cache probability."""
    if cache_total == 0:
        return 0.0
    return cache_counter.get(w, 0) / cache_total


def lex_rept_surprisal(tokens, cache_tokens):
    """
    Total sentence surprisal using interpolated cache + KN trigram LM.
    """
    cache = cache_tokens[-H:] if len(cache_tokens) > H else cache_tokens
    cache_counter = Counter(cache)
    cache_total = len(cache)

    full_tokens = ['<s>', '<s>'] + tokens + ['</s>']
    total_surp = 0.0

    for i in range(2, len(full_tokens)):
        w1, w2, w3 = full_tokens[i-2], full_tokens[i-1], full_tokens[i]

        p_tri   = kn_prob(w1, w2, w3)
        p_cache = cache_prob(w3, cache_counter, cache_total)

        p_interp = MU * p_cache + (1 - MU) * p_tri
        if p_interp <= 0:
            p_interp = 1e-10

        total_surp += -math.log2(p_interp)

    return total_surp

# ============================================================================
# BUILD PER-VARIANT POSITION MAP
# ============================================================================
sent_to_positions = defaultdict(list)
for i, v in enumerate(all_variants):
    sent_to_positions[v['sent_id']].append((i, v))

# Processing order (first appearance in all_variants)
seen = set()
processing_order = []
for v in all_variants:
    if v['sent_id'] not in seen:
        processing_order.append(v['sent_id'])
        seen.add(v['sent_id'])

# ============================================================================
# COMPUTE
# ============================================================================
print(f"\nComputing lex-rept surprisal (mu={MU}, H={H}) "
      f"for {len(processing_order)} sentences...")

results = [None] * len(all_variants)

for sent_id in tqdm(processing_order, desc="Lex-Rept Surprisal"):
    sent_obj = sent_lookup.get(sent_id)

    # Get preceding sentence tokens for the cache
    disc_idx = sent_id_to_disc_idx.get(sent_id)
    cache_tokens = []
    if disc_idx is not None and disc_idx > 0:
        prev_sent = sent_lookup.get(ordered_sent_ids[disc_idx - 1])
        if prev_sent:
            cache_tokens = [w.form for w in prev_sent.words]

    # Score each variant
    for orig_idx, variant in sent_to_positions[sent_id]:
        if sent_obj is None:
            surp = 0.0
        else:
            word_map = {w.idx: w.form for w in sent_obj.words}
            tokens = [word_map.get(idx, '<UNK>') for idx in variant['variant_order']]
            surp = lex_rept_surprisal(tokens, cache_tokens)

        results[orig_idx] = {
            'sent_id': sent_id,
            'lex_rept_surprisal': surp,
            'is_reference': variant['is_reference']
        }

# Fill gaps
for i, r in enumerate(results):
    if r is None:
        results[i] = {
            'sent_id': all_variants[i]['sent_id'],
            'lex_rept_surprisal': 0.0,
            'is_reference': all_variants[i]['is_reference']
        }

# ============================================================================
# SAVE
# ============================================================================
os.makedirs("./data/results", exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(results, f)

refs = sum(1 for r in results if r['is_reference'])
print(f"\nSaved {len(results):,} lex-rept scores to {OUTPUT_FILE}")
print(f"  References: {refs:,}  Variants: {len(results)-refs:,}")
