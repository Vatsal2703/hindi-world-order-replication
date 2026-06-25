#!/usr/bin/env python3
"""
Subtree-aware variant generation.
Bounded Rationality project — 2023 paper conditions.

Pipeline position: step 2 of 5
  02_data_preparation -> [THIS] -> train_trigram_br -> per_conti -> analyse

For each reference sentence:
  1. Identify preverbal direct dependents of root (excluding punct)
  2. Extract complete subtrees for each dependent (DFS)
  3. Permute subtree blocks (not individual words)
  4. Keep root verb + postverbal in original position
  5. Deduplicate by surface string
  6. If > 99 unique variants, randomly sample exactly 99

Input:  data/processed/reference_sentences.pkl
Output: data/processed/bounded_rationality_all_variants_final.pkl

Run from project root:
  python scripts/bounded_rationality/per_contituent.py/variant_generation.py
"""
import os, sys, pickle, itertools, random
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# --- path setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from parsers.ud_parser import Sentence, Word

# ============================================================================
# CONFIG
# ============================================================================
INPUT_FILE          = "./data/processed/reference_sentences.pkl"
OUTPUT_FILE         = "./data/processed/bounded_rationality_all_variants_final.pkl"
MAX_NON_REF_VARIANTS = 99

SUBJ_TAGS      = {'nsubj', 'csubj'}
DIRECT_OBJ_TAGS = {'obj'}
INDIRECT_OBJ_TAGS = {'iobj'}

# ============================================================================
# HELPERS
# ============================================================================
def get_subtree_indices(word_idx, sentence):
    """Iterative DFS — returns sorted list of token indices in subtree."""
    visited, stack = set(), [word_idx]
    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur)
        for w in sentence.words:
            if w.head == cur: stack.append(w.idx)
    return sorted(visited)


def get_construction_type(sentence):
    """Label sentence as SOV / DOSV / IOSV."""
    root = sentence.root_word
    if root is None: return "unknown"
    preverbal = sorted(
        [w for w in sentence.words if w.head==root.idx and w.idx<root.idx and w.deprel!='punct'],
        key=lambda w: w.idx
    )
    if not preverbal: return "unknown"
    subj = next((w.idx for w in preverbal if w.deprel in SUBJ_TAGS), None)
    dobj = next((w.idx for w in preverbal if w.deprel in DIRECT_OBJ_TAGS), None)
    iobj = next((w.idx for w in preverbal if w.deprel in INDIRECT_OBJ_TAGS), None)
    if dobj is not None and (subj is None or dobj < subj): return "DOSV"
    if iobj is not None and (subj is None or iobj < subj): return "IOSV"
    return "SOV"

# ============================================================================
# VARIANT GENERATION
# ============================================================================
def generate_variants(sentence, sentence_idx):
    root = sentence.root_word
    if root is None: return []
    root_idx = root.idx

    reference_forms = [w.form for w in sentence.words]
    reference_str   = " ".join(reference_forms)

    preverbal_deps = sorted(
        [w for w in sentence.words if w.head==root_idx and w.idx<root_idx and w.deprel!='punct'],
        key=lambda w: w.idx
    )
    if len(preverbal_deps) < 2: return []

    # build subtree blocks
    blocks = []
    for dep in preverbal_deps:
        idx_set = get_subtree_indices(dep.idx, sentence)
        blocks.append(sorted([w for w in sentence.words if w.idx in idx_set], key=lambda w: w.idx))

    all_block_idx = {w.idx for block in blocks for w in block}
    fixed_tokens  = sorted([w for w in sentence.words if w.idx not in all_block_idx], key=lambda w: w.idx)
    construction  = get_construction_type(sentence)

    results, seen = [], set()
    for perm in itertools.permutations(blocks):
        new_tokens   = [w for block in perm for w in block] + fixed_tokens
        variant_str  = " ".join(w.form for w in new_tokens)
        variant_order= [w.idx for w in new_tokens]
        if variant_str in seen: continue
        seen.add(variant_str)
        results.append({
            'sentence_id':     sentence_idx,
            'sent_id':         sentence.sent_id,
            'reference':       reference_str,
            'variant':         variant_str,
            'variant_order':   variant_order,
            'is_reference':    variant_str == reference_str,
            'construction_type': construction,
            'tokens':          sentence.words,
        })

    ref_entries   = [r for r in results if r['is_reference']]
    nonref_entries= [r for r in results if not r['is_reference']]
    if len(nonref_entries) > MAX_NON_REF_VARIANTS:
        nonref_entries = random.sample(nonref_entries, MAX_NON_REF_VARIANTS)
    return ref_entries + nonref_entries

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" SUBTREE-AWARE VARIANT GENERATION")
    print("=" * 70 + "\n")

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found."); return 1

    with open(INPUT_FILE, 'rb') as f:
        sentences = pickle.load(f)
    print(f"Loaded {len(sentences):,} reference sentences")
    print(f"Max non-reference variants per sentence: {MAX_NON_REF_VARIANTS}\n")

    all_results, refs_with_variants, total_variants = [], 0, 0
    for idx, sent in enumerate(tqdm(sentences, desc="Generating variants")):
        entries = generate_variants(sent, idx)
        if entries:
            refs_with_variants += 1
            total_variants += sum(1 for e in entries if not e['is_reference'])
            all_results.extend(entries)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_results, f)

    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    print(f"  Reference sentences with variants : {refs_with_variants:,}")
    print(f"  Total entries (ref + variants)    : {len(all_results):,}")
    print(f"  Non-reference variants            : {total_variants:,}")
    print(f"  Avg variants per reference        : {total_variants/max(refs_with_variants,1):.1f}")
    print(f"\n  Paper target (2023): ~184,818  |  Your result: {total_variants:,}")

    ref_entries = [r for r in all_results if r['is_reference']]
    print("\n  Construction type distribution:")
    for ctype, count in sorted(Counter(r['construction_type'] for r in ref_entries).items()):
        print(f"    {ctype:<8}: {count:>5,}  ({count/len(ref_entries)*100:.1f}%)")

    print(f"\n  Saved -> {OUTPUT_FILE}\n")
    return 0


if __name__ == "__main__":
    exit(main())