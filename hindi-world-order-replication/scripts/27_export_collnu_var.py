#!/usr/bin/env python3
"""
Script 27: Export References and Variants to CoNLL-U  — v1.0

(Renumbered from script 29.)

Writes every ordering (the attested sentence and each of its variants) as a
proper CoNLL-U block, so any ordering can be inspected in a dependency viewer
such as urd2.let.rug.nl/~kleiweg/conllu/.

Why this is well-defined: a variant is a permutation of the SAME words with the
SAME dependency relations. Only the linear positions change. So each token keeps
its FORM, LEMMA, UPOS, XPOS, FEATS and DEPREL; the ID and HEAD fields are
renumbered to the new positions.

Header lines emitted per sentence:
    # sent_id   train-s2-ref / train-s2-var1 / ...   (matches the raw dataset)
    # text      the tokens in THIS ordering
    # translit  regenerated for THIS ordering
    # construction_type  this ordering's own type
    # is_reference       1 or 0

NOTE on translit: a common mistake is to copy the reference's transliteration
onto every variant, so the text and translit disagree. Here translit is rebuilt
from this ordering's tokens.

RUN IN `digit` OR `base` CONDA ENV (no torch needed):

    python scripts/27_export_collnu_var.py                       # everything
    python scripts/27_export_collnu_var.py --limit 20            # first 20 sentences
    python scripts/27_export_collnu_var.py --sent-ids train-s2 train-s6
    python scripts/27_export_collnu_var.py --refs-only           # references only

Output: data/results/MTP2_all_orderings.conllu
"""

VERSION = "1.0"

import os
import sys
import pickle
import argparse
from itertools import permutations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))
from parsers.ud_parser import Sentence, Word  # noqa: F401  (needed to unpickle)


def find_base():
    d = SCRIPT_DIR
    while d and d != os.path.dirname(d):
        if os.path.basename(d) == "hindi-world-order-replication":
            return d
        if os.path.isdir(os.path.join(d, "hindi-world-order-replication")):
            return os.path.join(d, "hindi-world-order-replication")
        d = os.path.dirname(d)
    return "."


BASE     = find_base()
SENTS_IN = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
OUT_FILE = os.path.join(BASE, "data", "results", "MTP2_all_orderings.conllu")

MAX_VARIANTS = 99
SUBJECT_RELS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}

# ============================================================================
# TRANSLITERATION (Devanagari -> ISO 15919-ish, matching UD conventions)
# ============================================================================

_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    "ड़": "ṛ", "ढ़": "ṛh", "क़": "q", "ख़": "x", "ग़": "ġ",
    "ज़": "z", "फ़": "f",
}
_VOW = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū",
    "ऋ": "ṛ", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_MATRA = {
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū", "ृ": "ṛ",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_MISC = {"ं": "ṃ", "ँ": "ṁ", "ः": "ḥ", "्": "", "़": ""}


def translit_word(w):
    """Approximate ISO transliteration with inherent-'a' handling."""
    out, i = [], 0
    while i < len(w):
        ch = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch in _CONS:
            out.append(_CONS[ch])
            if nxt in _MATRA:
                out.append(_MATRA[nxt]); i += 2; continue
            if nxt == "्":                      # virama: no inherent vowel
                i += 2; continue
            if nxt in ("ं", "ँ", "ः"):
                out.append("a"); out.append(_MISC[nxt]); i += 2; continue
            out.append("a")                     # inherent vowel
            i += 1; continue
        if ch in _VOW:
            out.append(_VOW[ch]); i += 1; continue
        if ch in _MISC:
            out.append(_MISC[ch]); i += 1; continue
        if ch in _MATRA:
            out.append(_MATRA[ch]); i += 1; continue
        out.append(ch); i += 1
    return "".join(out)


def translit_sentence(tokens):
    return " ".join(translit_word(t) for t in tokens)

# ============================================================================
# ORDERINGS
# ============================================================================

def get_subtree(sent, head_idx):
    result = {head_idx}
    for w in sent.words:
        if w.head == head_idx and w.idx != head_idx:
            result |= get_subtree(sent, w.idx)
    return result


def build_orders(sent):
    if getattr(sent, "root_idx", None) is None:
        return None, []
    children = sent.get_children(sent.root_idx)
    preverbal = sorted([w for w in children
                        if w.idx < sent.root_idx and w.upos != "PUNCT"],
                       key=lambda w: w.idx)
    if len(preverbal) < 2:
        return None, []
    blocks = [sorted(get_subtree(sent, pw.idx)) for pw in preverbal]
    all_pre = set(i for b in blocks for i in b)
    post_v = [w.idx for w in sent.words if w.idx not in all_pre]
    ref = [i for b in blocks for i in b] + post_v
    variants, seen = [], {tuple(ref)}
    for perm in permutations(range(len(blocks))):
        order = [i for k in perm for i in blocks[k]] + post_v
        if tuple(order) not in seen:
            seen.add(tuple(order))
            variants.append(order)
            if len(variants) >= MAX_VARIANTS:
                break
    return ref, variants


def construction_from_order(sent, order):
    """Construction type for THIS ordering (not the reference's)."""
    if getattr(sent, "root_idx", None) is None:
        return "UNKNOWN"
    pos = {idx: p for p, idx in enumerate(order)}
    root_pos = pos.get(sent.root_idx)
    if root_pos is None:
        return "UNKNOWN"
    children = sent.get_children(sent.root_idx)
    pre = [w for w in children
           if w.idx in pos and pos[w.idx] < root_pos and w.upos != "PUNCT"]
    pre.sort(key=lambda w: pos[w.idx])
    subj = obj = iobj = None
    for i, w in enumerate(pre):
        if w.deprel in SUBJECT_RELS and subj is None:
            subj = i
        elif w.deprel == "obj" and obj is None:
            obj = i
        elif w.deprel == "iobj" and iobj is None:
            iobj = i
    if subj is None:
        return "UNKNOWN"
    if obj is not None and obj < subj:
        return "DOSV"
    if iobj is not None and iobj < subj:
        return "IOSV"
    return "SOV"

# ============================================================================
# CoNLL-U BLOCK
# ============================================================================

def field(value, default="_"):
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def conllu_block(sent, order, uid, is_ref):
    """
    Render one ordering as CoNLL-U.

    Positions are renumbered 1..n following `order`; each token's HEAD is
    remapped to its head's NEW position. Relations themselves are unchanged,
    since a variant is a permutation of the same tree.
    """
    words = sent.words
    n = len(words)
    valid = [idx for idx in order if 0 < idx <= n]

    old_to_new = {old: new for new, old in enumerate(valid, start=1)}
    toks = [words[idx - 1].form for idx in valid]

    lines = [
        f"# sent_id = {uid}",
        f"# text = {' '.join(toks)}",
        f"# translit = {translit_sentence(toks)}",
        f"# construction_type = {construction_from_order(sent, order)}",
        f"# is_reference = {1 if is_ref else 0}",
    ]

    for new_id, old_idx in enumerate(valid, start=1):
        w = words[old_idx - 1]
        if w.head == 0 or getattr(w, "is_root", lambda: False)():
            head_new, deprel = 0, "root"
        else:
            head_new = old_to_new.get(w.head, 0)
            deprel = field(w.deprel, "dep")
            if head_new == 0:            # head fell outside this ordering
                deprel = "root"
        lines.append("\t".join([
            str(new_id),
            field(w.form),
            field(getattr(w, "lemma", None)),
            field(getattr(w, "upos", None)),
            field(getattr(w, "xpos", None)),
            field(getattr(w, "feats", None)),
            str(head_new),
            deprel,
            "_",
            "_",
        ]))
    return "\n".join(lines) + "\n"

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N sentences")
    ap.add_argument("--sent-ids", nargs="+", default=None,
                    help="export only these sentence ids")
    ap.add_argument("--refs-only", action="store_true",
                    help="skip variants, export references only")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print(f" SCRIPT 27: EXPORT ORDERINGS TO CoNLL-U  (v{VERSION})")
    print("=" * 78)

    if not os.path.exists(SENTS_IN):
        sys.exit(f"ERROR: {SENTS_IN} not found.")
    with open(SENTS_IN, "rb") as f:
        sentences = pickle.load(f)

    if args.sent_ids:
        wanted = set(args.sent_ids)
        sentences = [s for s in sentences if s.sent_id in wanted]
        print(f"\nFiltered to {len(sentences)} requested sentence(s)")
    if args.limit:
        sentences = sentences[: args.limit]
        print(f"LIMIT: first {len(sentences)} sentences")
    print(f"\nLoaded {len(sentences):,} reference sentences")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_ref = n_var = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for i, sent in enumerate(sentences):
            ref_order, variants = build_orders(sent)
            if ref_order is None:
                continue

            out.write(conllu_block(sent, ref_order, f"{sent.sent_id}-ref", True))
            out.write("\n")
            n_ref += 1

            if not args.refs_only:
                for vid, order in enumerate(variants, start=1):
                    out.write(conllu_block(sent, order,
                                           f"{sent.sent_id}-var{vid}", False))
                    out.write("\n")
                    n_var += 1

            if (i + 1) % 200 == 0:
                print(f"  {i+1:>6,}/{len(sentences):,} sentences  "
                      f"({n_ref + n_var:,} blocks)")

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print("\n" + "-" * 78)
    print(" SUMMARY")
    print("-" * 78)
    print(f"  References : {n_ref:,}")
    print(f"  Variants   : {n_var:,}")
    print(f"  Total      : {n_ref + n_var:,} blocks   ({size_mb:.1f} MB)")
    print(f"\n  Saved -> {args.out}")
    print("""
  To inspect a few in a viewer, export a small file first, e.g.

      python scripts/27_export_collnu_var.py --sent-ids train-s2 \\
             --out data/results/sample.conllu

  then paste that file into urd2.let.rug.nl/~kleiweg/conllu/ — the full
  export is far too large to paste.
""")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()