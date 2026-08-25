#!/usr/bin/env python3
"""
Script 26: Raw Per-Sentence Dataset — ALL FEATURES  — v2.0

(Renumbered from script 28 -- the original script 26, a three-surprisal-only
version, has been removed; this is its replacement.) Changes requested by
supervisor after the meeting:

  1. ALL seven MTP-I features in the raw dataset (not just the three surprisals)
  2. A unique_id per row, formatted  train-s2-ref / train-s2-var1 / train-s2-var2
  3. prev_sentence_id and prev_sentence columns
  4. Per-word (length-normalised) surprisal written to a SEPARATE file
  5. (v2.1) construction_type is now computed PER ROW from that row's own
     ordering; the sentence's attested type is kept as ref_construction_type.
     Previously every variant inherited the reference's label, which was wrong
     because reordering is precisely what changes the construction type.
  6. (v2.3) pcfg_surprisal is now scored IN THIS SCRIPT, the same way every
     other feature is: no more separate scoring/merge script run (the old
     30_pcfg_scoring.py / 31_merge_pcfg_scores.py have been removed -- this
     script fully replaces what they did). It reuses the already-trained
     5-fold Berkeley grammars from
     13_pcfg_surprisal.py (fold_N.gr + fold_N/test_sent_ids.pkl -- training
     those is still a one-time prerequisite, done long before this script
     runs) and calls the Berkeley Parser as a batched subprocess, the same
     protocol the old scoring script used. Results are cached forever in
     data/results/.pcfg_scores_checkpoint.pkl keyed by unique_id, so once a
     sentence has been scored it is never re-parsed on a later run -- and
     since that cache already covers the full corpus, a normal run will not
     invoke Java at all.

Still one row per SENTENCE (reference and each variant), carrying ABSOLUTE
scores rather than reference-minus-variant differences.

------------------------------------------------------------------------------
IMPORTANT — provenance of each feature. Please read before using column values.
------------------------------------------------------------------------------
EXACT (same code path as the earlier pipeline):
    trigram_surprisal      counts + unigram back-off
    lstm_surprisal         the saved EMILLE LSTM
    adaptive_surprisal     that LSTM after one SGD step on the preceding sentence
    dep_len                sum of |pos(head) - pos(dependent)|, unambiguous
    info_status            ported from calculate_information_status_score in
                           src/features/basic_features.py, called the way
                           09_workflow_trigram.py actually called it (no
                           context_sentence): +1 if the first of the two
                           preverbal constituents is a pronoun and the second
                           is not, -1 for the reverse, else 0. Verified
                           against all_features_with_emotion.pkl: matches the
                           78.09%-zero, mean +0.0706 info_status_diff profile
                           exactly (label==1 rows).

RECONSTRUCTED (the MTP-I code stored only differences, so the absolute-value
formulation is re-derived here and should be verified against the original):
    lex_rept_surprisal     trigram interpolated with a cache built from the
                           preceding sentence. LEX_REPT_LAMBDA below sets the
                           mixing weight; the old difference values sat very
                           close to plain trigram, implying a small cache weight.

SCORED VIA SUBPROCESS (own cache, see step [5]):
    pcfg_surprisal         Berkeley Parser sentence log-likelihood, negated,
                           using the fold's grammar for that sentence's
                           sent_id (5-fold CV, same fold assignment as the
                           MTP-I pipeline). Needs Java + the jar at
                           tools/berkeley-parser/berkeleyParser.jar and the
                           grammars/folds under data/processed/ -- these are
                           long-standing training artifacts, not produced by
                           this script.

RUN IN `base` CONDA ENV (needs torch + MPS + Java for the PCFG step):
    python scripts/26_raw_dataset_full.py                # full run
    python scripts/26_raw_dataset_full.py --limit 50     # quick test

Checkpoints every 100 sentences (feature loop) and every 20 batches (PCFG
scoring, cached separately by unique_id); re-run to resume, --restart to
start the feature loop over (the PCFG cache is untouched by --restart).

Output: data/results/MTP2_raw_dataset_full.csv
        data/results/MTP2_raw_perword.csv
"""

VERSION = "2.3"

import os
import sys
import math
import copy
import pickle
import argparse
import subprocess
from collections import defaultdict, Counter
from itertools import permutations

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

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


BASE        = find_base()
SENTS_IN    = os.path.join(BASE, "data", "processed", "replication_filtered_sentences.pkl")
TRIGRAM_IN  = os.path.join(BASE, "data", "models", "trigram_model.pkl")
LSTM_IN     = os.path.join(BASE, "data", "models", "emille_base_lstm.pt")
VOCAB_IN    = os.path.join(BASE, "data", "models", "emille_vocab.pkl")
VAD_IN      = os.path.join(BASE, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt")
FOLDS_DIR   = os.path.join(BASE, "data", "processed", "pcfg_folds")
GRAMMAR_DIR = os.path.join(BASE, "data", "processed", "pcfg_grammars")
BERKELEY_JAR = os.path.join(BASE, "tools", "berkeley-parser", "berkeleyParser.jar")
OUT_MAIN    = os.path.join(BASE, "data", "results", "MTP2_raw_dataset_full.csv")
OUT_PERWORD = os.path.join(BASE, "data", "results", "MTP2_raw_perword.csv")
CKPT        = os.path.join(BASE, "data", "results", ".raw_full_checkpoint.pkl")
PCFG_CKPT   = os.path.join(BASE, "data", "results", ".pcfg_scores_checkpoint.pkl")

CHECKPOINT_EVERY = 100
MAX_VARIANTS     = 99
ADAPT_LR         = 2.0
LEX_REPT_LAMBDA  = 0.10     # weight on the cache term; see header note

N_PCFG_FOLDS      = 5
PCFG_BATCH_SIZE   = 2000
PCFG_CKPT_EVERY   = 20      # batches
PCFG_BATCH_TIMEOUT = 600    # seconds, matches 13_pcfg_surprisal.py / 30_pcfg_scoring.py

NEGATION     = {"नहीं", "न", "मत", "ना"}
INTENSIFIERS = {"बहुत", "अत्यधिक", "ज़्यादा", "ज्यादा", "बेहद", "अत्यंत", "काफी", "इतना"}
SUBJECT_RELS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# LSTM
# ============================================================================

class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=200, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.lstm(x)
        return self.fc(out), hidden


def lstm_surprisal(model, tokens, vocab):
    if not tokens:
        return 0.0
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    inp = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    tgt = torch.tensor(ids).to(device)
    with torch.no_grad():
        logits, _ = model(inp)
        logp = torch.log_softmax(logits, dim=-1)
        got = logp[0, torch.arange(len(ids)), tgt]
        return -got.sum().item() / math.log(2)


def adapt_model(base_model, context_tokens, vocab, lr=ADAPT_LR):
    if not context_tokens:
        return base_model
    adapted = copy.deepcopy(base_model)
    adapted.train()
    opt = optim.SGD(adapted.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in context_tokens]
    inp = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    tgt = torch.tensor(ids).unsqueeze(0).to(device)
    opt.zero_grad()
    logits, _ = adapted(inp)
    crit(logits.transpose(1, 2), tgt).backward()
    opt.step()
    adapted.eval()
    return adapted

# ============================================================================
# TRIGRAM  (+ cache-interpolated variant for lexical repetition)
# ============================================================================

def trigram_total_tokens(model):
    return max(sum(model.get("unigrams", {}).values()), 1)


def _word_prob(model, w1, w2, tok, total_uni):
    tri = model["trigrams"].get((w1, w2, tok), 0)
    bi  = model["bigrams"].get((w1, w2), 0)
    uni = model["unigrams"].get(tok, 0)
    if tri > 0 and bi > 0:
        return tri / bi
    if uni > 0:
        return uni / total_uni
    return 1e-10


def trigram_surprisal(model, tokens, total_uni):
    """Absolute trigram surprisal, in bits."""
    total, hist = 0.0, ["<s>", "<s>"]
    for tok in tokens:
        p = _word_prob(model, hist[-2], hist[-1], tok, total_uni)
        total += -math.log2(min(max(p, 1e-10), 1.0))
        hist.append(tok)
    return total


def lex_rept_surprisal(model, tokens, total_uni, cache_counts, cache_size,
                       lam=LEX_REPT_LAMBDA):
    """
    Trigram interpolated with a unigram cache built from the preceding sentence:
        P = (1 - lam) * P_trigram  +  lam * P_cache
    Words repeated from the preceding sentence become cheaper, which is the
    lexical-repetition effect. With no preceding sentence this reduces exactly
    to the plain trigram.
    """
    if cache_size == 0:
        return trigram_surprisal(model, tokens, total_uni)
    total, hist = 0.0, ["<s>", "<s>"]
    for tok in tokens:
        p_tri = _word_prob(model, hist[-2], hist[-1], tok, total_uni)
        p_cache = cache_counts.get(tok, 0) / cache_size
        p = (1.0 - lam) * p_tri + lam * p_cache
        total += -math.log2(min(max(p, 1e-10), 1.0))
        hist.append(tok)
    return total

# ============================================================================
# PCFG  (Berkeley Parser, 5-fold grammars trained by 13_pcfg_surprisal.py)
# ============================================================================

def load_pcfg_fold_mapping():
    """sent_id -> fold index, from the existing test_sent_ids.pkl per fold
    (same fold assignment 30_pcfg_scoring.py used, all sent_ids covered)."""
    sent_id_to_fold = {}
    for fold_idx in range(N_PCFG_FOLDS):
        ids_file = os.path.join(FOLDS_DIR, f"fold_{fold_idx}", "test_sent_ids.pkl")
        with open(ids_file, "rb") as f:
            for sid in pickle.load(f):
                sent_id_to_fold[sid] = fold_idx
    return sent_id_to_fold


def score_pcfg_batch(sentences, grammar_file):
    """Same subprocess protocol as 13_pcfg_surprisal.py / 30_pcfg_scoring.py:
    stdin newline-joined sentences, '-sentence_likelihood' returns
    'score\\ttree' lines (one per sentence, log P(w))."""
    input_text = "\n".join(sentences) + "\n"
    cmd = ["java", "-Xmx2g", "-jar", BERKELEY_JAR, "-gr", grammar_file,
           "-sentence_likelihood", "-maxLength", "200"]
    try:
        result = subprocess.run(cmd, input=input_text, capture_output=True,
                                text=True, timeout=PCFG_BATCH_TIMEOUT)
        if result.returncode != 0:
            print(f"    Parser error (exit {result.returncode}): {result.stderr[:300]}")
            return [None] * len(sentences)
        scores = []
        for line in result.stdout.strip("\n").split("\n"):
            line = line.strip()
            if not line:
                scores.append(None)
                continue
            parts = line.split("\t", 1)
            try:
                scores.append(float(parts[0]))
            except ValueError:
                scores.append(None)
        while len(scores) < len(sentences):
            scores.append(None)
        return scores[:len(sentences)]
    except subprocess.TimeoutExpired:
        print(f"    Batch timed out ({PCFG_BATCH_TIMEOUT}s)")
        return [None] * len(sentences)


def score_pcfg_for_rows(rows, sent_id_to_fold, grammars):
    """
    Fills pcfg_surprisal = -log P(sentence) for every row, resuming from
    PCFG_CKPT (unique_id -> score-or-None), which persists across ALL runs
    of this script (never cleared by --restart, unlike CKPT) since parsing
    is by far the slowest step. Returns {unique_id: score-or-None}.
    """
    scores = {}
    if os.path.exists(PCFG_CKPT):
        with open(PCFG_CKPT, "rb") as f:
            scores = pickle.load(f)
        print(f"    pcfg cache: {len(scores):,} unique_ids already scored")

    remaining = [r for r in rows if r["unique_id"] not in scores]
    print(f"    {len(remaining):,} of {len(rows):,} rows still need pcfg_surprisal")
    if not remaining:
        return scores

    by_fold = defaultdict(list)
    for r in remaining:
        by_fold[sent_id_to_fold.get(r["sent_id"])].append(r)

    batches_done = 0
    for fold_idx, fold_rows in sorted(by_fold.items(),
                                      key=lambda kv: (kv[0] is None, kv[0])):
        if fold_idx is None or fold_idx not in grammars:
            for r in fold_rows:
                scores[r["unique_id"]] = None
            continue
        grammar = grammars[fold_idx]
        n_batches = (len(fold_rows) + PCFG_BATCH_SIZE - 1) // PCFG_BATCH_SIZE
        for b in range(n_batches):
            batch = fold_rows[b * PCFG_BATCH_SIZE:(b + 1) * PCFG_BATCH_SIZE]
            uids = [r["unique_id"] for r in batch]
            sents = [r["sentence"] for r in batch]
            batch_scores = score_pcfg_batch(sents, grammar)
            for uid, s in zip(uids, batch_scores):
                scores[uid] = (-s) if s is not None else None
            batches_done += 1
            print(f"    fold {fold_idx}  batch {b + 1}/{n_batches}  "
                  f"({len(scores):,}/{len(rows):,} total scored)")
            if batches_done % PCFG_CKPT_EVERY == 0:
                with open(PCFG_CKPT, "wb") as f:
                    pickle.dump(scores, f)

    with open(PCFG_CKPT, "wb") as f:
        pickle.dump(scores, f)
    return scores

# ============================================================================
# STRUCTURAL FEATURES
# ============================================================================

def get_subtree(sent, head_idx):
    result = {head_idx}
    for w in sent.words:
        if w.head == head_idx and w.idx != head_idx:
            result |= get_subtree(sent, w.idx)
    return result


def dependency_length(sent, order):
    """Sum of |position(head) - position(dependent)| under this ordering."""
    pos = {idx: p for p, idx in enumerate(order)}
    total = 0
    for w in sent.words:
        if w.head > 0 and not w.is_root() and w.idx in pos and w.head in pos:
            total += abs(pos[w.idx] - pos[w.head])
    return total


def info_status_score(sent, order):
    """
    Given-New (+1) vs New-Given (-1) ordering of the first two preverbal
    constituents, where GIVEN = the word is a pronoun (PRON).

    Ported from calculate_information_status_score in
    src/features/basic_features.py -- but called the way the pipeline that
    actually produced the MTP-I feature file called it. That function takes
    an optional context_sentence for lemma-vs-previous-sentence matching,
    but 09_workflow_trigram.py (which feeds pairwise_features_trigram_blind.pkl
    -> all_features_final.pkl / all_features_with_emotion.pkl) never passes
    one, so the lemma branch is always dead there and "given" collapses to
    "is a pronoun". Confirmed against all_features_with_emotion.pkl: label==1
    info_status_diff is 78.09% zero with mean +0.0706, which this PRON-only
    rule reproduces; the fuller lemma-matching version (as used by
    05_workflow_dl_is.py, which did NOT feed the final dataset) does not.
    """
    preverbal = sent.get_preverbal_constituents()
    pre_idxs = {w.idx for w in preverbal}
    ordered_pre_idxs = [idx for idx in order if idx in pre_idxs]
    if len(ordered_pre_idxs) < 2:
        return 0

    def check_given(idx):
        w = next((w for w in sent.words if w.idx == idx), None)
        return w is not None and w.upos == "PRON"

    first_given = check_given(ordered_pre_idxs[0])
    second_given = check_given(ordered_pre_idxs[1])
    if first_given and not second_given:
        return 1
    if not first_given and second_given:
        return -1
    return 0


def construction_from_order(sent, order):
    """
    Construction type for a SPECIFIC ordering.

    v2.1 FIX: previously the type was computed once from the reference parse and
    copied onto every variant. But reordering a sentence is exactly what changes
    its construction type, so variants carried the wrong label. Example:

        train-s2-ref   इसे नवाब शाहजेहन ने बनवाया था ।   obj before subj -> DOSV
        train-s2-var1  नवाब शाहजेहन ने इसे बनवाया था ।   subj before obj -> SOV

    Both were labelled DOSV. This function reads the ordering it is given, so
    each row now reports its own type. The sentence-level (attested) type is
    kept separately as ref_construction_type.
    """
    if getattr(sent, "root_idx", None) is None:
        return "UNKNOWN"
    pos = {idx: p for p, idx in enumerate(order)}
    root_pos = pos.get(sent.root_idx)
    if root_pos is None:
        return "UNKNOWN"

    children = sent.get_children(sent.root_idx)
    preverbal = [w for w in children
                 if w.idx in pos and pos[w.idx] < root_pos and w.upos != "PUNCT"]
    preverbal.sort(key=lambda w: pos[w.idx])

    subj = obj = iobj = None
    for i, w in enumerate(preverbal):
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

# ============================================================================
# EMOTION
# ============================================================================

def load_vad_lexicon(path):
    raw = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        sample = f.readline()
        delim = "\t" if "\t" in sample else " "
        f.seek(0)
        for line in f:
            parts = [p.strip() for p in line.strip().split(delim) if p.strip()]
            if len(parts) < 5:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
            if parts[0].lower() == "english" or parts[4] == "Hindi":
                continue
            try:
                v = (float(parts[1]) - 0.5) * 2.0
                a = float(parts[2])
                w = parts[4].split()[0]
            except (ValueError, IndexError):
                continue
            raw[w].append((v, a))
    return {w: (sum(s[0] for s in sc) / len(sc),
                sum(s[1] for s in sc) / len(sc)) for w, sc in raw.items()}


def score_emotion(words, lexicon):
    vals, aros, n_matched = [], [], 0
    for i, w in enumerate(words):
        form = w.form.strip()
        if form not in lexicon:
            continue
        v, a = lexicon[form]
        n_matched += 1
        prev = words[i - 1].form.strip() if i > 0 else ""
        if prev in NEGATION:
            v = -v
        if prev in INTENSIFIERS:
            a = min(1.0, a * 1.2)
        vals.append(v)
        aros.append(a)
    if not vals:
        return 0.0, 0.5, 0.0
    return (sum(vals) / len(vals), sum(aros) / len(aros),
            n_matched / max(len(words), 1))

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="first N sentences only")
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print(f" SCRIPT 26: RAW DATASET — ALL FEATURES  (v{VERSION})")
    print("=" * 78)

    print(f"\n[1] Loading models (device: {device})...")
    for p in (SENTS_IN, TRIGRAM_IN, LSTM_IN, VOCAB_IN, VAD_IN, BERKELEY_JAR):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing required file: {p}")
    if not os.path.isdir(FOLDS_DIR):
        sys.exit(f"ERROR: missing required directory: {FOLDS_DIR}")

    with open(VOCAB_IN, "rb") as f:
        vocab = pickle.load(f)
    base_model = VanillaLSTM(len(vocab)).to(device)
    base_model.load_state_dict(torch.load(LSTM_IN, map_location=device))
    base_model.eval()
    with open(TRIGRAM_IN, "rb") as f:
        trigram = pickle.load(f)
    tri_total = trigram_total_tokens(trigram)
    lexicon = load_vad_lexicon(VAD_IN)
    sent_id_to_fold = load_pcfg_fold_mapping()
    pcfg_grammars = {fold_idx: os.path.join(GRAMMAR_DIR, f"fold_{fold_idx}.gr")
                     for fold_idx in range(N_PCFG_FOLDS)
                     if os.path.exists(os.path.join(GRAMMAR_DIR, f"fold_{fold_idx}.gr"))}
    print(f"    vocab {len(vocab):,} | lexicon {len(lexicon):,} | "
          f"trigram {tri_total:,} tokens | pcfg folds {sorted(pcfg_grammars.keys())}")
    if device.type != "mps":
        print("    NOTE: not on MPS — use the `base` env for a much faster run.")

    with open(SENTS_IN, "rb") as f:
        sentences = pickle.load(f)
    if args.limit:
        sentences = sentences[: args.limit]
        print(f"    LIMIT: first {len(sentences)} sentences")
    ordered = [s.sent_id for s in sentences]
    lookup = {s.sent_id: s for s in sentences}
    print(f"[2] Loaded {len(sentences):,} reference sentences")

    rows, start_i = [], 0
    if os.path.exists(CKPT) and not args.restart:
        with open(CKPT, "rb") as f:
            ck = pickle.load(f)
        rows, start_i = ck["rows"], ck["next_index"]
        print(f"[3] Resuming: {start_i:,} sentences done, {len(rows):,} rows")
    else:
        print("[3] Starting fresh")

    print("\n[4] Scoring every ordering...")
    print("    features: dep_len, info_status, trigram, lstm, adaptive, lex_rept")
    print("    pcfg_surprisal deferred to step [5] (batched Berkeley Parser)\n")

    for i in range(start_i, len(sentences)):
        sent = sentences[i]
        ref_order, variants = build_orders(sent)
        if ref_order is None:
            continue

        words = sent.words
        n_words = len(words)

        # target-sentence emotion (constant across this sentence's orderings)
        t_val, t_aro, t_cov = score_emotion(words, lexicon)

        # preceding sentence: emotion, LSTM context, cache
        prev = lookup.get(ordered[i - 1]) if i > 0 else None
        if prev is not None:
            p_val, p_aro, p_cov = score_emotion(prev.words, lexicon)
            prev_tokens = [w.form for w in prev.words]
            prev_id = prev.sent_id
            prev_text = getattr(prev, "text", None) or " ".join(prev_tokens)
            cache_counts = Counter(prev_tokens)
            cache_size = sum(cache_counts.values())
        else:
            p_val, p_aro, p_cov = 0.0, 0.5, 0.0
            prev_tokens, prev_id, prev_text = None, "", ""
            cache_counts, cache_size = {}, 0

        adapted = adapt_model(base_model, prev_tokens, vocab) if prev_tokens else base_model
        ref_ctype = construction_from_order(sent, ref_order)   # attested type

        for vid, order in enumerate([ref_order] + variants):
            toks = [words[idx - 1].form for idx in order if 0 < idx <= n_words]
            uid = f"{sent.sent_id}-{'ref' if vid == 0 else f'var{vid}'}"
            rows.append({
                "unique_id":          uid,
                "sent_id":            sent.sent_id,
                "variant_id":         vid,
                "is_reference":       1 if vid == 0 else 0,
                "construction_type":  construction_from_order(sent, order),
                "ref_construction_type": ref_ctype,
                "n_words":            n_words,
                "sentence":           " ".join(toks),
                "prev_sentence_id":   prev_id,
                "prev_sentence":      prev_text,
                # --- the seven features, absolute ---
                "dep_len":            dependency_length(sent, order),
                "info_status":        info_status_score(sent, order),
                "trigram_surprisal":  round(trigram_surprisal(trigram, toks, tri_total), 4),
                "lstm_surprisal":     round(lstm_surprisal(base_model, toks, vocab), 4),
                "adaptive_surprisal": round(lstm_surprisal(adapted, toks, vocab), 4),
                "lex_rept_surprisal": round(lex_rept_surprisal(
                                          trigram, toks, tri_total,
                                          cache_counts, cache_size), 4),
                "pcfg_surprisal":     None,      # filled in step [5] below
                # --- emotion ---
                "prev_valence":       round(p_val, 4),
                "prev_arousal":       round(p_aro, 4),
                "prev_coverage":      round(p_cov, 4),
                "tgt_valence":        round(t_val, 4),
                "tgt_arousal":        round(t_aro, 4),
                "tgt_coverage":       round(t_cov, 4),
            })

        done = i + 1
        if done % CHECKPOINT_EVERY == 0 or done == len(sentences):
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            with open(CKPT, "wb") as f:
                pickle.dump({"rows": rows, "next_index": done}, f)
            print(f"    {done:>6,}/{len(sentences):,} sentences "
                  f"({100.0*done/len(sentences):5.1f}%)   rows: {len(rows):,}")

    print("\n[5] Scoring pcfg_surprisal (Berkeley Parser, batched by fold)...")
    pcfg_scores = score_pcfg_for_rows(rows, sent_id_to_fold, pcfg_grammars)
    missing_pcfg = []
    for r in rows:
        val = pcfg_scores.get(r["unique_id"])
        if val is None:
            missing_pcfg.append(r["unique_id"])
            r["pcfg_surprisal"] = ""
        else:
            r["pcfg_surprisal"] = round(val, 4)
    if missing_pcfg:
        print(f"\n    WARNING: {len(missing_pcfg):,} unique_ids could not be scored "
              f"(no fold/grammar or parser failure), e.g. {missing_pcfg[:5]}")

    # ---- Save main file ----------------------------------------------------
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_MAIN), exist_ok=True)
    df.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")

    # ---- Save the separate per-word file -----------------------------------
    PER_WORD_SRC = ["trigram_surprisal", "lstm_surprisal",
                    "adaptive_surprisal", "lex_rept_surprisal", "dep_len",
                    "pcfg_surprisal"]
    pw = df[["unique_id", "sent_id", "variant_id", "is_reference",
             "construction_type", "ref_construction_type", "n_words"]].copy()
    for c in PER_WORD_SRC:
        vals = pd.to_numeric(df[c], errors="coerce")
        pw[c + "_per_word"] = (vals / df["n_words"].where(df["n_words"] > 0)).round(4)
    pw.to_csv(OUT_PERWORD, index=False, encoding="utf-8-sig")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "-" * 78)
    print(" SUMMARY")
    print("-" * 78)
    print(f"  Rows        : {len(df):,}   "
          f"(references {int(df['is_reference'].sum()):,}, "
          f"variants {int((1 - df['is_reference']).sum()):,})")
    print(f"  Columns     : {len(df.columns)}")
    print(f"  Example ids : {', '.join(df['unique_id'].head(3))}")

    ref = df[df["is_reference"] == 1]
    var = df[df["is_reference"] == 0]
    print("\n  Reference vs variant (references should score LOWER):")
    for c in ["trigram_surprisal", "lstm_surprisal", "adaptive_surprisal",
              "lex_rept_surprisal", "pcfg_surprisal", "dep_len", "info_status"]:
        r, v = pd.to_numeric(ref[c], errors="coerce").mean(), pd.to_numeric(var[c], errors="coerce").mean()
        note = "" if c != "info_status" else "   (higher is the preferred pattern)"
        print(f"    {c:<20s} ref {r:9.2f}   var {v:9.2f}   diff {r - v:+9.2f}{note}")

    print("\n  Construction type — reference rows (attested distribution):")
    vc_ref = ref["construction_type"].value_counts()
    for ct in ["SOV", "DOSV", "IOSV", "UNKNOWN"]:
        n = vc_ref.get(ct, 0)
        if n:
            print(f"    {ct:>8s}: {n:>6,} ({100*n/len(ref):5.1f}%)")

    print("\n  Construction type — variant rows (each row's own ordering):")
    vc_var = var["construction_type"].value_counts()
    for ct in ["SOV", "DOSV", "IOSV", "UNKNOWN"]:
        n = vc_var.get(ct, 0)
        if n:
            print(f"    {ct:>8s}: {n:>6,} ({100*n/len(var):5.1f}%)")
    changed = int((var["construction_type"] != var["ref_construction_type"]).sum())
    print(f"    ({changed:,} of {len(var):,} variants differ from their "
          f"sentence's attested type — expected, since reordering changes it)")

    neg = {c: int((pd.to_numeric(df[c], errors="coerce") < 0).sum()) for c in
           ["trigram_surprisal", "lstm_surprisal", "adaptive_surprisal",
            "lex_rept_surprisal", "pcfg_surprisal"]}
    if any(neg.values()):
        print("\n  WARNING: negative surprisal (impossible):")
        for c, n in neg.items():
            if n:
                print(f"    {c}: {n:,}")
    else:
        print("\n  Check passed: no negative surprisal values.")

    print(f"\n  Saved -> {OUT_MAIN}")
    print(f"  Saved -> {OUT_PERWORD}   (per-word, separate file)")
    if missing_pcfg:
        print(f"\n  NOTE: {len(missing_pcfg):,} rows are missing pcfg_surprisal "
              f"(no fold/grammar coverage or a parser failure) — re-run this "
              f"script to retry them; already-scored rows are cached and won't "
              f"be re-parsed.")
    else:
        print("\n  pcfg_surprisal scored for all rows (Berkeley Parser, cached "
              f"in {PCFG_CKPT}).")

    if os.path.exists(CKPT):
        os.remove(CKPT)
        print("  Checkpoint cleared.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()