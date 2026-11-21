#!/usr/bin/env python3
# v1.5 (2026-11-02): renumbered from 31_score_disco.py; naming only, no logic change.
# v1.4 (2026-11-02): renamed from 31_disco_affect_as_information.py; naming only, no logic change.
"""
Script 23: DISCO Hindi — emotion, surprisal and disfluency per utterance — v1.5

Step 1 of 3 in the DISCO pipeline (23 -> 24 -> 25; run all with
scripts/run_disco_pipeline.sh).

Data: DISCO (Bhat, Jyothi & Bhattacharyya, Findings of EMNLP 2023), Hindi
part: 3,180 spoken-style commands to AI assistants (from PRESTO, Goel et al.
2023), each paired with a human-corrected fluent version (deletions only).

For every utterance this script computes, on the FLUENT sentence:
  - valence, arousal, dominance, coverage : NRC-VAD v2.1 Hindi lexicon with
        script 20's rules (valence/dominance rescaled to [-1, +1], negation
        flips valence and dominance, intensifiers scale arousal; see v1.3), so values
        are directly comparable to the HUTB master sheet
  - surprisal : trigram surprisal per word, EMILLE trigram model,
        trigram -> bigram -> unigram backoff (script 20 v2.5)
  - zipf      : mean word Zipf frequency (wordfreq 'hi'), used as a control
and from the disfluency annotation:
  - disfluent : 1 if Disfluency Type != FL (fluent), else 0
  - deletion  : share of tokens the annotator removed
  - repair    : 1 if type is C (correction) or FS (false start)
Utterances with no Devanagari word (no frequency available) are dropped
(9 of 3,180).

Input : data/disco/Hindi.xlsx                       (github.com/vineet2104/DISCO)
        data/processed/Hindi-NRC-VAD-Lexicon.txt
        data/models/trigram_model_emille.pkl
Output: data/results/disco_affect_scored.csv        (read by scripts 24 and 25)

Version history
  v1.0  scoring + affect-as-information models (disfluency outcomes)
  v1.1  added task-domain fixed effects and z-scored outcomes to the models
  v1.2  scoring only: the model block (and its outputs disco_affect_results.csv,
        disco_affect_by_type.csv, disco_affect_report.txt) was removed after
        the 1 Oct clean-up; disco_affect_scored.csv is unchanged. Runtime
        drops from ~45 min to ~1 min.
  v1.3  modifier fix (script 20 v2.6 rules): negators and intensifiers are no
        longer scored as emotion words themselves, and negation flips the
        word the negator attaches to in the cached parse (data/disco/
        disco_dl.csv) instead of whatever word follows it. Emotion columns
        (val, aro, dom, coverage) change; surprisal, zipf and disfluency do not.
"""
VERSION = "v1.5"

import math, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/results"
OUT.mkdir(parents=True, exist_ok=True)
print(f"=== Script 23: DISCO emotion + surprisal scoring {VERSION} ===")

# ---------------- 1. emotion scoring (same rules as script 20) --------------
NEGATION = {"नहीं", "न", "मत", "ना"}
INTENSIFIERS = {"बहुत", "अत्यधिक", "ज़्यादा", "ज्यादा", "बेहद", "अत्यंत", "काफी", "इतना"}

def load_vad_lexicon(path):
    raw = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        sample = f.readline(); delim = "\t" if "\t" in sample else " "; f.seek(0)
        for line in f:
            parts = [p.strip() for p in line.strip().split(delim) if p.strip()]
            if len(parts) < 5:
                parts = line.strip().split()
                if len(parts) < 5: continue
            if parts[0].lower() == "english" or parts[4] == "Hindi": continue
            try:
                v = (float(parts[1]) - 0.5) * 2.0; a = float(parts[2])
                d = (float(parts[3]) - 0.5) * 2.0; w = parts[4].split()[0]
            except (ValueError, IndexError):
                continue
            raw[w].append((v, a, d))
    return {w: tuple(sum(s[k] for s in sc) / len(sc) for k in range(3)) for w, sc in raw.items()}

PARSE_ITEM = re.compile(r"^(\d+):(.*)->(\d+):(.*)$")

def load_heads(path):
    """Sentence Number -> list of 1-based head ids (0 = root), from the cached
    spaCy parse of the fluent sentence (data/disco/disco_dl.csv, see script 24)."""
    heads = {}
    if not Path(path).exists():
        return heads
    for sn, ps in pd.read_csv(path)[["Sentence Number", "parse"]].itertuples(index=False):
        hs = []
        for item in str(ps).split(" "):
            m = PARSE_ITEM.match(item)
            hs.append(int(m.group(3)) if m else None)
        heads[sn] = hs
    return heads

def score_emotion(tokens, lex, heads=None):
    """Same rules as script 20 v2.6.
    - NEGATION / INTENSIFIERS words are modifiers only: they are never scored
      as emotion words themselves (they are in the lexicon, e.g. बेहद has
      valence -0.98 from the single entry 'disheartening').
    - Negation flips valence and dominance of the word the negator attaches
      to in the dependency parse (its head). Without a parse (heads=None) it
      falls back to the word right after the negator.
    - An intensifier scales arousal (x1.2, cap 1.0) of the word right after it."""
    toks = [t.strip() for t in tokens]
    flipped = set()
    for i, t in enumerate(toks):
        if t not in NEGATION:
            continue
        if heads is not None:
            h = heads[i]
            if h and 1 <= h <= len(toks) and h - 1 != i:
                flipped.add(h - 1)
        elif i + 1 < len(toks):
            flipped.add(i + 1)
    vals, aros, doms = [], [], []
    for i, t in enumerate(toks):
        if t in NEGATION or t in INTENSIFIERS or t not in lex:
            continue
        v, a, d = lex[t]
        if i in flipped:
            v, d = -v, -d
        if i > 0 and toks[i - 1] in INTENSIFIERS:
            a = min(1.0, a * 1.2)
        vals.append(v); aros.append(a); doms.append(d)
    if not vals:
        return 0.0, 0.5, 0.0, 0.0
    return (np.mean(vals), np.mean(aros), np.mean(doms), len(vals) / max(len(tokens), 1))

# ---------------- 2. trigram surprisal (same backoff as script 20) ----------
def _p(m, w1, w2, w3, tot):
    n, dn = m["trigrams"].get((w1, w2, w3), 0), m["bigrams"].get((w1, w2), 0)
    if n > 0 and dn > 0: return n / dn
    n, dn = m["bigrams"].get((w2, w3), 0), m["unigrams"].get(w2, 0)
    if n > 0 and dn > 0: return n / dn
    u = m["unigrams"].get(w3, 0)
    return u / tot if u > 0 else 1e-10

def trigram_surprisal(m, toks, tot):
    s, h = 0.0, ["<s>", "<s>"]
    for t in toks:
        s += -math.log2(min(max(_p(m, h[-2], h[-1], t, tot), 1e-10), 1.0)); h.append(t)
    return s

PUNCT = re.compile(r"([।?!,.|])")
def tokenize(s):
    return PUNCT.sub(r" \1 ", str(s)).split()
DEV = re.compile(r"[ऀ-ॿ]+")

# ---------------- 3. load DISCO ---------------------------------------------
df = pd.read_excel(ROOT / "data/disco/Hindi.xlsx")
df = df.rename(columns={"Disfluent Sentence": "disfl", "Hindi Fluent Sentence": "fluent", "Domain Label": "domain",
                        "Disfluency Type": "dtype"}).dropna(subset=["disfl", "fluent", "dtype", "domain"])
df["dtype"] = df.dtype.str.strip()
print(f"utterances: {len(df)}  types: {df.dtype.value_counts().to_dict()}")

lex = load_vad_lexicon(ROOT / "data/processed/Hindi-NRC-VAD-Lexicon.txt")
print(f"lexicon entries: {len(lex):,}; loading trigram model ...")
with open(ROOT / "data/models/trigram_model_emille.pkl", "rb") as f:
    tri = pickle.load(f)
tot = max(sum(tri.get("unigrams", {}).values()), 1)

# ---------------- 4. score every utterance ----------------------------------
ft = df.fluent.map(tokenize); dt = df.disfl.map(tokenize)
HEADS = load_heads(ROOT / "data/disco/disco_dl.csv")
def _heads_for(sn, toks):
    h = HEADS.get(sn)
    return h if h is not None and len(h) == len(toks) and None not in h else None
_hs = [_heads_for(sn, t) for sn, t in zip(df["Sentence Number"], ft)]
print(f"negation scope from parse: {sum(h is not None for h in _hs)}/{len(_hs)} utterances "
      f"(rest fall back to the next word)")
emo = [score_emotion(t, lex, h) for t, h in zip(ft, _hs)]
df["val"], df["aro"], df["dom"], df["coverage"] = zip(*emo)
df["n_fluent"] = ft.map(len); df["n_disfl"] = dt.map(len)
df["surprisal"] = [trigram_surprisal(tri, t, tot) / max(len(t), 1) for t in ft]
df["zipf"] = df.fluent.map(lambda s: np.mean([zipf_frequency(w, "hi") for w in DEV.findall(s)] or [np.nan]))
df["disfluent"] = (df.dtype != "FL").astype(int)
df["deletion"] = ((df.n_disfl - df.n_fluent) / df.n_disfl).clip(lower=0)
df["repair"] = df.dtype.isin(["C", "FS"]).astype(int)
df["log_len"] = np.log(df.n_fluent)
df["pos"] = df.val.clip(lower=0); df["neg"] = (-df.val).clip(lower=0)
df = df.dropna(subset=["zipf"]).reset_index(drop=True)
print(f"scored: {len(df)}; with >=1 emotion word: {(df.coverage > 0).sum()}")

# ---------------- 5. save ---------------------------------------------------
df.to_csv(OUT / "disco_affect_scored.csv", index=False)
print(f"saved {OUT / 'disco_affect_scored.csv'}")
print(f"=== done {VERSION} ===")
