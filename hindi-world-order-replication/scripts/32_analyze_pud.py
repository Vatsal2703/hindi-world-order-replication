#!/usr/bin/env python3
# v1.3 (2026-11-02): renumbered from 43_analyze_pud.py; naming only, no logic change.
# v1.2 (2026-11-02): renamed from 43_pud_third_corpus.py; naming only, no logic change.
"""
32_analyze_pud.py  (v1.3)
  v1.1: loads the two n-gram models one after the other (both at once ran out of memory)

MTP-II on a THIRD corpus: Hindi-PUD (1,000 sentences, gold UD trees,
data_pud/hi_pud-ud-test.conllu). Same rules as HUTB and DISCO, so the two
cross-corpus findings (dominance ~ DL, valence ~ LSTM surprisal) and the
trigram sign conflict can be checked on new data.

Per sentence (reference word order only):
  emotion   NRC-VAD v2.1 Hindi, script 20 v2.6 rules (negator flips V and D
            of its gold head; intensifier x1.2 arousal on the next word)
  DL        sum |pos(head) - pos(dep)| over non-root words / tokens
  trigram   unsmoothed EMILLE trigram, tri->bi->uni backoff (script 20 v2.5)
  KN        Kneser-Ney trigram of MTP-I (script 07/09)
  LSTM      retrained EMILLE LSTM v2 (script 28) -- only if torch is
            installed; otherwise skipped with a message
  controls  log length + EMILLE log10 frequency per million (script 27 rule)
Model: z-scored OLS, HC3 SE: emotion ~ surprisal + DL + loglen + freq,
one surprisal measure per model (as script 30).

OUTPUTS  data_pud/results/pud_mtp2_scores.csv
         data_pud/results/pud_mtp2_regressions.csv
RUN:  python scripts/32_analyze_pud.py     (conda base for the LSTM part)
"""
import os, math, pickle
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from collections import defaultdict, namedtuple

VERSION = "1.3"
print(f"=== 32_analyze_pud.py v{VERSION} ===")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(BASE, *p)
OUT = R("data_pud", "results"); os.makedirs(OUT, exist_ok=True)

NEGATION = {"नहीं", "न", "मत", "ना"}
INTENSIFIERS = {"बहुत", "अत्यधिक", "ज़्यादा", "ज्यादा", "बेहद", "अत्यंत", "काफी", "इतना"}
W = namedtuple("W", "idx form head")

# ------------------------------------------------------------- read PUD ----
sents, cur, sid = [], [], None
for line in open(R("data_pud", "hi_pud-ud-test.conllu"), encoding="utf-8"):
    line = line.rstrip("\n")
    if line.startswith("# sent_id"):
        sid = line.split("=", 1)[1].strip()
    elif not line.strip():
        if cur: sents.append((sid, cur)); cur = []
    elif not line.startswith("#"):
        c = line.split("\t")
        if c[0].isdigit():
            cur.append(W(int(c[0]), c[1], int(c[6])))
if cur: sents.append((sid, cur))
print(f"PUD sentences: {len(sents):,}")

# -------------------------------------------- emotion (script 20 v2.6) ----
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
                v = (float(parts[1]) - 0.5) * 2.0; a = float(parts[2]); d = (float(parts[3]) - 0.5) * 2.0
                w = parts[4].split()[0]
            except (ValueError, IndexError):
                continue
            raw[w].append((v, a, d))
    return {w: tuple(sum(s[k] for s in sc) / len(sc) for k in range(3)) for w, sc in raw.items()}

def score_emotion(words, lex):
    forms = [w.form.strip() for w in words]
    pos = {w.idx: i for i, w in enumerate(words)}
    flipped = {pos.get(words[i].head) for i, f in enumerate(forms)
               if f in NEGATION and pos.get(words[i].head) not in (None, i)}
    V, A, D, n = [], [], [], 0
    for i, f in enumerate(forms):
        if f in NEGATION or f in INTENSIFIERS or f not in lex: continue
        v, a, d = lex[f]; n += 1
        if i in flipped: v, d = -v, -d
        if i > 0 and forms[i - 1] in INTENSIFIERS: a = min(1.0, a * 1.2)
        V.append(v); A.append(a); D.append(d)
    if not V: return 0.0, 0.5, 0.0, 0.0
    return np.mean(V), np.mean(A), np.mean(D), n / max(len(words), 1)

lex = load_vad_lexicon(R("data", "processed", "Hindi-NRC-VAD-Lexicon.txt"))
print(f"lexicon entries: {len(lex):,}")

# ----------------------------------------------------- n-gram models ----
TRI = pickle.load(open(R("data", "models", "trigram_model_emille.pkl"), "rb"))
TOT = max(sum(TRI["unigrams"].values()), 1)
def p_backoff(w1, w2, w3):
    a, b = TRI["trigrams"].get((w1, w2, w3), 0), TRI["bigrams"].get((w1, w2), 0)
    if a > 0 and b > 0: return a / b
    a, b = TRI["bigrams"].get((w2, w3), 0), TRI["unigrams"].get(w2, 0)
    if a > 0 and b > 0: return a / b
    u = TRI["unigrams"].get(w3, 0)
    return u / TOT if u > 0 else 1e-10
def tri_bits(toks):
    h, t = ["<s>", "<s>"], 0.0
    for w in toks:
        t += -math.log2(min(max(p_backoff(h[-2], h[-1], w), 1e-10), 1.0)); h.append(w)
    return t

KN = None
def kn_prob(w1, w2, w3, m=None):
    m = m or KN
    D_ = m["D"]; tb = m["total_bigram_types"]
    p_uni = m["unigram_continuation_count"].get(w3, 0) / tb if tb > 0 else 1e-10
    bc, bct = m["bigram_continuation_count"].get((w2, w3), 0), m["bigram_cont_total"].get(w2, 0)
    p_bi = (max(bc - D_, 0) / bct + D_ * m["bigram_unique_followers"].get(w2, 0) / bct * p_uni) if bct > 0 else p_uni
    tc, bi = m["trigrams"].get((w1, w2, w3), 0), m["bigrams"].get((w1, w2), 0)
    p_tri = (max(tc - D_, 0) / bi + D_ * m["bigram_followers_count"].get((w1, w2), 0) / bi * p_bi) if bi > 0 else p_bi
    return max(p_tri, 1e-10)
def kn_bits(toks):
    t = ["<s>", "<s>"] + toks + ["</s>"]
    return sum(-math.log2(kn_prob(t[i - 2], t[i - 1], t[i])) for i in range(2, len(t)))

logf = {w: math.log10(c / TOT * 1e6) for w, c in TRI["unigrams"].items() if c > 0}
med_lf = float(np.median(list(logf.values())))
PRE = {sid: (tri_bits([w.form for w in ws]), float(np.mean([logf.get(w.form, med_lf) for w in ws]))) for sid, ws in sents}
import gc
del TRI, logf; gc.collect()
print("unsmoothed trigram + frequency scored; loading the Kneser-Ney model")
KN = pickle.load(open(R("data", "models", "trigram_model_blind.pkl"), "rb"))

# -------------------------------------------------------- LSTM (opt.) ----
lstm_bits = None
try:
    import torch, torch.nn as nn
    vocab = pickle.load(open(R("data", "models", "emille_vocab.pkl"), "rb"))
    class VanillaLSTM(nn.Module):
        def __init__(s, V, e=200, h=200, L=2):
            super().__init__(); s.embedding = nn.Embedding(V, e, padding_idx=0)
            s.lstm = nn.LSTM(e, h, num_layers=L, batch_first=True, dropout=0.2); s.fc = nn.Linear(h, V)
        def forward(s, x):
            o, h = s.lstm(s.embedding(x)); return s.fc(o), h
    LM = VanillaLSTM(len(vocab)); LM.load_state_dict(torch.load(R("data", "models", "emille_lstm_v2.pt"), map_location="cpu")); LM.eval()
    UNK, S = vocab["<UNK>"], vocab["<s>"]
    def lstm_bits(toks):
        ids = [vocab.get(t, UNK) for t in toks]
        with torch.no_grad():
            lg, _ = LM(torch.tensor([S] + ids[:-1]).unsqueeze(0))
            lp = torch.log_softmax(lg, -1)[0, torch.arange(len(ids)), torch.tensor(ids)]
        return -lp.sum().item() / math.log(2)
    print("LSTM v2 loaded (CPU)")
except ImportError:
    print("torch not installed here: LSTM models skipped (run this script in conda base to add them)")

# ------------------------------------------------------------- score ----
rows = []
for sid, ws in sents:
    toks = [w.form for w in ws]; n = len(toks)
    pos = {w.idx: i for i, w in enumerate(ws)}
    dl = sum(abs(pos[w.idx] - pos[w.head]) for w in ws if w.head > 0 and w.head in pos)
    v, a, d, cov = score_emotion(ws, lex)
    rows.append(dict(sent_id=sid, n_words=n, valence=v, arousal=a, dominance=d, coverage=cov,
                     dl=dl / n, tri=PRE[sid][0] / n, kn=kn_bits(toks) / n,
                     lstm=(lstm_bits(toks) / n) if lstm_bits else np.nan,
                     freq=PRE[sid][1], loglen=math.log(n),
                     sentence=" ".join(toks)))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, "pud_mtp2_scores.csv"), index=False, encoding="utf-8-sig")
print(f"scored {len(P):,}; no lexicon match in {(P.coverage == 0).sum()} sentences")
print(P[["valence", "arousal", "dominance", "dl", "tri", "kn", "lstm"]].describe().round(3).to_string())
have = ["tri", "kn"] + (["lstm"] if P.lstm.notna().all() else [])
print("correlations between surprisal measures:\n", P[have].corr().round(2).to_string())

# -------------------------------------------------------- regressions ----
Z = P.copy()
for c in ["valence", "arousal", "dominance", "dl", "loglen", "freq"] + have:
    Z[c + "_z"] = (Z[c] - Z[c].mean()) / Z[c].std()
res = []
for emo in ["valence", "arousal", "dominance"]:
    for s in have:
        m = smf.ols(f"{emo}_z ~ {s}_z + dl_z + loglen_z + freq_z", Z).fit(cov_type="HC3")
        for t in (s, "dl"):
            res.append(("PUD", emo, s, t, len(Z), round(m.params[t + "_z"], 4), round(m.pvalues[t + "_z"], 4)))
out = pd.DataFrame(res, columns=["corpus", "emotion", "model", "predictor", "n", "beta", "p"])
out.to_csv(os.path.join(OUT, "pud_mtp2_regressions.csv"), index=False)
t = out.copy(); t["cell"] = t.beta.map("{:+.3f}".format) + " (p=" + t.p.map("{:.3f}".format) + ")"
print("\n===== PUD (standardised beta, p) =====")
print(t.pivot_table(index=["emotion", "predictor"], columns="model", values="cell", aggfunc="first").to_string())
print(f"wrote {OUT}/pud_mtp2_scores.csv, pud_mtp2_regressions.csv\nDONE v{VERSION}")
