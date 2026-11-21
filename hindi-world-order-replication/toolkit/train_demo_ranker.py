#!/usr/bin/env python3
# v1.1 (2026-11-02): script numbers in comments and messages updated; naming only, no logic change.
"""
toolkit/train_demo_ranker.py  (v1.1)

Trains the small word-order ranker used by the demo toolkit (toolkit/app.py).

Why a separate ranker: the full MTP-I model (script 15) also uses PCFG,
lexical-repetition and adaptive-LSTM surprisal, which need a Berkeley parser
and document context and cannot be computed live for a typed sentence. The
demo therefore uses the four features it CAN compute live:

    dep_len_diff      Temperley dependency length (script 05 definition)
    info_status_diff  given-new score against an optional previous sentence
    trigram_diff      EMILLE trigram surprisal, plain backoff (script 20/23 rules)
    lstm_diff         retrained EMILLE LSTM surprisal (emille_lstm_v2.pt)

All differences are reference minus variant (label 1) and variant minus
reference (label 0), as in script 14 (Joachims pairwise set-up). A logistic
regression with no intercept is fitted; its weights w turn into a per-variant
score s(x) = w . x, so the demo ranks every variant of a sentence by s.

Inputs (all already in the repo):
  data/processed/all_variants_final.pkl
  data/processed/replication_filtered_sentences.pkl
  data/results/final_scored_variants_v2lstm.pkl   (script 10 with emille_lstm_v2.pt)
  data/features/all_features_final_v2lstm.csv     (script 14 output, v2 LSTM)
  data/models/trigram_model_emille.pkl
Output:
  toolkit/ranker.json   weights, feature order, 10-fold CV accuracy

Run (any env with pandas + scikit-learn; no torch needed):
  python toolkit/train_demo_ranker.py
"""
import os, sys, json, math, pickle
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

VERSION = "1.1"
print(f"=== train_demo_ranker.py v{VERSION} ===")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

# ---- trigram surprisal: same rules as script 23 v1.3 (plain backoff) -------
def _p(m, w1, w2, w3, tot):
    n, dn = m["trigrams"].get((w1, w2, w3), 0), m["bigrams"].get((w1, w2), 0)
    if n > 0 and dn > 0: return n / dn
    n, dn = m["bigrams"].get((w2, w3), 0), m["unigrams"].get(w2, 0)
    if n > 0 and dn > 0: return n / dn
    u = m["unigrams"].get(w3, 0)
    return u / tot if u > 0 else 1e-10

def trigram_total(m, toks, tot):
    s, h = 0.0, ["<s>", "<s>"]
    for t in toks:
        s += -math.log2(min(max(_p(m, h[-2], h[-1], t, tot), 1e-10), 1.0)); h.append(t)
    return s

print("loading data ...")
variants = pickle.load(open("data/processed/all_variants_final.pkl", "rb"))
sents = {s.sent_id: s for s in pickle.load(open("data/processed/replication_filtered_sentences.pkl", "rb"))}
lstm = np.array([r["avg_surprisal"] for r in pickle.load(open("data/results/final_scored_variants_v2lstm.pkl", "rb"))])
feat = pd.read_csv("data/features/all_features_final_v2lstm.csv")
assert len(lstm) == len(variants)
print("loading trigram model (about 2 GB in memory) ...")
tri = pickle.load(open("data/models/trigram_model_emille.pkl", "rb"))
tot = max(sum(tri["unigrams"].values()), 1)

print("scoring variants with the trigram model ...")
tri_s = np.zeros(len(variants))
for i, v in enumerate(variants):
    s = sents.get(v["sent_id"])
    if s is None:
        continue
    wm = {w.idx: w.form for w in s.words}
    tri_s[i] = trigram_total(tri, [wm.get(k, "<UNK>") for k in v["variant_order"]], tot)

# ---- rebuild pairwise rows exactly like script 14 -----------------------------
by_sent = defaultdict(list)
for i, v in enumerate(variants):
    by_sent[v["sent_id"]].append(i)
rows = []
for sid, idxs in by_sent.items():
    ref = next((i for i in idxs if variants[i]["is_reference"]), None)
    if ref is None:
        continue
    for i in idxs:
        if i == ref:
            continue
        rows.append((sid, 1, lstm[ref] - lstm[i], tri_s[ref] - tri_s[i]))
        rows.append((sid, 0, lstm[i] - lstm[ref], tri_s[i] - tri_s[ref]))
p = pd.DataFrame(rows, columns=["sent_id", "label", "lstm_diff", "trigram_diff"])
p = p.sort_values(["sent_id", "label"], ascending=[True, False]).reset_index(drop=True)
p = p[p.sent_id.isin(set(feat.sent_id))].reset_index(drop=True)
print(f"rebuilt rows {len(p):,}   feature rows {len(feat):,}")
if len(p) != len(feat) or not (p.sent_id.values == feat.sent_id.values).all():
    sys.exit("ERROR: rows do not line up with all_features_final_v2lstm.csv -- tell Claude.")
ok = np.allclose(p.lstm_diff.values, feat.lstm_surprisal_diff.values, atol=1e-3)
print("alignment check (LSTM diff reproduced):", "PASS" if ok else "FAIL")
if not ok:
    sys.exit("ERROR: alignment check failed -- tell Claude.")
p["dep_len_diff"] = feat.dep_len_diff.values
p["info_status_diff"] = feat.info_status_diff.values

# ---- fit + 10-fold CV ---------------------------------------------------------
COLS = ["dep_len_diff", "info_status_diff", "trigram_diff", "lstm_diff"]
def cv(cols):
    X, y = p[cols].values, p.label.values
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y):
        m = LogisticRegression(max_iter=1000, fit_intercept=False).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return round((pred == y).mean() * 100, 2)

acc = {"DL alone": cv(["dep_len_diff"]), "IS alone": cv(["info_status_diff"]),
       "trigram alone": cv(["trigram_diff"]), "LSTM alone": cv(["lstm_diff"]),
       "all four": cv(COLS)}
for k, a in acc.items():
    print(f"  10-fold CV accuracy  {k:14s} {a:6.2f}%")
m = LogisticRegression(max_iter=1000, fit_intercept=False).fit(p[COLS].values, p.label.values)
out = {"version": VERSION, "features": ["dep_len", "info_status", "trigram_total_bits", "lstm_total_bits"],
       "weights": [float(w) for w in m.coef_[0]], "cv_accuracy": acc,
       "n_pairs": int(len(p)), "n_sentences": int(p.sent_id.nunique()),
       "note": "score(variant) = sum(weights * features); higher = more preferred order"}
os.makedirs("toolkit", exist_ok=True)
json.dump(out, open("toolkit/ranker.json", "w"), indent=2, ensure_ascii=False)
print("weights:", dict(zip(out["features"], [round(w, 4) for w in out["weights"]])))
print("wrote toolkit/ranker.json")
print(f"DONE v{VERSION}")
