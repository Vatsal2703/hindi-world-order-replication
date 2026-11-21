#!/usr/bin/env python
# v1.2 (2026-11-02): renumbered from 40_check_mtp1_lstm_v2.py; naming only, no logic change.
# v1.1 (2026-11-02): renamed from 40_mtp1_lstm_v2_check.py; naming only, no logic change.
"""
check_mtp1_lstm_v2.py  (v1.2)

MTP-I check with the RETRAINED LSTM.

The original vanilla LSTM (emille_base_lstm.pt) ignored context (see script extra_diag/diagnose_lstm_context.py),
which is the likely reason the replication's LSTM-alone accuracy was 90.27 %
against 94.01 % in Ranjan et al. (2022). This script:

  1. rescored every variant in all_variants_final.pkl with emille_lstm_v2.pt
     (same total-bits convention as script 10), batched per sentence;
  2. rebuilds the pairwise LSTM difference exactly as script 14 does
     (label 1: ref - var ; label 0: var - ref ; sorted by sent_id, label desc)
     and CHECKS the rebuild against the existing lstm_surprisal_diff column using
     the OLD scores, so we know the new column lines up row for row;
  3. reruns the 10-fold CV logistic regression (no intercept, as script 15) for
       f      : LSTM alone
       base2  : IS + DL + PCFG + trigram + LSTM          (no adaptive, no lex-rept)
       base1  : IS + DL + PCFG + lex-rept + trigram + LSTM
     with the OLD LSTM and the NEW LSTM side by side.
  Adaptive-LSTM (feature g) is NOT recomputed here: it needs a per-sentence
  fine-tuning pass and is reported as unchanged.

RUN (conda `base`, needs torch + scikit-learn):
    python scripts/check_mtp1_lstm_v2.py

OUTPUT: data/results/mtp1_lstm_v2_check.csv
"""
import os, sys, math, pickle
from collections import defaultdict
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

VERSION = "1.2"
print(f"=== check_mtp1_lstm_v2.py v{VERSION} ===")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

vocab = pickle.load(open("data/models/emille_vocab.pkl", "rb"))
V = len(vocab); UNK, S = vocab["<UNK>"], vocab["<s>"]

class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=200, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        out, h = self.lstm(self.embedding(x))
        return self.fc(out), h

model = VanillaLSTM(V).to(device)
model.load_state_dict(torch.load("data/models/emille_lstm_v2.pt", map_location=device))
model.eval()

# ---------------------------------------------------------------- data ----
all_variants = pickle.load(open("data/processed/all_variants_final.pkl", "rb"))
ref_path = "data/processed/replication_filtered_sentences.pkl"
if not os.path.exists(ref_path):
    ref_path = "data/processed/reference_sentences.pkl"
ref_sents = pickle.load(open(ref_path, "rb"))
sent_lookup = {s.sent_id: s for s in ref_sents}
old_scores = pickle.load(open("data/results/final_scored_variants.pkl", "rb"))
assert len(old_scores) == len(all_variants)
print(f"variants {len(all_variants):,}  sentences {len(sent_lookup):,}  ref file {ref_path}")

# ------------------------------------------------- score variants (new) ----
by_sent = defaultdict(list)                       # sent_id -> [variant indices]
for i, v in enumerate(all_variants):
    by_sent[v["sent_id"]].append(i)

new_bits = np.zeros(len(all_variants))
missing = 0
for sid, idxs in tqdm(by_sent.items(), desc="scoring"):
    sent = sent_lookup.get(sid)
    if sent is None:
        missing += len(idxs); continue
    wmap = {w.idx: w.form for w in sent.words}
    seqs = []
    for i in idxs:
        toks = [wmap.get(k, "<UNK>") for k in all_variants[i]["variant_order"]]
        seqs.append([vocab.get(t, UNK) for t in toks])
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):             # safety: score one by one
        for i, s in zip(idxs, seqs):
            ids = torch.tensor([S] + s[:-1]).unsqueeze(0).to(device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids)[0], -1)[0, torch.arange(len(s)), torch.tensor(s).to(device)]
            new_bits[i] = -lp.sum().item() / math.log(2)
        continue
    ids = torch.tensor([[S] + s[:-1] for s in seqs]).to(device)
    tgt = torch.tensor(seqs).to(device)
    with torch.no_grad():
        lp = torch.log_softmax(model(ids)[0], -1)
        got = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        bits = (-got.sum(1) / math.log(2)).cpu().numpy()
    for i, b in zip(idxs, bits):
        new_bits[i] = b
print("variants without a sentence object:", missing)
pickle.dump(new_bits, open("data/results/lstm_v2_variant_scores.pkl", "wb"))

# ------------------------------------- rebuild pairwise like script 14 ----
def pairwise(scores):
    rows = []
    for sid, idxs in by_sent.items():
        ref = next((i for i in idxs if all_variants[i]["is_reference"]), None)
        non = [i for i in idxs if not all_variants[i]["is_reference"]]
        if ref is None or not non:
            continue
        for i in non:
            rows.append((sid, 1, scores[ref] - scores[i]))
            rows.append((sid, 0, scores[i] - scores[ref]))
    d = pd.DataFrame(rows, columns=["sent_id", "label", "diff"])
    return d.sort_values(["sent_id", "label"], ascending=[True, False]).reset_index(drop=True)

old_arr = np.array([r["avg_surprisal"] for r in old_scores])
p_old, p_new = pairwise(old_arr), pairwise(new_bits)

feat = pd.read_csv("data/features/all_features_final.csv")
feat = feat.sort_values(["sent_id", "label"], ascending=[True, False], kind="stable").reset_index(drop=True)
print(f"feature rows {len(feat):,}   rebuilt rows {len(p_old):,}")
if len(feat) != len(p_old):
    sys.exit("ERROR: row counts differ -- cannot align safely. Stop and tell Claude.")
ok = np.allclose(feat["lstm_surprisal_diff"].values, p_old["diff"].values, atol=1e-3)
print("alignment check (old LSTM diff reproduced from pickles):", "PASS" if ok else "FAIL")
if not ok:
    sys.exit("ERROR: alignment check failed -- not using the new column. Tell Claude.")
feat["lstm_v2_surprisal_diff"] = p_new["diff"].values

# --------------------------------------------------------- classification ----
def cv_acc(df, cols):
    X, y = df[cols].values, df["label"].values
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, fit_intercept=False)
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return (pred == y).mean() * 100

core = ["info_status_diff", "dep_len_diff", "pcfg_surprisal_diff", "trigram_surprisal_diff"]
sets = {
    "f (LSTM alone)":        (["lstm_surprisal_diff"],                                   ["lstm_v2_surprisal_diff"], 94.01),
    "base2 (IS+DL+PCFG+tri+LSTM)": (core + ["lstm_surprisal_diff"],                      core + ["lstm_v2_surprisal_diff"], 95.06),
    "base1 (base2 + lex-rept)":    (core + ["lex_rept_surprisal_diff", "lstm_surprisal_diff"],
                                    core + ["lex_rept_surprisal_diff", "lstm_v2_surprisal_diff"], 95.05),
}
out = []
for name, (old_cols, new_cols, paper) in sets.items():
    a_old, a_new = cv_acc(feat, old_cols), cv_acc(feat, new_cols)
    out.append((name, round(a_old, 2), round(a_new, 2), paper))
    print(f"{name:32s} old LSTM {a_old:6.2f}   new LSTM {a_new:6.2f}   paper {paper:6.2f}", flush=True)
pd.DataFrame(out, columns=["model", "old_lstm_acc", "new_lstm_acc", "paper_acc"]).to_csv(
    "data/results/mtp1_lstm_v2_check.csv", index=False)
print("wrote data/results/mtp1_lstm_v2_check.csv")
print(f"DONE v{VERSION}")
