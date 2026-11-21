#!/usr/bin/env python
# v1.2 (2026-11-02): renumbered from 38_score_lstm_v2.py; naming only, no logic change.
# v1.1 (2026-11-02): renamed from 38_lstm_v2.py; naming only, no logic change.
"""
29_score_lstm_v2.py  (v1.2)

Scores the HUTB reference sentences and the DISCO fluent utterances with the
RETRAINED EMILLE LSTM (data/models/emille_lstm_v2.pt, from 28_retrain_lstm.py).
Scoring only -- no regression here (that is script 30).

  HUTB : data/results/MTP2_refs_with_emille_freq.csv   (reference rows, 'sentence')
  DISCO: data/disco/disco_dl.csv                       (tokens read from 'parse')

Surprisal = -log2 p(word | previous words), summed, then divided by the number
of tokens (punctuation included), same convention as the trigram measure.

OUTPUT: data/results/lstm_v2_scores.csv
        columns: corpus, key, n_tokens, lstm_v2_total_bits, lstm_v2_pw

RUN (same environment you trained in):
    python scripts/29_score_lstm_v2.py
    python scripts/29_score_lstm_v2.py --model data/models/emille_lstm_v2.pt
"""
import os, re, math, pickle, argparse
import pandas as pd, torch, torch.nn as nn

VERSION = "1.2"
print(f"=== 29_score_lstm_v2.py v{VERSION} ===")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="data/models/emille_lstm_v2.pt")
args = ap.parse_args()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

vocab = pickle.load(open(os.path.join(BASE, "data/models/emille_vocab.pkl"), "rb"))
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
model.load_state_dict(torch.load(os.path.join(BASE, args.model), map_location=device))
model.eval()
print("loaded", args.model)

def total_bits(tokens):
    if not tokens:
        return 0.0
    ids = [vocab.get(t, UNK) for t in tokens]
    inp = torch.tensor([S] + ids[:-1]).unsqueeze(0).to(device)
    tgt = torch.tensor(ids).to(device)
    with torch.no_grad():
        lg, _ = model(inp)
        lp = torch.log_softmax(lg, -1)[0, torch.arange(len(ids)), tgt]
    return -lp.sum().item() / math.log(2)

rows = []

# ---- HUTB references -------------------------------------------------------
h = pd.read_csv(os.path.join(BASE, "data/results/MTP2_refs_with_emille_freq.csv"),
                encoding="utf-8-sig")
h = h[h.is_reference == 1]
print(f"HUTB references: {len(h):,}")
for i, (_, r) in enumerate(h.iterrows(), 1):
    toks = str(r["sentence"]).split()
    b = total_bits(toks)
    rows.append(("HUTB", r["sent_id"], len(toks), round(b, 4), round(b / max(len(toks), 1), 4)))
    if i % 500 == 0:
        print(f"  HUTB {i:,}", flush=True)

# ---- DISCO fluent utterances ----------------------------------------------
PAT = re.compile(r"^(\d+):(.*)->(\d+):(.+)$")
d = pd.read_csv(os.path.join(BASE, "data/disco/disco_dl.csv"), encoding="utf-8-sig")
print(f"DISCO utterances: {len(d):,}")
bad = 0
for i, (_, r) in enumerate(d.iterrows(), 1):
    toks = []
    for piece in str(r["parse"]).split(" "):
        m = PAT.match(piece)
        if m:
            toks.append(m.group(2))
    if not toks:
        bad += 1
        continue
    b = total_bits(toks)
    rows.append(("DISCO", r["Sentence Number"], len(toks), round(b, 4), round(b / len(toks), 4)))
    if i % 1000 == 0:
        print(f"  DISCO {i:,}", flush=True)
print("DISCO utterances without a usable parse:", bad)

out = pd.DataFrame(rows, columns=["corpus", "key", "n_tokens", "lstm_v2_total_bits", "lstm_v2_pw"])
path = os.path.join(BASE, "data/results/lstm_v2_scores.csv")
out.to_csv(path, index=False, encoding="utf-8-sig")
print(out.groupby("corpus").lstm_v2_pw.describe().round(2))
print("wrote", path)
print(f"DONE v{VERSION}")