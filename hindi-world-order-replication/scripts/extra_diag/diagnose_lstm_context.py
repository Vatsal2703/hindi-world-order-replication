#!/usr/bin/env python
# v1.2 (2026-11-02): renumbered from 36_diagnose_lstm_context.py; naming only, no logic change.
# v1.1 (2026-11-02): renamed from 36_lstm_diagnostic.py; naming only, no logic change.
"""
diagnose_lstm_context.py  (v1.2)

WHY: in MTP2_raw_dataset_full.csv, ~26% of sentences get exactly the same
lstm_surprisal for EVERY word order, and 55% of scrambled variants equal their
reference. A left-to-right LSTM should almost never do that. This script checks
whether the saved EMILLE LSTM is order-sensitive.

RUN (conda `base` env, needs torch):
    python scripts/diagnose_lstm_context.py
"""
import os, pickle, random, math
import pandas as pd, torch, torch.nn as nn

VERSION = "1.2"
print(f"=== diagnose_lstm_context.py v{VERSION} ===")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cpu")

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

vocab = pickle.load(open(os.path.join(BASE, "data/models/emille_vocab.pkl"), "rb"))
model = VanillaLSTM(len(vocab)).to(device)
sd = torch.load(os.path.join(BASE, "data/models/emille_base_lstm.pt"), map_location=device)
print("load_state_dict:", model.load_state_dict(sd))
model.eval()

def per_token(tokens):
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    inp = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0)
    tgt = torch.tensor(ids)
    with torch.no_grad():
        logits, _ = model(inp)
        lp = torch.log_softmax(logits, -1)[0, torch.arange(len(ids)), tgt]
    return (-lp / math.log(2)).tolist()

d = pd.read_csv(os.path.join(BASE, "data/results/MTP2_raw_dataset_full.csv"), encoding="utf-8-sig")
refs = d[d.is_reference == 1].head(300)
random.seed(0)
same = 0
for k, (_, r) in enumerate(refs.iterrows()):
    toks = r.sentence.split()
    a = sum(per_token(toks))
    sh = toks[:]; random.shuffle(sh)
    b = sum(per_token(sh))
    same += abs(a - b) < 1e-3
    if k < 3:
        print("\nREF :", " ".join(toks)); print("  bits/token:", [round(x, 1) for x in per_token(toks)])
        print("SHUF:", " ".join(sh));  print("  bits/token:", [round(x, 1) for x in per_token(sh)])
        print(f"  total ref {a:.3f} vs shuffled {b:.3f}")
print(f"\nshuffled total == reference total in {same}/300 sentences")
print("If this is ~0/300 the LSTM is order-sensitive and the CSV values are the problem;")
print("if it is large, the saved LSTM itself ignores word order.")
