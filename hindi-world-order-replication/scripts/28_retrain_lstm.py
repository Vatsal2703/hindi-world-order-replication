#!/usr/bin/env python
# v1.3 (2026-11-02): renumbered from 37_retrain_lstm.py; naming only, no logic change.
# v1.2 (2026-11-02): references to other scripts updated to their new names; naming only, no logic change.
"""
28_retrain_lstm.py  (v1.3)

WHY: scripts/diagnose_lstm_context.py showed the saved EMILLE LSTM
(data/models/emille_base_lstm.pt) gives every word the same surprisal wherever
it appears, i.e. it behaves like a unigram model and ignores context.
Likely causes in 08_train_lstm.py: SGD lr=20 with NO gradient clipping (training
diverges / collapses), no <s> in the training input, batch size 4.

THIS SCRIPT trains a replacement with the same architecture (2 layers, 200
embed, 200 hidden, same vocab) but:
  - Adam lr 1e-3, gradient-norm clipping 1.0
  - <s> prepended and </s> appended (matches how scoring feeds the model)
  - batch size 64, length-bucketed
  - 2% held-out set; reports perplexity vs a UNIGRAM baseline so we can SEE
    that context is being used
  - saves the best epoch to data/models/emille_lstm_v2.pt  (the old model is
    NOT overwritten)
  - finishes with the order-sensitivity check from script extra_diag/diagnose_lstm_context.py

RUN (conda `base` env):
    python scripts/28_retrain_lstm.py --max-sents 100000 --epochs 1   # quick test
    python scripts/28_retrain_lstm.py                                 # full, 3 epochs
"""
import os, sys, math, time, pickle, random, argparse
from collections import Counter
import torch, torch.nn as nn, torch.optim as optim

VERSION = "1.3"  # v1.1: --resume (continue from the saved best) and per-epoch LR decay
print(f"=== 28_retrain_lstm.py v{VERSION} ===")

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max-sents", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--resume", action="store_true", help="continue from data/models/emille_lstm_v2.pt")
ap.add_argument("--decay", type=float, default=1.0, help="multiply LR by this after each epoch (e.g. 0.7)")
args = ap.parse_args()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE, "data/processed/emille_written_tokenized.pkl")
VOCAB = os.path.join(BASE, "data/models/emille_vocab.pkl")
OUT = os.path.join(BASE, "data/models/emille_lstm_v2.pt")
MAXLEN = 100
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
random.seed(0); torch.manual_seed(0)

vocab = pickle.load(open(VOCAB, "rb"))
V = len(vocab); UNK, S, E = vocab["<UNK>"], vocab["<s>"], vocab["</s>"]
corpus = pickle.load(open(CORPUS, "rb"))
if args.max_sents:
    corpus = corpus[:args.max_sents]
seqs = []
for sent in corpus:
    ids = [vocab.get(w, UNK) for w in sent][:MAXLEN - 2]
    if len(ids) >= 2:
        seqs.append([S] + ids + [E])
del corpus
random.shuffle(seqs)
n_val = max(1000, int(0.02 * len(seqs)))
val, train = seqs[:n_val], seqs[n_val:]
print(f"train {len(train):,}  val {len(val):,}  vocab {V:,}")

# unigram baseline perplexity on the held-out set (context-free reference point)
uni = Counter(t for s in train for t in s[1:])
tot = sum(uni.values())
nll = n = 0
for s in val:
    for t in s[1:]:
        nll += -math.log((uni.get(t, 0) + 1) / (tot + V)); n += 1
print(f"UNIGRAM baseline held-out perplexity: {math.exp(nll / n):.1f}")

def batches(data, bs, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    chunk = bs * 50
    out = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: len(data[j]))
        out += [part[k:k + bs] for k in range(0, len(part), bs)]
    if shuffle:
        random.shuffle(out)
    for b in out:
        L = max(len(data[j]) for j in b)
        x = torch.zeros(len(b), L, dtype=torch.long)
        for r, j in enumerate(b):
            x[r, :len(data[j])] = torch.tensor(data[j])
        yield x

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
opt = optim.Adam(model.parameters(), lr=args.lr)
crit = nn.CrossEntropyLoss(ignore_index=0)

def evaluate():
    model.eval(); nll = n = 0
    with torch.no_grad():
        for x in batches(val, 128, False):
            x = x.to(device)
            lg, _ = model(x[:, :-1])
            tg = x[:, 1:]
            l = nn.functional.cross_entropy(lg.reshape(-1, V), tg.reshape(-1),
                                            ignore_index=0, reduction="sum")
            nll += l.item(); n += (tg != 0).sum().item()
    return math.exp(nll / n)

best = float("inf")
if args.resume:
    model.load_state_dict(torch.load(OUT, map_location=device))
    best = evaluate()
    print(f"resumed from {OUT}: held-out perplexity {best:.1f}")
sched = optim.lr_scheduler.ExponentialLR(opt, gamma=args.decay)
for ep in range(1, args.epochs + 1):
    model.train(); t0 = time.time(); run = 0.0
    for step, x in enumerate(batches(train, args.batch, True), 1):
        x = x.to(device)
        opt.zero_grad()
        lg, _ = model(x[:, :-1])
        loss = crit(lg.reshape(-1, V), x[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item()
        if step % 500 == 0:
            print(f"  ep{ep} step {step}  train loss {run / 500:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            run = 0.0
    sched.step()
    ppl = evaluate()
    print(f"EPOCH {ep}: held-out perplexity {ppl:.1f}  (unigram baseline above)")
    if ppl < best:
        best = ppl; torch.save(model.state_dict(), OUT); print("  saved", OUT)

# order-sensitivity check (same idea as script extra_diag/diagnose_lstm_context.py)
model.load_state_dict(torch.load(OUT, map_location=device)); model.eval()
def total_bits(ids):
    inp = torch.tensor([S] + ids[:-1]).unsqueeze(0).to(device)
    with torch.no_grad():
        lp = torch.log_softmax(model(inp)[0], -1)[0, torch.arange(len(ids)), torch.tensor(ids).to(device)]
    return -lp.sum().item() / math.log(2)
same = 0; k = 0
for s in val[:300]:
    ids = s[1:-1]; sh = ids[:]; random.shuffle(sh)
    same += abs(total_bits(ids) - total_bits(sh)) < 1e-3; k += 1
print(f"order check: shuffled total == original total in {same}/{k} sentences (want ~0)")
print("DONE v" + VERSION)
