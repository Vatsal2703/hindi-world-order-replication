#!/usr/bin/env python3
"""
Script 18: Vanilla LSTM Surprisal Scoring

Scores every variant in all_variants_final.pkl using the trained EMILLE LSTM.
variant_order contains word POSITION INDICES (1, 2, ..., N), not vocab IDs.
We must convert them to actual word forms using the reference sentence word map,
then look up vocab IDs.

Output: final_scored_variants.pkl
  - Same length and order as all_variants_final.pkl
  - Each entry: {sent_id, avg_surprisal, is_reference}
"""

import sys
import os
import torch
import torch.nn as nn
import pickle
import math
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}")

# Load vocab
with open("./data/models/emille_vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)

class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=200, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)

model = VanillaLSTM(len(vocab)).to(device)
model.load_state_dict(torch.load("./data/models/emille_base_lstm.pt", map_location=device))
model.eval()

def get_surprisal(tokens):
    """
    Compute total sentence surprisal (bits) for a list of word-form tokens.
    Uses <s> as initial context token.
    """
    if not tokens:
        return 0.0
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).to(device)
    with torch.no_grad():
        logits = model(input_ids)          # [1, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs[0, torch.arange(len(ids)), target_ids]
        total_surp = -gathered.sum().item() / math.log(2)
    return total_surp

# Load data
with open("./data/processed/all_variants_final.pkl", 'rb') as f:
    all_variants = pickle.load(f)
print(f"Loaded {len(all_variants):,} variants")

# Load reference sentences for word form lookup
with open("./data/processed/replication_filtered_sentences.pkl", 'rb') as f:
    ref_sents = pickle.load(f)
sent_lookup = {s.sent_id: s for s in ref_sents}
print(f"Loaded {len(sent_lookup):,} reference sentences")

# Sanity check on a single sentence
sample = next(v for v in all_variants if v['is_reference'])
sent_obj = sent_lookup.get(sample['sent_id'])
word_map = {w.idx: w.form for w in sent_obj.words}
tokens = [word_map.get(idx, '<UNK>') for idx in sample['variant_order']]
print(f"\nSanity check — sent_id: {sample['sent_id']}")
print(f"  variant_order: {sample['variant_order']}")
print(f"  word forms:    {tokens}")
print(f"  total surprisal: {get_surprisal(tokens):.4f} bits\n")

# Score all variants
results = []
skipped = 0
for item in tqdm(all_variants, desc="Scoring variants"):
    sent_obj = sent_lookup.get(item['sent_id'])
    if sent_obj is None:
        results.append({
            'sent_id': item['sent_id'],
            'avg_surprisal': 0.0,
            'is_reference': item['is_reference']
        })
        skipped += 1
        continue

    word_map = {w.idx: w.form for w in sent_obj.words}
    tokens = [word_map.get(idx, '<UNK>') for idx in item['variant_order']]
    surp = get_surprisal(tokens)
    # Paper: "summed word-level surprisal of all words" → store total, not per-word avg
    results.append({
        'sent_id': item['sent_id'],
        'avg_surprisal': surp,
        'is_reference': item['is_reference']
    })

print(f"\nSkipped (no sent_obj): {skipped}")

os.makedirs("./data/results", exist_ok=True)
with open("./data/results/final_scored_variants.pkl", 'wb') as f:
    pickle.dump(results, f)

print(f"Done! Scored {len(results):,} variants.")
print("Results saved to: ./data/results/final_scored_variants.pkl")
print("Next: Re-run scripts/merge_all_feature.py")
