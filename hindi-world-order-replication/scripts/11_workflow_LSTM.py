#!/usr/bin/env python3
"""
Script 16: Adaptive LSTM Surprisal (Discourse Predictability)

Paper Section 3.1 Feature 7:
  For each test sentence N, adapt the base LSTM to sentence N-1 (preceding context),
  then compute surprisal for every variant of sentence N using that adapted model.
  Adaptive learning rate = 2 (matches paper Table 1).

Output: adaptive_lstm_scores.pkl
  - Same length and order as all_variants_final.pkl
  - Each entry: {sent_id, adaptive_surprisal, is_reference}
  - Script 20 zips this directly with final_scored_variants.pkl
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import copy
import math
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}")

# ============================================================================
# MODEL (must match 17_train_LSTM.py exactly)
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

# ============================================================================
# HELPERS
# ============================================================================
def get_surprisal(model, tokens, vocab):
    """Total sentence surprisal in bits."""
    model.eval()
    if not tokens:
        return 0.0
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).to(device)
    with torch.no_grad():
        logits, _ = model(input_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs[0, torch.arange(len(ids)), target_ids]
        return -gathered.sum().item() / math.log(2)


def adapt_model(base_model, tokens, vocab, lr=2.0):
    """
    Fine-tune a COPY of base_model on one sentence using SGD.
    Paper uses adaptive learning rate = 2 (Section 3.1, Table 1).
    """
    adapted = copy.deepcopy(base_model)
    adapted.train()
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if not tokens:
        return adapted
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).unsqueeze(0).to(device)
    optimizer.zero_grad()
    logits, _ = adapted(input_ids)
    loss = criterion(logits.transpose(1, 2), target_ids)
    loss.backward()
    optimizer.step()
    adapted.eval()
    return adapted

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

with open("./data/models/emille_vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)

with open("./data/processed/replication_filtered_sentences.pkl", 'rb') as f:
    all_ref_sents = pickle.load(f)

with open("./data/processed/all_variants_final.pkl", 'rb') as f:
    all_variants = pickle.load(f)

base_model = VanillaLSTM(len(vocab)).to(device)
base_model.load_state_dict(torch.load(
    "./data/models/emille_base_lstm.pt", map_location=device))
base_model.eval()

print(f"  Vocab size: {len(vocab)}")
print(f"  Reference sentences (discourse order): {len(all_ref_sents)}")
print(f"  Total variants: {len(all_variants)}")

# ============================================================================
# BUILD LOOKUPS
# ============================================================================
# sent_id -> sentence object
sent_lookup = {s.sent_id: s for s in all_ref_sents}

# Discourse order: index -> sent_id, sent_id -> discourse index
ordered_sent_ids = [s.sent_id for s in all_ref_sents]
sent_id_to_disc_idx = {sid: i for i, sid in enumerate(ordered_sent_ids)}

# all_variants position -> (sent_id, variant)
sent_to_positions = defaultdict(list)
for i, v in enumerate(all_variants):
    sent_to_positions[v['sent_id']].append((i, v))

# Unique sentence IDs in the order they first appear in all_variants
seen = set()
processing_order = []
for v in all_variants:
    if v['sent_id'] not in seen:
        processing_order.append(v['sent_id'])
        seen.add(v['sent_id'])

# ============================================================================
# COMPUTE ADAPTIVE SURPRISAL
# ============================================================================
print(f"\nComputing adaptive LSTM surprisal for {len(processing_order)} sentences...")

results = [None] * len(all_variants)

for sent_id in tqdm(processing_order, desc="Adaptive Surprisal"):
    sent_obj = sent_lookup.get(sent_id)

    # Find preceding sentence in discourse
    disc_idx = sent_id_to_disc_idx.get(sent_id)
    context_tokens = None
    if disc_idx is not None and disc_idx > 0:
        prev_sent = sent_lookup.get(ordered_sent_ids[disc_idx - 1])
        if prev_sent:
            context_tokens = [w.form for w in prev_sent.words]

    # Adapt base model to context (or use base model if no context)
    if context_tokens:
        adapted = adapt_model(base_model, context_tokens, vocab, lr=2.0)
    else:
        adapted = base_model

    # Score all variants for this sentence
    for orig_idx, variant in sent_to_positions[sent_id]:
        if sent_obj is None:
            surp = 0.0
        else:
            word_map = {w.idx: w.form for w in sent_obj.words}
            tokens = [word_map.get(idx, '<UNK>') for idx in variant['variant_order']]
            surp = get_surprisal(adapted, tokens, vocab)

        results[orig_idx] = {
            'sent_id': sent_id,
            'adaptive_surprisal': surp,
            'is_reference': variant['is_reference']
        }

# Fill any gaps
for i, r in enumerate(results):
    if r is None:
        results[i] = {
            'sent_id': all_variants[i]['sent_id'],
            'adaptive_surprisal': 0.0,
            'is_reference': all_variants[i]['is_reference']
        }

# ============================================================================
# SAVE
# ============================================================================
os.makedirs("./data/results", exist_ok=True)
out_path = "./data/results/adaptive_lstm_scores.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

refs = sum(1 for r in results if r['is_reference'])
print(f"\nSaved {len(results)} adaptive LSTM scores to {out_path}")
print(f"  References: {refs}, Variants: {len(results)-refs}")
print("Next: Run scripts/merge_all_feature.py")
