import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from collections import Counter
from tqdm import tqdm

# 1. Device Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")


# 2. Dataset Class for Batching
MAX_SEQ_LEN = 100  # truncate very long sentences to avoid OOM on MPS

class HindiDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        for sent in corpus:
            ids = [vocab.get(w, vocab["<UNK>"]) for w in sent]
            ids = ids[:MAX_SEQ_LEN]  # truncate
            if len(ids) > 1:
                self.data.append(torch.tensor(ids))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    # Sort by length descending — minimises padding waste in each batch
    batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)
    return pad_sequence(batch, batch_first=True, padding_value=0)

# 3. Load Data
# Try the full written corpus first, fall back to silver EMILLE
CORPUS_PATH = "./data/processed/emille_written_tokenized.pkl"
FALLBACK_PATH = "./data/processed/emille_tokenized.pkl"

if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, 'rb') as f:
        emille_corpus = pickle.load(f)
    print(f"Loaded full EMILLE written corpus: {len(emille_corpus):,} sentences")
elif os.path.exists(FALLBACK_PATH):
    print(f"WARNING: {CORPUS_PATH} not found.")
    print(f"  Run scripts/15b_parse_emille_raw.py first to build the full corpus.")
    print(f"  Falling back to: {FALLBACK_PATH}")
    with open(FALLBACK_PATH, 'rb') as f:
        emille_corpus = pickle.load(f)
    print(f"  Loaded {len(emille_corpus):,} sentences (smaller subset — not paper-equivalent)")
else:
    print(f"ERROR: No EMILLE corpus found. Run scripts/15b_parse_emille_raw.py first.")
    exit(1)

print(f"Using {len(emille_corpus):,} sentences for training")

# Build vocab with min_freq=10 cutoff (matches old emille_vocab.pkl ~32k tokens)
# Reuse existing vocab if available to keep scoring scripts compatible
VOCAB_PATH = "./data/models/emille_vocab.pkl"
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Loaded existing vocab: {len(vocab):,} tokens")
else:
    freq = Counter(w for sent in emille_corpus for w in sent)
    vocab = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}
    for word, count in freq.items():
        if count >= 10:
            vocab[word] = len(vocab)
    print(f"Built vocab (min_freq=10): {len(vocab):,} tokens")
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)

dataset = HindiDataset(emille_corpus, vocab)
del emille_corpus  # free ~600MB of Python string objects before training
# batch_size=8 avoids MPS OOM (16 caused 14+ GiB allocation on 1M sentences)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 4. Model Definition — Paper architecture: 2 layers, 200 hidden, 200-dim embed
# (Section 3.1, Table 1: "200 dimensions for word embeddings and LSTM hidden layer")
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

model = VanillaLSTM(len(vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 5. Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        inputs  = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}") 

        # Clear MPS cache every 500 steps to prevent memory fragmentation
        if step % 500 == 0 and device.type == "mps":
            torch.mps.empty_cache()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f}")
    # Save checkpoint after each epoch
    torch.save(model.state_dict(), "./data/models/emille_base_lstm.pt")
    print(f"  Checkpoint saved.")

# Save
torch.save(model.state_dict(), "./data/models/emille_base_lstm.pt")
with open("./data/models/emille_vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)
print("Done!")