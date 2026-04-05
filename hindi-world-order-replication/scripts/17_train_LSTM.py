import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from collections import Counter
from tqdm import tqdm

# 1. Device Setup for MacBook Air
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Dataset Class for Batching
class HindiDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        for sent in corpus:
            ids = [vocab.get(w, vocab["<UNK>"]) for w in sent]
            if len(ids) > 1:
                self.data.append(torch.tensor(ids))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    # Pads sentences in the batch to the same length
    return pad_sequence(batch, batch_first=True, padding_value=0)

# 3. Load Data
with open("./data/processed/emille_written_tokenized.pkl", 'rb') as f:
    emille_corpus = pickle.load(f)

# Paper: trained on EMILLE 1M written sentences (Section 3.1)
# Capped at 500k due to 8GB RAM constraint on Intel Mac (1M causes OOM)
import random
random.seed(42)
MAX_SENTS = 500_000
if len(emille_corpus) > MAX_SENTS:
    emille_corpus = random.sample(emille_corpus, MAX_SENTS)
    print(f"Sampled {MAX_SENTS:,} sentences (8GB RAM limit; paper used 1M)")
else:
    print(f"Using all {len(emille_corpus):,} sentences")

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
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

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
    
    for batch in pbar:
        batch = batch.to(device)
        # Shift for next-word prediction
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

# Save
torch.save(model.state_dict(), "./data/models/emille_base_lstm.pt")
with open("./data/models/emille_vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)
print("Done!")