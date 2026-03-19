import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys
from tqdm import tqdm

# Ensure the script can find your local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MacBook M-series Acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# 1. LOAD TOKENIZED EMILLE DATA
# ============================================================================
INPUT_FILE = "./data/processed/emille_tokenized.pkl"
MODEL_OUT = "./data/models/emille_base_lstm.pt"
VOCAB_OUT = "./data/models/emille_vocab.pkl"

with open(INPUT_FILE, 'rb') as f:
    emille_corpus = pickle.load(f)

# Build Vocab
vocab = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}
for sent in emille_corpus:
    for word in sent:
        if word not in vocab:
            vocab[word] = len(vocab)

# ============================================================================
# 2. DEFINE THE VANILLA LSTM (As per Professor's guidance)
# ============================================================================
class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(VanillaLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# Training Setup
model = VanillaLSTM(len(vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================
print(f"\nTraining Base LSTM on EMILLE using {device}...")
os.makedirs("./data/models", exist_ok=True)

# Training for 5 epochs to ensure it learns the syntax well
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for sent in tqdm(emille_corpus, desc="Training"):
        ids = [vocab.get(w, vocab["<UNK>"]) for w in sent]
        if len(ids) < 2: continue
        
        # Shifted for next-word prediction
        inputs = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
        targets = torch.tensor(ids).unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(emille_corpus):.4f}")

# Save the learned weights and the vocabulary
torch.save(model.state_dict(), MODEL_OUT)
with open(VOCAB_OUT, 'wb') as f:
    pickle.dump(vocab, f)

print(f"\nTraining Complete. Model and Vocab saved to ./data/models/")