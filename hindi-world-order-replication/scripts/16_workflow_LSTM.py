import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import copy
import pandas as pd
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_surprisal(model, tokens, vocab):
    model.eval()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Extract log-probs of actual words
        target_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze()
        return -torch.sum(target_log_probs).item() / 0.693147 # Convert to bits (log2)

def adapt_model(model, prime_tokens, vocab, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    ids = [vocab.get(t, vocab["<UNK>"]) for t in prime_tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).unsqueeze(0).to(device)
    
    optimizer.zero_grad()
    logits, _ = model(input_ids)
    loss = criterion(logits.transpose(1, 2), target_ids)
    loss.backward()
    optimizer.step()
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================
# 1. Load Everything
with open("./data/models/emille_vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
with open("./data/processed/reference_sentences.pkl", 'rb') as f:
    hdbt_refs = pickle.load(f) # Original sentences in order
with open("./data/processed/all_variants_final.pkl", 'rb') as f:
    variants = pickle.load(f)

# Load Vanilla LSTM Base
from scripts.train_lstm import VanillaLSTM # Import your class definition
model = VanillaLSTM(len(vocab)).to(device)
model.load_state_dict(torch.load("./data/models/emille_base_lstm.pt"))
base_weights = copy.deepcopy(model.state_dict())

results = []

# 2. Sequential Adaptation Loop
for i in tqdm(range(1, len(hdbt_refs)), desc="Adaptive Testing"):
    # Prime = Sentence N-1, Target = Sentence N
    prime_sent = hdbt_refs[i-1]
    target_sent = hdbt_refs[i]
    
    # Get variants for the target sentence
    current_vars = [v for v in variants if v['sent_id'] == target_sent.sent_id]
    if not current_vars: continue
    
    # --- METHOD B: ADAPTIVE ---
    model.load_state_dict(base_weights) # Reset to Base
    model = adapt_model(model, [w.form for w in prime_sent.words], vocab)
    
    ref_tokens = [w.form for w in target_sent.words]
    s_ref_adapt = get_surprisal(model, ref_tokens, vocab)
    
    for var in current_vars:
        if var['is_reference']: continue
        var_tokens = [target_sent.words[idx].form for idx in var['variant_order']]
        s_var_adapt = get_surprisal(model, var_tokens, vocab)
        
        results.append({
            'sent_id': target_sent.sent_id,
            'label': 1,
            'surprisal_diff_adaptive': s_ref_adapt - s_var_adapt
        })

# 3. Save for Final Choice Model
df = pd.DataFrame(results)
df.to_pickle("./data/features/pairwise_adaptive_surprisal.pkl")
print(f"✅ Saved Adaptive Surprisal features for {len(df)} pairs.")