import pickle
from tqdm import tqdm

# 1. Load Vocab
with open("./data/models/emille_vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)

# 2. Load Variants
with open("./data/processed/all_variants_final.pkl", 'rb') as f:
    data = pickle.load(f)

unk_id = vocab.get("<UNK>")
total_words = 0
unk_words = 0

for item in tqdm(data, desc="Checking Vocabulary Coverage"):
    ids = item['variant_order']
    total_words += len(ids)
    unk_words += ids.count(unk_id)

coverage = ((total_words - unk_words) / total_words) * 100

print(f"\n--- 🧪 VOCABULARY DIAGNOSTIC ---")
print(f"Total Words in Test Set: {total_words}")
print(f"Unknown (<UNK>) Words:   {unk_words}")
print(f"Vocabulary Coverage:     {coverage:.2f}%")

if coverage < 50:
    print("\n🚨 CRITICAL: Coverage is too low. The model is 'blind' to most of the text.")
else:
    print("\n✅ Coverage is fine. The issue is likely in the Model Loading or Padding.")