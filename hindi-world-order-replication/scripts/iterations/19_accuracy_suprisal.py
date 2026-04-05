import pickle
import pandas as pd

# 1. Load your results
with open("./data/results/final_scored_variants.pkl", 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

# 2. Group by sent_id to compare variants of the same sentence
correct_predictions = 0
total_sentences = 0

for sent_id, group in df.groupby('sent_id'):
    # Get the reference surprisal
    ref_row = group[group['is_reference'] == True]
    if ref_row.empty: continue
    
    ref_surprisal = ref_row['avg_surprisal'].values[0]
    
    # Get the minimum surprisal among the variants
    # (Excluding the reference itself)
    variants = group[group['is_reference'] == False]
    if variants.empty: continue
    
    min_variant_surprisal = variants['avg_surprisal'].min()
    
    # Prediction is 'Correct' if the reference has the LOWEST surprisal
    if ref_surprisal < min_variant_surprisal:
        correct_predictions += 1
    
    total_sentences += 1

# 3. Final Accuracy
accuracy = (correct_predictions / total_sentences) * 100
print(f"Comparison Complete!")
print(f"Total Unique Sentences: {total_sentences}")
print(f"Model Accuracy (Reference < Variants): {accuracy:.2f}%")