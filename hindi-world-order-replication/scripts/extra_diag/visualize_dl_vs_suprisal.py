import os
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 1. First Pass: Build the Global N-Gram Model
silver_dir = "./data/processed/silver_emille"
all_tokens = []
for f in os.listdir(silver_dir):
    if f.endswith(".conllu"):
        with open(os.path.join(silver_dir, f), 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) > 1: all_tokens.append(parts[1].lower())

unigrams = Counter(all_tokens)
bigrams = Counter(zip(all_tokens, all_tokens[1:]))
trigrams = Counter(zip(all_tokens, all_tokens[1:], all_tokens[2:]))
vocab_size = len(unigrams)

# 2. Second Pass: Calculate Metrics per File
file_metrics = []
for f in os.listdir(silver_dir):
    if f.endswith(".conllu"):
        tokens = []
        total_dl, dl_count = 0, 0
        with open(os.path.join(silver_dir, f), 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) > 6:
                        tokens.append(parts[1].lower())
                        idx, head = int(parts[0]), int(parts[6])
                        if head != 0:
                            total_dl += abs(idx - head); dl_count += 1
        
        avg_dl = total_dl / dl_count if dl_count > 0 else 0
        
        # Surprisal
        total_s, s_count = 0, 0
        for i in range(2, len(tokens)):
            context = (tokens[i-2], tokens[i-1])
            target = tokens[i]
            prob = (trigrams[context + (target,)] + 1) / (bigrams[context] + vocab_size)
            total_s += -math.log2(prob); s_count += 1
        
        avg_s = total_s / s_count if s_count > 0 else 0
        if avg_dl > 0 and avg_s > 0: file_metrics.append((avg_dl, avg_s))

# 3. Plotting the results
dl_arr, s_arr = zip(*file_metrics)
plt.figure(figsize=(10, 6))
plt.scatter(dl_arr, s_arr, color='#457b9d', alpha=0.7, edgecolors='w', s=80, label='EMILLE Files')

# Regression Line
m, b = np.polyfit(dl_arr, s_arr, 1)
plt.plot(dl_arr, m*np.array(dl_arr) + b, color='#e63946', linewidth=2, label='Trend Line')

plt.title('Trade-off: Dependency Length vs. Trigram Surprisal', fontsize=15, fontweight='bold')
plt.xlabel('Average Dependency Length (DL)', fontsize=12); plt.ylabel('Average Surprisal (bits)', fontsize=12)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("./results/dl_vs_surprisal_emille.png", dpi=300)