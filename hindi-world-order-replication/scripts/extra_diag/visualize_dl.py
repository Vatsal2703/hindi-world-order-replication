import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_file_dl(file_path):
    total_dl = 0
    word_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) > 6:
                    idx, head = int(parts[0]), int(parts[6])
                    if head != 0:
                        total_dl += abs(idx - head)
                        word_count += 1
    return total_dl / word_count if word_count > 0 else 0

# 1. Collect DL for all 72 files
silver_dir = "./data/processed/silver_emille"
all_dls = []
for f in sorted(os.listdir(silver_dir)):
    if f.endswith(".conllu"):
        dl = calculate_file_dl(os.path.join(silver_dir, f))
        if dl > 0: all_dls.append(dl)

# 2. Plotting
plt.figure(figsize=(10, 6))
plt.hist(all_dls, bins=15, color='#a8dadc', edgecolor='#457b9d', alpha=0.8, label='EMILLE Spoken Files')

# Add reference lines
mean_dl = np.mean(all_dls)
plt.axvline(mean_dl, color='#1d3557', linestyle='dashed', linewidth=2, label=f'Mean Observed: {mean_dl:.3f}')
plt.axvline(9.443, color='#e63946', linestyle='dashed', linewidth=2, label='Random Baseline: 9.443')

# Aesthetics for the Professor
plt.title('Consistency of Dependency Length Minimization (DLM) in Hindi', fontsize=16, fontweight='bold')
plt.xlabel('Average Dependency Length (DL)', fontsize=13)
plt.ylabel('Number of Files', fontsize=13)
plt.xlim(2, 11) # Zooming in to show the massive gap to the random baseline
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Save for the meeting
output_img = "./results/dl_distribution_emille.png"
os.makedirs("./results", exist_ok=True)
plt.savefig(output_img, dpi=300)
print(f"✅ Success! Plot saved to {output_img}")