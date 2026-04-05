import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
print("Loading results...")
with open("./data/results/final_scored_variants.pkl", 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

# Rename for better legend labels
df['Sentence Type'] = df['is_reference'].map({True: 'Reference (SOV)', False: 'Scrambled Variant'})

# 2. Create the Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Use a Kernel Density Estimate (KDE) plot
sns.kdeplot(data=df, x='avg_surprisal', hue='Sentence Type', fill=True, common_norm=False, palette='viridis', alpha=0.5, linewidth=2)

# 3. Formatting
plt.title('Distribution of Average Surprisal: Hindi Reference vs. Scrambled Variants', fontsize=14, pad=15)
plt.xlabel('Average Surprisal (bits)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()

# 4. Save and Show
plt.savefig('./data/results/surprisal_distribution.png', dpi=300)
print("Plot saved to: ./data/results/surprisal_distribution.png")
plt.show()