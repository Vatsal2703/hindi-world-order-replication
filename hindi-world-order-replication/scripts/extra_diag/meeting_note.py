import os
import numpy as np

def get_stats(silver_dir):
    all_dls = []
    for f in os.listdir(silver_dir):
        if f.endswith(".conllu"):
            # Reuse your DL calculation logic here
            # (Simplified for this snippet)
            pass 

    # Since you already have the 3.66 and 9.44, 
    # Let's prepare the "Talking Points":
    print("📋 MEETING TALKING POINTS:")
    print(f"1. OBSERVED DL: 3.66 (Stanza HDTB Parser)")
    print(f"2. RANDOM DL: 9.44 (Shuffled Baseline)")
    print(f"3. EFFICIENCY GAIN: {(9.44-3.66)/9.44*100:.1f}% reduction in memory load.")
    print(f"4. TRIGRAM SURPRISAL: 12.87 bits (High context sensitivity).")
    print(f"5. OUTLIER ANALYSIS: Found 'Spoken' sentences reaching DL 6.23 due to fillers like 'मतलब'.")

get_stats("./data/processed/silver_emille")