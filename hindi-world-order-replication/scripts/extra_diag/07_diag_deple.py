#!/usr/bin/env python3
import sys
import pickle
sys.path.insert(0, 'src')
from features.basic_features import is_punctuation

# Load one sentence
with open('data/processed/filtered_reference_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# Pick first sentence
sent = sentences[0]

print(f"\nSentence: {sent.text}")
print(f"Total words: {len(sent)}\n")

print("Dependencies:")
print(f"{'Word':<15} {'→ Head':<15} {'Distance':<10} {'Intervening'}")
print("-" * 60)

total = 0

for word in sent.words:
    if word.head == 0 or is_punctuation(word):
        continue
    
    # Get head word
    head_word = sent.words[word.head - 1]
    
    # Calculate distance
    start = min(word.idx, word.head)
    end = max(word.idx, word.head)
    
    # Count intervening non-punctuation words
    intervening = 0
    for i in range(start + 1, end):
        w = sent.words[i - 1]
        if not is_punctuation(w):
            intervening += 1
    
    total += intervening
    
    print(f"{word.form:<15} → {head_word.form:<15} {abs(word.idx - word.head):<10} {intervening}")

print(f"\n{'='*60}")
print(f"Total dependency length: {total}")
print(f"Sentence length: {len(sent)}")
print(f"Ratio: {total / len(sent):.2f}")
print(f"{'='*60}")

if total / len(sent) > 1.5:
    print("\n⚠️  RATIO TOO HIGH (>1.5)")
    print("   This suggests:")
    print("   - Very complex sentence structure")
    print("   - OR: Long-distance dependencies")
    print("   - OR: Calculation might be summing differently than expected")
elif total / len(sent) < 0.5:
    print("\n✅ RATIO LOW (<0.5) - Simple structure")
else:
    print("\n✅ RATIO NORMAL (0.5-1.5) - Reasonable complexity")

# Let's also check a few more sentences
print(f"\n{'='*60}")
print("CHECKING 10 SENTENCES:")
print(f"{'='*60}\n")

for i in range(10):
    sent = sentences[i]
    
    total = 0
    for word in sent.words:
        if word.head == 0 or is_punctuation(word):
            continue
        
        start = min(word.idx, word.head)
        end = max(word.idx, word.head)
        
        intervening = 0
        for j in range(start + 1, end):
            w = sent.words[j - 1]
            if not is_punctuation(w):
                intervening += 1
        
        total += intervening
    
    ratio = total / len(sent) if len(sent) > 0 else 0
    print(f"{i+1}. Length: {len(sent):3d} | DepLen: {total:4d} | Ratio: {ratio:.2f} | {sent.text[:50]}...")

print(f"\n{'='*60}\n")