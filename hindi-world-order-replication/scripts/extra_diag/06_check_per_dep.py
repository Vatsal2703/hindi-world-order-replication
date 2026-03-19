#!/usr/bin/env python3
import sys
import pickle
sys.path.insert(0, 'src')
from parsers.ud_parser import Sentence
from features.basic_features import calculate_dependency_length_temperley, is_punctuation

# Load sentences
with open('data/processed/filtered_reference_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

total_deplen = 0
total_deps = 0

for sent in sentences[:100]:  # Sample 100 sentences
    deplen = calculate_dependency_length_temperley(sent)
    
    # Count non-punctuation dependencies
    num_deps = 0
    for word in sent.words:
        if word.head != 0 and not is_punctuation(word):
            num_deps += 1
    
    total_deplen += deplen
    total_deps += num_deps

print(f"Total sentences: 100")
print(f"Total dependency length: {total_deplen}")
print(f"Total dependencies: {total_deps}")
print(f"\nAverage dependency length PER SENTENCE: {total_deplen / 100:.2f}")
print(f"Average dependency length PER DEPENDENCY: {total_deplen / total_deps:.2f}")
print(f"\nPaper reports: ~12")
print(f"Your per-dependency: {total_deplen / total_deps:.2f}")