#!/usr/bin/env python3
import sys
import os
import pickle
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsers.ud_parser import Sentence, Word

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*70)
print(" DEEP VARIANT INSPECTION - PAPER COMPLIANCE CHECK")
print("="*70 + "\n")

print("Loading data...")

with open('data/processed/reference_sentences.pkl', 'rb') as f:
    refs = pickle.load(f)

with open('data/processed/all_variants.pkl', 'rb') as f:
    variants = pickle.load(f)

with open('data/processed/pairwise_dataset.pkl', 'rb') as f:
    pairs = pickle.load(f)

print(f" Loaded {len(refs):,} references")
print(f" Loaded {len(variants):,} variants")
print(f" Loaded {len(pairs):,} pairs\n")

# ============================================================================
# ANALYSIS 1: BASIC STATISTICS
# ============================================================================

print("="*70)
print("1. BASIC STATISTICS")
print("="*70 + "\n")

# Count references vs variants
ref_count = sum(1 for v in variants if v['is_reference'])
var_count = sum(1 for v in variants if not v['is_reference'])

print(f"References in variants list: {ref_count:,}")
print(f"Non-reference variants: {var_count:,}")
print(f"Total: {len(variants):,}")
print(f"\nAverage variants per reference: {var_count / ref_count:.2f}")
print(f"Paper's average: ~36 variants per reference")

if var_count / ref_count < 10:
    print(f"LOW: You have much fewer variants than paper")
elif var_count / ref_count > 50:
    print(f"HIGH: You have more variants than paper")
else:
    print(f"REASONABLE: Within expected range")

# ============================================================================
# ANALYSIS 2: PREVERBAL CONSTITUENT DISTRIBUTION
# ============================================================================

print(f"\n{'='*70}")
print("2. PREVERBAL CONSTITUENT DISTRIBUTION")
print("="*70 + "\n")

# Count by number of preverbal constituents
preverbal_counts = Counter()
variants_by_preverbal = defaultdict(list)

for v in variants:
    if v['is_reference']:
        # Count preverbal constituents
        num_preverbal = len(v['original_order'])
        preverbal_counts[num_preverbal] += 1
        variants_by_preverbal[num_preverbal].append(v)

print("Reference sentences by preverbal count:")
for count in sorted(preverbal_counts.keys()):
    pct = 100 * preverbal_counts[count] / ref_count
    print(f"  {count} constituents: {preverbal_counts[count]:,} ({pct:.1f}%)")

# Calculate expected variants per preverbal count
import math
print("\nExpected permutations (factorial):")
for count in sorted(preverbal_counts.keys()):
    factorial = math.factorial(count)
    print(f"  {count} constituents: {factorial} permutations (before filtering)")

# ============================================================================
# ANALYSIS 3: GRAMMATICALITY FILTERING EFFECTIVENESS
# ============================================================================

print(f"\n{'='*70}")
print("3. GRAMMATICALITY FILTERING ANALYSIS")
print("="*70 + "\n")

# Group variants by sentence
variants_by_sent = defaultdict(list)
for v in variants:
    variants_by_sent[v['sent_id']].append(v)

# Calculate filtering rate
filtering_stats = []

for sent_id, sent_variants in variants_by_sent.items():
    # Find reference
    ref_variant = None
    for v in sent_variants:
        if v['is_reference']:
            ref_variant = v
            break
    
    if ref_variant:
        num_preverbal = len(ref_variant['original_order'])
        total_possible = math.factorial(num_preverbal)
        actual_generated = len(sent_variants)
        
        filtering_stats.append({
            'sent_id': sent_id,
            'num_preverbal': num_preverbal,
            'possible': total_possible,
            'generated': actual_generated,
            'filter_rate': (total_possible - actual_generated) / total_possible if total_possible > 0 else 0
        })

# Summary by preverbal count
print("Filtering effectiveness by constituent count:\n")
print(f"{'Constituents':<15} {'Possible':<10} {'Generated':<12} {'Filter Rate':<12}")
print("-" * 50)

for count in sorted(preverbal_counts.keys()):
    stats_for_count = [s for s in filtering_stats if s['num_preverbal'] == count]
    if stats_for_count:
        avg_possible = sum(s['possible'] for s in stats_for_count) / len(stats_for_count)
        avg_generated = sum(s['generated'] for s in stats_for_count) / len(stats_for_count)
        avg_filter = sum(s['filter_rate'] for s in stats_for_count) / len(stats_for_count)
        
        print(f"{count:<15} {avg_possible:<10.0f} {avg_generated:<12.1f} {avg_filter*100:<11.1f}%")

# ============================================================================
# ANALYSIS 4: DEPENDENCY RELATION PATTERNS
# ============================================================================

print(f"\n{'='*70}")
print("4. DEPENDENCY RELATION PATTERNS")
print("="*70 + "\n")

# Extract all unique patterns
all_patterns = Counter()
reference_patterns = Counter()

for v in variants:
    pattern = v['deprel_sequence']
    all_patterns[pattern] += 1
    if v['is_reference']:
        reference_patterns[pattern] += 1

print(f"Unique dependency patterns: {len(all_patterns)}")
print(f"Patterns in references: {len(reference_patterns)}")
print(f"\nTop 10 most common patterns:\n")

for pattern, count in all_patterns.most_common(10):
    pattern_str = ' → '.join(pattern)
    in_refs = reference_patterns.get(pattern, 0)
    print(f"  {pattern_str:<40} {count:>5} variants ({in_refs} references)")

# ============================================================================
# ANALYSIS 5: DETAILED EXAMPLES
# ============================================================================

print(f"\n{'='*70}")
print("5. DETAILED EXAMPLES (First 3 Sentences)")
print("="*70 + "\n")

shown = 0
for sent_id in sorted(variants_by_sent.keys())[:3]:
    sent_variants = variants_by_sent[sent_id]
    
    # Find reference
    reference = None
    for v in sent_variants:
        if v['is_reference']:
            reference = v
            break
    
    if not reference:
        continue
    
    shown += 1
    print(f"{shown}. Sentence ID: {sent_id}")
    print(f"   Text: {reference['reference_text'][:80]}...")
    print(f"   Root verb: {reference['root_form']}")
    print(f"   Preverbal constituents: {len(reference['original_order'])}")
    print(f"   Total permutations possible: {math.factorial(len(reference['original_order']))}")
    print(f"   Actual variants generated: {len(sent_variants)}")
    print(f"   Filter rate: {(1 - len(sent_variants)/math.factorial(len(reference['original_order'])))*100:.1f}%")
    print(f"\n   REFERENCE:")
    print(f"     Order: {reference['original_order']}")
    print(f"     Words: {' '.join(reference['preverbal_words'])} {reference['root_form']}")
    print(f"     Deprels: {' → '.join(reference['deprel_sequence'])}")
    
    print(f"\n   VARIANTS:")
    variant_num = 0
    for v in sent_variants:
        if not v['is_reference']:
            variant_num += 1
            print(f"     [{variant_num}] {' '.join(v['preverbal_words'])} {v['root_form']}")
            print(f"         Deprels: {' → '.join(v['deprel_sequence'])}")
            if variant_num >= 5:  # Show max 5 variants
                if len(sent_variants) - 1 > 5:
                    print(f"         ... and {len(sent_variants) - 6} more variants")
                break
    print()

# ============================================================================
# ANALYSIS 6: PAPER COMPARISON
# ============================================================================

print("="*70)
print("6. PAPER COMPARISON")
print("="*70 + "\n")

paper_stats = {
    'references': 1996,
    'variants': 72833,
    'avg_variants': 36.5,
    'pairs': 145666
}

your_stats = {
    'references': ref_count,
    'variants': var_count,
    'avg_variants': var_count / ref_count if ref_count > 0 else 0,
    'pairs': len(pairs)
}

print(f"{'Metric':<25} {'Paper':<15} {'Your Data':<15} {'Ratio':<10}")
print("-" * 70)

for key in ['references', 'variants', 'avg_variants', 'pairs']:
    paper_val = paper_stats[key]
    your_val = your_stats[key]
    ratio = your_val / paper_val if paper_val > 0 else 0
    
    status = "Meet the criteria" if 0.5 <= ratio <= 1.5 else "Does not meet the criteria"
    
    print(f"{key:<25} {paper_val:<15,.1f} {your_val:<15,.1f} {ratio:<9.2f} {status}")

# ============================================================================
# ANALYSIS 7: QUALITY CHECKS
# ============================================================================

print(f"\n{'='*70}")
print("7. QUALITY CHECKS")
print("="*70 + "\n")

# Check 1: All references should have variants
refs_without_variants = []
for ref in refs:
    ref_variants = variants_by_sent.get(ref.sent_id, [])
    if len(ref_variants) <= 1:  # Only reference, no variants
        refs_without_variants.append(ref.sent_id)

if refs_without_variants:
    print(f"{len(refs_without_variants)} references have NO variants")
    print(f"   Examples: {refs_without_variants[:5]}")
else:
    print(f"All references have at least one variant")

# Check 2: Pairs should be balanced
positive_pairs = sum(1 for p in pairs if p['label'] == 1)
negative_pairs = sum(1 for p in pairs if p['label'] == 0)

print(f"\nPairwise dataset balance:")
print(f"   Positive (ref preferred): {positive_pairs:,}")
print(f"   Negative (var preferred): {negative_pairs:,}")
print(f"   Balance ratio: {positive_pairs / negative_pairs:.3f}")

if abs(positive_pairs - negative_pairs) > 10:
    print(f"Imbalanced!")
else:
    print(f"Perfectly balanced")

# Check 3: Variant order consistency
print(f"\nChecking variant order consistency...")
errors = 0
for v in variants[:100]:  # Check first 100
    if len(v['variant_order']) != len(v['deprel_sequence']):
        errors += 1

if errors > 0:
    print(f"Found {errors} inconsistencies")
else:
    print(f"All variants have consistent structure")

print(f"\n{'='*70}")
print("INSPECTION COMPLETE")
print("="*70 + "\n")