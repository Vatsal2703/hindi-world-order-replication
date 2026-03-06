#!/usr/bin/env python3
"""
Variant Generator for Hindi Word Order
Generates grammatical permutations of preverbal constituents

Author: [Your Name]
Date: [Date]
"""

import os
import pickle
from typing import List, Set, Tuple, Dict
from itertools import permutations
from collections import defaultdict, Counter


class VariantGenerator:
    """Generate word order variants by permuting preverbal constituents"""
    
    def __init__(self, sentences):
        """
        Initialize generator with parsed sentences
        
        Args:
            sentences: List of Sentence objects from parser
        """
        self.sentences = sentences
        self.attested_patterns = set()
        self.stats = {
            'total_references': 0,
            'total_variants': 0,
            'filtered_variants': 0,
            'avg_variants_per_ref': 0
        }
    
    def extract_dependency_patterns(self):
        """
        Extract all attested dependency relation sequences from corpus
        These patterns define grammatical orderings
        """
        print("Extracting dependency patterns from corpus...")
        
        for sent in self.sentences:
            if sent.root_idx is None:
                continue
            
            # Get preverbal constituents
            preverbal = sent.get_preverbal_constituents()
            
            if len(preverbal) < 2:
                continue
            
            # Extract dependency relation sequence
            deprel_sequence = tuple(w.deprel for w in preverbal)
            self.attested_patterns.add(deprel_sequence)
            
            # Also store reverse patterns (for flexibility)
            self.attested_patterns.add(tuple(reversed(deprel_sequence)))
        
        print(f"  → Extracted {len(self.attested_patterns)} unique dependency patterns")
        print(f"\nMost common patterns:")
        
        # Show top 10 patterns
        pattern_counts = Counter()
        for sent in self.sentences:
            preverbal = sent.get_preverbal_constituents()
            if len(preverbal) >= 2:
                pattern = tuple(w.deprel for w in preverbal)
                pattern_counts[pattern] += 1
        
        for pattern, count in pattern_counts.most_common(10):
            print(f"    {' → '.join(pattern)}: {count} times")
        
        print()
    
    def is_grammatical(self, deprel_sequence: Tuple[str]) -> bool:
        """
        Check if a dependency relation sequence is attested in corpus
        
        Args:
            deprel_sequence: Tuple of dependency relations
            
        Returns:
            True if pattern is attested, False otherwise
        """
        # Check exact match
        if deprel_sequence in self.attested_patterns:
            return True
        
        # For 2-constituent orders, also check reverse
        if len(deprel_sequence) == 2:
            if tuple(reversed(deprel_sequence)) in self.attested_patterns:
                return True
        
        return False
    
    def generate_variants_for_sentence(self, sentence, max_variants: int = 100) -> List[Dict]:
        """
        Generate all grammatical variants for a single sentence
        
        Args:
            sentence: Sentence object
            max_variants: Maximum variants to generate (prevent explosion)
            
        Returns:
            List of variant dictionaries
        """
        # Get preverbal constituents
        preverbal = sentence.get_preverbal_constituents()
        
        # Need at least 2 constituents to reorder
        if len(preverbal) < 2:
            return []
        
        # Don't generate variants for very long constituent lists (too many permutations)
        if len(preverbal) > 5:
            return []
        
        variants = []
        
        # Generate all permutations
        all_perms = list(permutations(preverbal))
        
        # Limit to max_variants
        if len(all_perms) > max_variants:
            all_perms = all_perms[:max_variants]
        
        for perm in all_perms:
            # Extract dependency pattern
            deprel_seq = tuple(w.deprel for w in perm)
            
            # Check grammaticality
            if self.is_grammatical(deprel_seq):
                # Create variant representation
                variant = {
                    'sent_id': sentence.sent_id,
                    'reference_text': sentence.text,
                    'root_idx': sentence.root_idx,
                    'root_form': sentence.root_word.form if sentence.root_word else '',
                    'original_order': [w.idx for w in preverbal],
                    'variant_order': [w.idx for w in perm],
                    'deprel_sequence': deprel_seq,
                    'preverbal_words': [w.form for w in perm],
                    'is_reference': (perm == tuple(preverbal))  # Is this the original order?
                }
                
                variants.append(variant)
        
        return variants
    
    def generate_all_variants(self, 
                            min_preverbal: int = 2,
                            max_preverbal: int = 5,
                            target_references: int = 2000) -> Tuple[List, List]:
        """
        Generate variants for all suitable sentences
        
        Args:
            min_preverbal: Minimum preverbal constituents required
            max_preverbal: Maximum preverbal constituents allowed
            target_references: Target number of reference sentences
            
        Returns:
            Tuple of (reference_sentences, all_variants)
        """
        print(f"\n{'='*60}")
        print("GENERATING VARIANTS")
        print(f"{'='*60}\n")
        
        # First, extract dependency patterns from corpus
        self.extract_dependency_patterns()
        
        # Filter sentences suitable for variant generation
        suitable_sentences = []
        
        print("Selecting suitable sentences...")
        for sent in self.sentences:
            preverbal = sent.get_preverbal_constituents()
            num_preverbal = len(preverbal)
            
            # Check criteria
            if (min_preverbal <= num_preverbal <= max_preverbal and
                sent.has_subject() and 
                sent.has_object() and
                sent.root_word is not None):
                
                suitable_sentences.append(sent)
        
        print(f"  → Found {len(suitable_sentences)} suitable sentences")
        
        # Select target number
        if len(suitable_sentences) > target_references:
            # Prefer sentences with more preverbal constituents (more interesting)
            suitable_sentences.sort(key=lambda s: len(s.get_preverbal_constituents()), reverse=True)
            selected_sentences = suitable_sentences[:target_references]
        else:
            selected_sentences = suitable_sentences
        
        print(f"  → Selected {len(selected_sentences)} reference sentences\n")
        
        # Generate variants
        print("Generating variants for each reference sentence...")
        
        all_variants = []
        reference_sentences = []
        
        from tqdm import tqdm
        
        for sent in tqdm(selected_sentences, desc="Processing"):
            variants = self.generate_variants_for_sentence(sent)
            
            if variants:
                # Store reference sentence
                reference_sentences.append(sent)
                
                # Store all variants (including reference)
                all_variants.extend(variants)
                
                self.stats['total_variants'] += len(variants) - 1  # Exclude reference itself
        
        self.stats['total_references'] = len(reference_sentences)
        
        if self.stats['total_references'] > 0:
            self.stats['avg_variants_per_ref'] = self.stats['total_variants'] / self.stats['total_references']
        
        print(f"\n{'='*60}")
        print("VARIANT GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"Reference sentences: {self.stats['total_references']:,}")
        print(f"Total variants (excluding references): {self.stats['total_variants']:,}")
        print(f"Total sentences (refs + variants): {len(all_variants):,}")
        print(f"Average variants per reference: {self.stats['avg_variants_per_ref']:.2f}")
        print(f"{'='*60}\n")
        
        return reference_sentences, all_variants
    
    def create_pairwise_dataset(self, all_variants: List[Dict]) -> List[Dict]:
        """
        Create pairwise comparisons using Joachims transformation
        
        Each reference is paired with each of its variants
        Creates balanced dataset: 50% Reference-Variant, 50% Variant-Reference
        
        Args:
            all_variants: List of all variants (including references)
            
        Returns:
            List of paired comparisons
        """
        print("Creating pairwise dataset (Joachims transformation)...")
        
        # Group variants by sentence ID
        variants_by_sent = defaultdict(list)
        for variant in all_variants:
            variants_by_sent[variant['sent_id']].append(variant)
        
        pairs = []
        
        for sent_id, sent_variants in variants_by_sent.items():
            # Find reference
            reference = None
            non_references = []
            
            for v in sent_variants:
                if v['is_reference']:
                    reference = v
                else:
                    non_references.append(v)
            
            if reference is None:
                continue
            
            # Create pairs: Reference vs each Variant
            for variant in non_references:
                # Pair 1: Reference-Variant (label=1, reference preferred)
                pair1 = {
                    'sent_id': sent_id,
                    'sentence_a': reference,
                    'sentence_b': variant,
                    'label': 1,  # Reference is preferred
                    'pair_type': 'ref-var'
                }
                pairs.append(pair1)
                
                # Pair 2: Variant-Reference (label=0, variant not preferred)
                pair2 = {
                    'sent_id': sent_id,
                    'sentence_a': variant,
                    'sentence_b': reference,
                    'label': 0,  # Variant is not preferred
                    'pair_type': 'var-ref'
                }
                pairs.append(pair2)
        
        print(f"  → Created {len(pairs):,} pairwise comparisons")
        print(f"  → Balance: {sum(1 for p in pairs if p['label']==1)} positive, {sum(1 for p in pairs if p['label']==0)} negative")
        print()
        
        return pairs


def show_variant_examples(all_variants: List[Dict], n: int = 3):
    """Display example variants"""
    print(f"{'='*60}")
    print(f"EXAMPLE VARIANTS (first {n} reference sentences)")
    print(f"{'='*60}\n")
    
    # Group by sentence
    variants_by_sent = defaultdict(list)
    for v in all_variants:
        variants_by_sent[v['sent_id']].append(v)
    
    shown = 0
    for sent_id, variants in variants_by_sent.items():
        if shown >= n:
            break
        
        # Find reference
        reference = None
        for v in variants:
            if v['is_reference']:
                reference = v
                break
        
        if reference is None:
            continue
        
        print(f"Reference Sentence ID: {sent_id}")
        print(f"Root verb: {reference['root_form']}")
        print(f"Number of variants: {len(variants) - 1}\n")
        
        # Show reference
        print(f"  [REFERENCE] Order: {reference['original_order']}")
        print(f"              Words: {' '.join(reference['preverbal_words'])} {reference['root_form']}")
        print(f"              Deprels: {' → '.join(reference['deprel_sequence'])}\n")
        
        # Show variants
        for i, v in enumerate(variants, 1):
            if v['is_reference']:
                continue
            print(f"  [VARIANT {i}] Order: {v['variant_order']}")
            print(f"              Words: {' '.join(v['preverbal_words'])} {v['root_form']}")
            print(f"              Deprels: {' → '.join(v['deprel_sequence'])}")
        
        print()
        shown += 1


if __name__ == "__main__":
    # Test the generator
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python variant_generator.py <path_to_valid_sentences.pkl>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    
    print("Loading sentences...")
    with open(pkl_path, 'rb') as f:
        sentences = pickle.load(f)
    
    print(f"Loaded {len(sentences):,} sentences\n")
    
    # Generate variants
    generator = VariantGenerator(sentences)
    references, all_variants = generator.generate_all_variants(
        min_preverbal=2,
        max_preverbal=4,
        target_references=2000
    )
    
    # Show examples
    show_variant_examples(all_variants, n=3)
    
    # Create pairwise dataset
    pairs = generator.create_pairwise_dataset(all_variants)
    
    print(f"\n Variant generation complete!")
    print(f"   References: {len(references):,}")
    print(f"   Variants: {len(all_variants):,}")
    print(f"   Pairs: {len(pairs):,}")