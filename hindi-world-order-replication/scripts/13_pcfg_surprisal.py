#!/usr/bin/env python3
"""
Step 13: PCFG Surprisal Pipeline

Paper: "Discourse Context Predictability Effects in Hindi Word Order"
       Ranjan et al. (EMNLP 2022), Section 3.1 item 3

Pipeline:
  1. Convert HUTB dependency trees → constituency (phrase structure) trees
     using ChunkId/ChunkType annotations (Yadav et al., 2017)
  2. 5-fold CV: train Berkeley PCFG parser on 4 folds, parse 5th fold
  3. Score all variants with sentence log-likelihood

Requirements:
  - Java 8+ installed
  - Berkeley Parser jar at tools/berkeley-parser/berkeleyParser.jar
  - HDTB .conllu files with ChunkId/ChunkType in MISC column

Usage:
  python scripts/13_pcfg_surprisal.py --step convert   # Step 1: dep→constituency
  python scripts/13_pcfg_surprisal.py --step train      # Step 2: 5-fold CV training
  python scripts/13_pcfg_surprisal.py --step score      # Step 3: score variants
  python scripts/13_pcfg_surprisal.py --step all        # Run all steps
"""

import sys
import os
import re
import pickle
import subprocess
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================================
# CONFIGURATION
# ============================================================================

CONLLU_FILE = os.path.expanduser(
    "~/Downloads/Mtech Thesis/UD_Hindi-HDTB-master/hi_hdtb-ud-train.conllu"
)
BERKELEY_JAR = "./tools/berkeley-parser/berkeleyParser.jar"

# Output directories
CONSTITUENCY_DIR = "./data/processed/constituency"
FOLDS_DIR = "./data/processed/pcfg_folds"
GRAMMAR_DIR = "./data/processed/pcfg_grammars"
SCORES_DIR = "./data/processed/pcfg_scores"

VARIANTS_FILE = "./data/processed/all_variants_final.pkl"
OUTPUT_FILE = "./data/processed/pcfg_scores.pkl"

N_FOLDS = 5


# ============================================================================
# PART 1: DEPENDENCY → CONSTITUENCY CONVERSION
# ============================================================================

def parse_conllu_with_chunks(filepath):
    """
    Parse CoNLL-U file and extract chunk information from MISC field.
    
    Each token gets:
      - Standard CoNLL-U fields (idx, form, upos, head, deprel)
      - chunk_id: e.g., "NP", "NP2", "VGF", "BLK"
      - chunk_type: "head" or "child"
    
    Returns list of sentences, each a list of token dicts.
    """
    sentences = []
    current = []
    current_sent_id = ""

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current:
                    sentences.append({
                        'sent_id': current_sent_id,
                        'tokens': current
                    })
                    current = []
                    current_sent_id = ""
                continue

            if line.startswith('#'):
                if line.startswith('# sent_id'):
                    current_sent_id = line.split('=', 1)[1].strip()
                continue

            parts = line.split('\t')
            if not parts[0].isdigit():
                continue

            # Parse MISC field for ChunkId and ChunkType
            misc = parts[9] if len(parts) > 9 else '_'
            chunk_id = None
            chunk_type = None
            for kv in misc.split('|'):
                if kv.startswith('ChunkId='):
                    chunk_id = kv.split('=', 1)[1]
                elif kv.startswith('ChunkType='):
                    chunk_type = kv.split('=', 1)[1]

            current.append({
                'idx': int(parts[0]),
                'form': parts[1],
                'lemma': parts[2],
                'upos': parts[3],
                'xpos': parts[4],
                'head': int(parts[6]) if parts[6].isdigit() else 0,
                'deprel': parts[7],
                'chunk_id': chunk_id,
                'chunk_type': chunk_type,
            })

    if current:
        sentences.append({
            'sent_id': current_sent_id,
            'tokens': current
        })

    return sentences


def dep_to_constituency(sentence_data):
    """
    Convert a dependency tree with chunk annotations to a 
    constituency (phrase structure) tree in PTB bracket format.
    
    Algorithm (simplified Yadav et al. 2017):
      1. Group tokens by ChunkId → each chunk becomes a phrase node
      2. The chunk's head token determines the phrase label
      3. Map chunk labels to phrase categories:
         NP* → NP, VGF/VGNF/VGINF → VP, JJP → ADJP, RBP → ADVP, etc.
      4. Build tree bottom-up: chunks → clause → sentence
    
    Returns PTB bracket string or None if conversion fails.
    """
    tokens = sentence_data['tokens']

    if not tokens:
        return None

    # ── Step 1: Group tokens into chunks ──
    chunks = defaultdict(list)
    chunk_order = []  # preserve left-to-right order

    for tok in tokens:
        cid = tok['chunk_id']
        if cid is None:
            cid = f"_UNK_{tok['idx']}"
        chunks[cid].append(tok)
        if cid not in chunk_order:
            chunk_order.append(cid)

    # ── Step 2: Build phrase nodes for each chunk ──
    phrase_nodes = []

    for cid in chunk_order:
        chunk_tokens = chunks[cid]
        # Sort by position
        chunk_tokens.sort(key=lambda t: t['idx'])

        # Determine phrase label from chunk ID
        phrase_label = chunk_id_to_phrase(cid)

        # Build the phrase: (NP (NOUN word1) (ADP word2))
        children = []
        for tok in chunk_tokens:
            pos = tok['upos']
            form = tok['form']
            # Escape brackets in form
            form = form.replace('(', '-LRB-').replace(')', '-RRB-')
            children.append(f"({pos} {form})")

        phrase_str = f"({phrase_label} {' '.join(children)})"
        phrase_nodes.append(phrase_str)

    # ── Step 3: Wrap in sentence node ──
    tree = f"(S {' '.join(phrase_nodes)})"

    return tree


def chunk_id_to_phrase(chunk_id):
    """
    Map HDTB chunk IDs to standard phrase labels.
    
    HDTB uses: NP, NP2, NP3, ..., VGF, VGNF, VGINF, JJP, RBP, BLK, CCP, FRAGP
    Strip trailing digits to get base chunk type.
    """
    # Remove trailing digits: NP2 → NP, NP3 → NP
    base = re.sub(r'\d+$', '', chunk_id)

    mapping = {
        'NP': 'NP',
        'VGF': 'VP',      # Verb Group Finite
        'VGNF': 'VP',     # Verb Group Non-Finite
        'VGINF': 'VP',    # Verb Group Infinitive
        'JJP': 'ADJP',    # Adjective Phrase
        'RBP': 'ADVP',    # Adverb Phrase
        'CCP': 'CCP',     # Conjunction Phrase
        'BLK': 'BLK',     # Punctuation block
        'FRAGP': 'FRAGP', # Fragment
        'NEGP': 'NEGP',   # Negation
        '_UNK': 'FRAG',   # Unknown chunk
    }

    # Handle _UNK_N patterns
    if base.startswith('_UNK'):
        return 'FRAG'

    return mapping.get(base, base)


def convert_all_trees(conllu_file, output_dir):
    """
    Convert all sentences in a CoNLL-U file to constituency trees.
    Saves individual tree files and a combined file.
    """
    print("Parsing CoNLL-U with chunk annotations...")
    sentences = parse_conllu_with_chunks(conllu_file)
    print(f"  Parsed {len(sentences):,} sentences")

    os.makedirs(output_dir, exist_ok=True)

    trees = []
    failed = 0

    for sent_data in tqdm(sentences, desc="Converting to constituency"):
        tree = dep_to_constituency(sent_data)
        if tree is None:
            failed += 1
            continue
        trees.append({
            'sent_id': sent_data['sent_id'],
            'tree': tree,
        })

    # Save combined file (one tree per line — Berkeley Parser format)
    combined_file = os.path.join(output_dir, "all_trees.txt")
    with open(combined_file, 'w', encoding='utf-8') as f:
        for t in trees:
            f.write(t['tree'] + '\n')

    # Save mapping for later reference
    mapping_file = os.path.join(output_dir, "tree_mapping.pkl")
    with open(mapping_file, 'wb') as f:
        pickle.dump(trees, f)

    print(f"\n  Successfully converted: {len(trees):,}")
    print(f"  Failed: {failed}")
    print(f"  Paper target: ~12,000 constituency trees")
    print(f"  Saved to: {combined_file}")

    return trees


# ============================================================================
# PART 2: 5-FOLD CV TRAINING
# ============================================================================

def create_folds(trees, n_folds, output_dir):
    """
    Split constituency trees into n_folds for cross-validation.
    Each fold directory gets a train.txt and test.txt file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle with fixed seed for reproducibility
    indices = list(range(len(trees)))
    random.seed(42)
    random.shuffle(indices)

    fold_size = len(indices) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            end = len(indices)  # last fold gets remainder
        else:
            end = start + fold_size
        folds.append(indices[start:end])

    print(f"\nCreating {n_folds} folds...")

    for fold_idx in range(n_folds):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        test_indices = set(folds[fold_idx])
        train_indices = [i for i in indices if i not in test_indices]

        # Write train file
        train_file = os.path.join(fold_dir, "train.txt")
        with open(train_file, 'w', encoding='utf-8') as f:
            for i in train_indices:
                f.write(trees[i]['tree'] + '\n')

        # Write test file
        test_file = os.path.join(fold_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            for i in folds[fold_idx]:
                f.write(trees[i]['tree'] + '\n')

        # Save test sent_ids for later matching
        test_ids_file = os.path.join(fold_dir, "test_sent_ids.pkl")
        with open(test_ids_file, 'wb') as f:
            pickle.dump([trees[i]['sent_id'] for i in folds[fold_idx]], f)

        print(f"  Fold {fold_idx}: train={len(train_indices):,}, test={len(folds[fold_idx]):,}")

    return folds


def train_berkeley_grammar(fold_dir, grammar_dir, berkeley_jar):
    """
    Train a Berkeley Parser grammar on one fold's training data.
    
    Uses GrammarTrainer to learn a latent-variable PCFG.
    """
    train_file = os.path.join(fold_dir, "train.txt")
    fold_name = os.path.basename(fold_dir)
    grammar_file = os.path.join(grammar_dir, f"{fold_name}.gr")

    os.makedirs(grammar_dir, exist_ok=True)

    cmd = [
        'java', '-Xmx4g',
        '-cp', berkeley_jar,
        'edu.berkeley.nlp.PCFGLA.GrammarTrainer',
        '-path', train_file,
        '-out', grammar_file,
        '-treebank', 'SINGLEFILE',
        '-SMcycles', '5',     # 5 split-merge cycles (default for Berkeley)
    ]

    print(f"\n  Training grammar for {fold_name}...")
    print(f"    Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:500]}")
            return None
        print(f"    Saved grammar → {grammar_file}")
        return grammar_file
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Training timed out (1 hour limit)")
        return None
    except FileNotFoundError:
        print(f"    ERROR: Java not found")
        return None


def train_all_folds(folds_dir, grammar_dir, berkeley_jar, n_folds):
    """Train Berkeley grammars for all folds."""
    print("\n" + "=" * 70)
    print(" TRAINING BERKELEY PARSER (5-FOLD CV)")
    print("=" * 70)

    grammars = {}
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(folds_dir, f"fold_{fold_idx}")
        grammar_file = train_berkeley_grammar(fold_dir, grammar_dir, berkeley_jar)
        if grammar_file:
            grammars[fold_idx] = grammar_file

    print(f"\n  Successfully trained: {len(grammars)}/{n_folds} grammars")
    return grammars


# ============================================================================
# PART 3: SCORING
# ============================================================================

def get_sentence_loglikelihood(sentence_text, grammar_file, berkeley_jar):
    """
    Get the sentence log-likelihood P(w) from Berkeley Parser.
    
    Uses -sentence_likelihood flag which outputs log P(w) by
    summing over all possible parse trees.
    """
    cmd = [
        'java', '-Xmx2g',
        '-jar', berkeley_jar,
        '-gr', grammar_file,
        '-sentence_likelihood',
        '-maxLength', '200',
    ]

    try:
        result = subprocess.run(
            cmd,
            input=sentence_text + '\n',
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None

        # Parse the log-likelihood from output
        output = result.stdout.strip()
        if output:
            try:
                return float(output)
            except ValueError:
                return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    return None


def score_variants_batch(sentences, grammar_file, berkeley_jar):
    """
    Score a batch of sentences using Berkeley Parser.
    
    More efficient than one-at-a-time: sends all sentences
    via stdin and reads all outputs.
    """
    input_text = '\n'.join(sentences) + '\n'

    cmd = [
        'java', '-Xmx2g',
        '-jar', berkeley_jar,
        '-gr', grammar_file,
        '-sentence_likelihood',
        '-maxLength', '200',
    ]

    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"  Parser error: {result.stderr[:200]}")
            return [None] * len(sentences)

        scores = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                scores.append(None)
                continue
            # Berkeley outputs: "score\ttree" when -sentence_likelihood is used
            # e.g.: "-50.190\t( ( (NP (PRON इसे)) ... ) )"
            # Split by tab and take the first element as the score
            parts = line.split('\t', 1)
            try:
                scores.append(float(parts[0]))
            except ValueError:
                scores.append(None)

        # Pad if we got fewer outputs than inputs
        while len(scores) < len(sentences):
            scores.append(None)

        return scores

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return [None] * len(sentences)


def determine_fold_for_sentence(sent_id, folds_dir, n_folds):
    """
    Determine which fold a sentence belongs to (for selecting
    the correct held-out grammar).
    """
    for fold_idx in range(n_folds):
        ids_file = os.path.join(folds_dir, f"fold_{fold_idx}", "test_sent_ids.pkl")
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                test_ids = pickle.load(f)
            if sent_id in test_ids:
                return fold_idx
    return None


def score_all_variants(variants_file, folds_dir, grammar_dir, berkeley_jar, n_folds):
    """
    Score all variants using the appropriate held-out grammar.
    
    For each variant, find which fold its reference sentence
    belongs to, then use that fold's grammar (trained on the
    other 4 folds) to get the sentence log-likelihood.
    
    PCFG surprisal = -log P(sentence)
    """
    print("\n" + "=" * 70)
    print(" SCORING VARIANTS WITH PCFG")
    print("=" * 70)

    with open(variants_file, 'rb') as f:
        variants = pickle.load(f)

    print(f"  Loaded {len(variants):,} entries to score")

    # Build sent_id → fold mapping
    print("  Building fold mapping...")
    sent_id_to_fold = {}
    for fold_idx in range(n_folds):
        ids_file = os.path.join(folds_dir, f"fold_{fold_idx}", "test_sent_ids.pkl")
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                test_ids = pickle.load(f)
            for sid in test_ids:
                sent_id_to_fold[sid] = fold_idx

    # Load grammar paths
    grammars = {}
    for fold_idx in range(n_folds):
        gr_path = os.path.join(grammar_dir, f"fold_{fold_idx}.gr")
        if os.path.exists(gr_path):
            grammars[fold_idx] = gr_path

    if not grammars:
        print("  ERROR: No trained grammars found!")
        return

    # Group variants by fold for batch processing
    fold_groups = defaultdict(list)
    for i, entry in enumerate(variants):
        sid = entry['sent_id']
        fold_idx = sent_id_to_fold.get(sid)
        if fold_idx is not None and fold_idx in grammars:
            fold_groups[fold_idx].append((i, entry))

    # Score each fold's variants
    results = {}
    batch_size = 100

    for fold_idx in sorted(fold_groups.keys()):
        entries = fold_groups[fold_idx]
        grammar = grammars[fold_idx]
        print(f"\n  Scoring fold {fold_idx}: {len(entries):,} entries with {grammar}")

        for batch_start in tqdm(range(0, len(entries), batch_size),
                                desc=f"  Fold {fold_idx}"):
            batch = entries[batch_start:batch_start + batch_size]
            sentences = [e['variant'] for _, e in batch]
            scores = score_variants_batch(sentences, grammar, berkeley_jar)

            for (orig_idx, entry), score in zip(batch, scores):
                if score is not None:
                    # PCFG surprisal = -log_likelihood
                    # Berkeley outputs log P(w), we want surprisal = -log P(w)
                    results[orig_idx] = -score

    # Attach scores to variants
    scored_count = 0
    for i, entry in enumerate(variants):
        if i in results:
            entry['pcfg_surprisal'] = results[i]
            scored_count += 1
        else:
            entry['pcfg_surprisal'] = None

    print(f"\n  Scored: {scored_count:,} / {len(variants):,}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(variants, f)

    print(f"  Saved → {OUTPUT_FILE}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PCFG Surprisal Pipeline")
    parser.add_argument(
        '--step',
        choices=['convert', 'train', 'score', 'all'],
        default='all',
        help='Which step to run'
    )
    args = parser.parse_args()

    if args.step in ('convert', 'all'):
        print("\n" + "=" * 70)
        print(" STEP 1: DEPENDENCY → CONSTITUENCY CONVERSION")
        print("=" * 70)

        if not os.path.exists(CONLLU_FILE):
            print(f"ERROR: {CONLLU_FILE} not found!")
            sys.exit(1)

        trees = convert_all_trees(CONLLU_FILE, CONSTITUENCY_DIR)
        folds = create_folds(trees, N_FOLDS, FOLDS_DIR)

    if args.step in ('train', 'all'):
        if not os.path.exists(BERKELEY_JAR):
            print(f"ERROR: Berkeley Parser jar not found at {BERKELEY_JAR}")
            print("Download it first — see script header for instructions.")
            sys.exit(1)

        grammars = train_all_folds(FOLDS_DIR, GRAMMAR_DIR, BERKELEY_JAR, N_FOLDS)

    if args.step in ('score', 'all'):
        if not os.path.exists(VARIANTS_FILE):
            print(f"ERROR: {VARIANTS_FILE} not found. Run step 04 first.")
            sys.exit(1)

        score_all_variants(
            VARIANTS_FILE, FOLDS_DIR, GRAMMAR_DIR,
            BERKELEY_JAR, N_FOLDS
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
