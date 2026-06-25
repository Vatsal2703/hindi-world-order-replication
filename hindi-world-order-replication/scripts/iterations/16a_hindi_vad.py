#!/usr/bin/env python3
"""
Script 16a: Build Hindi VAD Lexicon (Preprocessing - Run Once)

The NRC-VAD-Lexicon-v2.1 is ENGLISH only. Our HUTB sentences are Hindi.
This script translates the English NRC-VAD terms to Hindi, producing a
reusable Hindi VAD lexicon that Script 16 reads.

Uses Helsinki-NLP/opus-mt-en-hi: Open-access, light (~300MB), no login required.
"""

import os
import sys
import csv
import argparse


# ============================================================================
# DYNAMIC PATH RESOLUTION
# ============================================================================

def find_workspace_paths(input_arg, output_arg):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir
    while repo_root and repo_root != os.path.dirname(repo_root):
        if os.path.basename(repo_root) == "hindi-world-order-replication":
            return os.path.abspath(os.path.join(repo_root, "data", "lexicon", "NRC-VAD-Lexicon-v2.1.txt")), \
                   os.path.abspath(os.path.join(repo_root, "data", "processed", "nrc_vad_hindi.tsv"))
        if os.path.isdir(os.path.join(repo_root, "hindi-world-order-replication")):
            base = os.path.join(repo_root, "hindi-world-order-replication")
            return os.path.abspath(os.path.join(base, "data", "lexicon", "NRC-VAD-Lexicon-v2.1.txt")), \
                   os.path.abspath(os.path.join(base, "data", "processed", "nrc_vad_hindi.tsv"))
        repo_root = os.path.dirname(repo_root)
    return os.path.abspath(input_arg), os.path.abspath(output_arg)


# ============================================================================
# TRANSLATION BACKEND (Helsinki Open-MT)
# ============================================================================

class OpenTranslator:
    """
    Wraps MarianMT for seamless English -> Hindi translation.
    No Hugging Face token required. Fast download.
    """
    def __init__(self):
        self.ok = False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.torch = torch
            
            model_name = "Helsinki-NLP/opus-mt-en-hi"
            print(f"  Loading {model_name} (Using secure safetensors format)...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Explicitly force loading from the safe tensors format to bypass CVE-2025-32434 check
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
            
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
            self.model.to(self.device)
            self.model.eval()
            self.ok = True
            print(f"  Translator ready on {self.device}")
        except Exception as e:
            print(f"  WARNING: Translation engine unavailable ({e})")
            print("  Make sure you ran: pip install torch transformers sentencepiece")
    def translate_batch(self, words, batch_size=128):
        """Translate a list of English words to Hindi."""
        if not self.ok:
            return [None] * len(words)
            
        results = []
        for start in range(0, len(words), batch_size):
            batch = words[start:start + batch_size]
            
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True,
                truncation=True, 
                max_length=64
            ).to(self.device)
            
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_length=64)
                
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for d in decoded:
                results.append(d.strip() if d.strip() else None)
                
            print(f"    Translated {min(start+batch_size, len(words))}/{len(words)}", end="\r")
        print()
        return results


# ============================================================================
# LEXICON BUILDING
# ============================================================================

def load_english_vad(path, unigrams_only=True):
    if not os.path.exists(path):
        print(f"ERROR: NRC-VAD lexicon file not found at evaluated absolute path:\n  {path}")
        sys.exit(1)

    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            term = row[0].strip()
            if term.lower() == 'term':
                continue
            if unigrams_only and (' ' in term or '-' in term):
                continue
            try:
                v = float(row[1])
                a = float(row[2])
                d = float(row[3])
            except ValueError:
                continue
            entries.append((term, v, a, d))
    return entries


def build_hindi_vad(input_path, output_path, unigrams_only=True):
    print("\n" + "=" * 70)
    print(" SCRIPT 16a: BUILD HINDI VAD LEXICON")
    print("=" * 70 + "\n")

    print(f"Resolved Input Path:  {input_path}")
    print(f"Resolved Output Path: {output_path}\n")

    print("Loading English NRC-VAD...")
    english = load_english_vad(input_path, unigrams_only=unigrams_only)
    print(f"  Loaded {len(english):,} English {'unigram' if unigrams_only else ''} entries")

    print("\nInitializing translator...")
    translator = OpenTranslator()
    if not translator.ok:
        print("\nERROR: Cannot proceed without an operational translation engine.")
        sys.exit(1)

    print("\nTranslating English terms to Hindi (This may take a couple of minutes)...")
    english_words = [e[0] for e in english]
    hindi_words = translator.translate_batch(english_words)

    hindi_vad = {}
    n_translated = 0
    
    for (term, v, a, d), hi in zip(english, hindi_words):
        if not hi:
            continue
        n_translated += 1
        
        # Pull clean first word if complex translations arise
        hi_tok = hi.split()[0] if hi.split() else hi
        
        if hi_tok in hindi_vad:
            ov, oa, od, cnt = hindi_vad[hi_tok]
            hindi_vad[hi_tok] = (
                (ov * cnt + v) / (cnt + 1),
                (oa * cnt + a) / (cnt + 1),
                (od * cnt + d) / (cnt + 1), 
                cnt + 1
            )
        else:
            hindi_vad[hi_tok] = (v, a, d, 1)

    print(f"  Translated {n_translated:,} terms -> {len(hindi_vad):,} unique Hindi words")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("term\tvalence\tarousal\tdominance\n")
        for hi_word, (v, a, d, cnt) in sorted(hindi_vad.items()):
            f.write(f"{hi_word}\t{v:.4f}\t{a:.4f}\t{d:.4f}\n")

    print(f"\n  Saved Hindi VAD lexicon -> {output_path}")
    print(f"  Total Hindi entries: {len(hindi_vad):,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="hindi-world-order-replication/data/lexicon/NRC-VAD-Lexicon-v2.1.txt")
    ap.add_argument("--output", default="hindi-world-order-replication/data/processed/nrc_vad_hindi.tsv")
    ap.add_argument("--all-terms", action="store_true")
    args = ap.parse_args()

    abs_input, abs_output = find_workspace_paths(args.input, args.output)
    build_hindi_vad(abs_input, abs_output, unigrams_only=not args.all_terms)