#!/usr/bin/env python3
"""
Universal Dependencies Hindi Parser
For UD_Hindi-HDTB corpus (.conllu format)
"""

from functools import cached_property
import os
from typing import List, Dict, Optional

SUBJECT_RELATIONS = {'nsubj', 'nsubj:pass', 'csubj', 'csubj:pass'}
OBJECT_RELATIONS = {'obj', 'iobj'}

class Word:
    def __init__(self, idx, form, lemma, upos, xpos, feats, head, deprel, deps, misc):
        self.idx = idx
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
    
    def __repr__(self):
        return f"{self.idx}:{self.form}({self.deprel}→{self.head})"
    
    def is_root(self) -> bool:
        return self.head == 0 or self.deprel == 'root'



class Sentence:
    def __init__(self, words, sent_id="", text=""):
        self.words = words
        self.sent_id = sent_id
        self.text = text
        
        # Find root
        self._root_idx = None
        for word in self.words:
            if word.is_root():
                self._root_idx = word.idx
                break
    
    @property
    def root_idx(self):
        return self._root_idx
    
    @property
    def root_word(self):
        if self._root_idx is None:
            return None
        # CoNLL-U indices are 1-based, list indices are 0-based
        return self.words[self._root_idx - 1]
    
    def get_children(self, head_idx):
        return [w for w in self.words if w.head == head_idx]
    
    def get_words_by_relation(self, relation):
        return [w for w in self.words if w.deprel == relation]
    
    def has_relation(self, relation):
        return any(w.deprel == relation for w in self.words)
    
    @cached_property
    def has_subject(self) -> bool:
        return any(w.deprel in SUBJECT_RELATIONS for w in self.words)
    
    @cached_property
    def has_object(self) -> bool:
        return any(w.deprel in OBJECT_RELATIONS for w in self.words)
    
    def get_preverbal_constituents(self):
        if self.root_idx is None:
            return []
        
        root_children = self.get_children(self.root_idx)
        preverbal = [w for w in root_children if w.idx < self.root_idx]
        preverbal.sort(key=lambda w: w.idx)
        
        return preverbal
    
    def __len__(self):
        return len(self.words)
    
    def __repr__(self):
        root_form = self.root_word.form if self.root_word else 'None'
        return f"Sentence(id={self.sent_id}, len={len(self)}, root={root_form})"

class UDParser:
    
    def __init__(self):
        self.sentences: List[Sentence] = []
        self.stats = {
            'total_sentences': 0,
            'total_words': 0,
            'parse_errors': 0,
            'skipped_lines': 0
        }
    
    def parse_line(self, line: str, line_num: int) -> Optional[Word]:
        parts = line.split('\t')

        # Skip multi-word tokens (e.g., '1-2') and empty nodes (e.g., '3.1')
        if not parts[0].isdigit():
            self.stats['skipped_lines'] += 1
            return None

        try:
            word = Word(
                idx=int(parts[0]),
                form=parts[1],
                lemma=parts[2],
                upos=parts[3],
                xpos=parts[4],
                feats=parts[5],
                head=int(parts[6]) if parts[6].isdigit() else 0,
                deprel=parts[7],
                deps=parts[8] if len(parts) > 8 else '_',
                misc=parts[9] if len(parts) > 9 else '_'
            )
            
            return word
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing line {line_num}: {line[:80]}")
            print(f"  Error: {e}")
            self.stats['parse_errors'] += 1
            return None
    
    def parse_file(self, filepath: str, verbose: bool = True) -> List[Sentence]:
        sentences = []
        current_words = []
        sent_id = ""
        sent_text = ""
        
        if verbose:
            print(f"Parsing: {os.path.basename(filepath)}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        if current_words:
                            sent = Sentence(
                                words=current_words,
                                sent_id=sent_id,
                                text=sent_text
                            )
                            sentences.append(sent)
                            self.stats['total_words'] += len(current_words)
                            current_words = []
                            sent_id = ""
                            sent_text = ""
                        continue
                    
                    if line.startswith('#'):
                        if line.startswith('# sent_id'):
                            sent_id = line.split('=', 1)[1].strip()
                        elif line.startswith('# text'):
                            sent_text = line.split('=', 1)[1].strip()
                        continue
                    
                    word = self.parse_line(line, line_num)
                    if word:
                        current_words.append(word)
                
                if current_words:
                    sent = Sentence(
                        words=current_words,
                        sent_id=sent_id,
                        text=sent_text
                    )
                    sentences.append(sent)
                    self.stats['total_words'] += len(current_words)
        
        except FileNotFoundError:
            print(f"ERROR: File not found: {filepath}")
            return []
        except Exception as e:
            print(f"ERROR parsing {filepath}: {e}")
            return []
        
        self.stats['total_sentences'] += len(sentences)
        
        if verbose:
            print(f"  → Parsed {len(sentences):,} sentences")
        
        return sentences
    
    def print_stats(self):
        print(f"\n{'='*60}")
        print("PARSING STATISTICS")
        print(f"{'='*60}")
        print(f"Sentences: {self.stats['total_sentences']:,}")
        print(f"Words: {self.stats['total_words']:,}")
        print(f"Parse errors: {self.stats['parse_errors']}")
        print(f"Skipped lines: {self.stats['skipped_lines']}")
        if self.stats['total_sentences'] > 0:
            avg_len = self.stats['total_words'] / self.stats['total_sentences']
            print(f"Average sentence length: {avg_len:.2f} words")
        print(f"{'='*60}\n")


def parse_ud_hindi(ud_dir: str) -> List[Sentence]:
    parser = UDParser()
    all_sentences = []
    
    print(f"\n{'='*60}")
    print("PARSING UD HINDI-HDTB CORPUS")
    print(f"{'='*60}\n")
    
    files = [
        'hi_hdtb-ud-train.conllu'
    ]
    
    for filename in files:
        filepath = os.path.join(ud_dir, filename)
        if os.path.exists(filepath):
            sentences = parser.parse_file(filepath, verbose=True)
            all_sentences.extend(sentences)
            print()
        else:
            print(f"️ile not found: {filename}\n")
    
    parser.print_stats()
    
    return all_sentences


def filter_valid_sentences(sentences: List[Sentence]) -> List[Sentence]:
    # 1. Fixed: removed () here
    valid = [s for s in sentences if s.has_subject and s.has_object]    
    
    print(f"{'='*60}")
    print("FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total sentences: {len(sentences):,}")
    
    # 2. Fixed: removed () from s.has_subject
    print(f"With subject (nsubj): {sum(1 for s in sentences if s.has_subject):,}")
    
    # 3. Fixed: removed () from s.has_object
    print(f"With object (obj/iobj): {sum(1 for s in sentences if s.has_object):,}")
    
    print(f"With BOTH subject AND object: {len(valid):,}")
    print(f"\nTarget for paper: 1,996")
    print(f"Status: {' SUFFICIENT' if len(valid) >= 1996 else '️  INSUFFICIENT'}")
    print(f"{'='*60}\n")
    
    return valid

def show_sample_sentences(sentences: List[Sentence], n: int = 3):
    print(f"{'='*60}")
    print(f"SAMPLE SENTENCES (first {n})")
    print(f"{'='*60}\n")
    
    for i, sent in enumerate(sentences[:n], 1):
        print(f"{i}. ID: {sent.sent_id}")
        print(f"   Text: {sent.text}")
        print(f"   Length: {len(sent)} words")
        
        if sent.root_word:
            print(f"   Root: {sent.root_word.form} (pos {sent.root_idx})")
        
        subjects = [w for w in sent.words if w.deprel in ['nsubj', 'nsubj:pass']]
        if subjects:
            print(f"   Subjects: {[w.form for w in subjects]}")
        
        objects = [w for w in sent.words if w.deprel in ['obj', 'iobj']]
        if objects:
            print(f"   Objects: {[w.form for w in objects]}")
        
        preverbal = sent.get_preverbal_constituents()
        print(f"   Preverbal constituents: {len(preverbal)}")
        for w in preverbal:
            print(f"      - {w.form} ({w.deprel})")
        
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ud_parser.py <path_to_UD_Hindi_HDTB>")
        sys.exit(1)
    
    ud_dir = sys.argv[1]
    sentences = parse_ud_hindi(ud_dir)
    
    if not sentences:
        print("\n No sentences parsed!")
        sys.exit(1)
    
    valid_sentences = filter_valid_sentences(sentences)
    show_sample_sentences(valid_sentences, n=3)
