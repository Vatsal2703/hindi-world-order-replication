#!/usr/bin/env python3
"""
pud_full_pipeline.py — Full MTP-II Validation on Hindi-PUD (with LSTM)

Now that emille_base_lstm.pt is trained, runs the complete 7-feature
pipeline on Hindi-PUD including adaptive LSTM surprisal, then tests
the emotion x adaptive interaction.

RUN IN base CONDA ENV (has torch 2.2.2 + MPS):
    python scripts/pud_full_pipeline.py

Outputs in data_pud/results/
"""

import os, sys, pickle, math, copy, random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))
from parsers.ud_parser import UDParser

# ============================================================================
# PATHS
# ============================================================================
def find_base():
    d = SCRIPT_DIR
    while d and d != os.path.dirname(d):
        if os.path.basename(d) == "hindi-world-order-replication": return d
        if os.path.isdir(os.path.join(d, "hindi-world-order-replication")):
            return os.path.join(d, "hindi-world-order-replication")
        d = os.path.dirname(d)
    return "."

BASE     = find_base()
PUD_FILE = os.path.expanduser("~/Downloads/UD_Hindi-PUD-master/hi_pud-ud-test.conllu")
TRIGRAM  = os.path.join(BASE, "data", "models", "trigram_model.pkl")
LSTM_PT  = os.path.join(BASE, "data", "models", "emille_base_lstm.pt")
VOCAB    = os.path.join(BASE, "data", "models", "emille_vocab.pkl")
VAD_LEX  = os.path.join(BASE, "data", "processed", "Hindi-NRC-VAD-Lexicon.txt")
OUT_DIR  = os.path.join(BASE, "data_pud", "results")

NEGATION     = {"नहीं", "न", "मत"}
SUBJECT_RELS = {"nsubj", "csubj"}
OBJECT_RELS  = {"obj", "iobj"}
INTENSIFIERS = {"बहुत","अत्यधिक","ज़्यादा","ज्यादा","बेहद","अत्यंत","काफी","इतना"}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# LSTM MODEL
# ============================================================================
class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=200, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.lstm(x)
        return self.fc(out), hidden

def get_surprisal(model, tokens, vocab):
    model.eval()
    if not tokens: return 0.0
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).to(device)
    with torch.no_grad():
        logits, _ = model(input_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs[0, torch.arange(len(ids)), target_ids]
        return -gathered.sum().item() / math.log(2)

def adapt_model(base_model, tokens, vocab, lr=2.0):
    adapted = copy.deepcopy(base_model)
    adapted.train()
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if not tokens: return adapted
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_ids = torch.tensor([vocab["<s>"]] + ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(ids).unsqueeze(0).to(device)
    optimizer.zero_grad()
    logits, _ = adapted(input_ids)
    loss = criterion(logits.transpose(1, 2), target_ids)
    loss.backward()
    optimizer.step()
    adapted.eval()
    return adapted

# ============================================================================
# FILTER PUD (5 conditions — skip declarative)
# ============================================================================
def is_projective(sent):
    arcs = [(min(w.idx,w.head),max(w.idx,w.head))
            for w in sent.words if w.head>0 and not w.is_root()]
    for i,(s1,e1) in enumerate(arcs):
        for s2,e2 in arcs[i+1:]:
            if s1<s2<e1<e2 or s2<s1<e2<e1: return False
    return True

def filter_pud(sentences):
    passed = []
    for s in sentences:
        if s.root_idx is None: continue
        children = s.get_children(s.root_idx)
        if not any(w.deprel in SUBJECT_RELS for w in children): continue
        if not any(w.deprel in OBJECT_RELS  for w in children): continue
        if not is_projective(s): continue
        if s.root_word is None or s.root_word.upos not in ("VERB","AUX"): continue
        if any(w.form in NEGATION for w in s.words): continue
        preverbal = [w for w in children
                     if w.idx < s.root_idx and w.upos != "PUNCT"]
        if len(preverbal) < 2: continue
        passed.append(s)
    return passed

# ============================================================================
# VARIANT GENERATION
# ============================================================================
def get_subtree(sent, head_idx):
    result = {head_idx}
    for w in sent.words:
        if w.head == head_idx and w.idx != head_idx:
            result |= get_subtree(sent, w.idx)
    return result

def generate_variants(sent, max_variants=99):
    if sent.root_idx is None: return []
    children = sent.get_children(sent.root_idx)
    preverbal = sorted([w for w in children
                        if w.idx < sent.root_idx and w.upos != "PUNCT"],
                       key=lambda w: w.idx)
    if len(preverbal) < 2: return []
    blocks = [sorted(get_subtree(sent, pw.idx)) for pw in preverbal]
    all_pre = set(idx for b in blocks for idx in b)
    post_v  = [w.idx for w in sent.words if w.idx not in all_pre]
    ref_order = [idx for b in blocks for idx in b] + post_v
    from itertools import permutations
    variants = []
    seen = {tuple(ref_order)}
    for perm in permutations(range(len(blocks))):
        var_order = [idx for i in perm for idx in blocks[i]] + post_v
        key = tuple(var_order)
        if key not in seen:
            seen.add(key)
            variants.append(var_order)
            if len(variants) >= max_variants: break
    return variants

# ============================================================================
# FEATURES
# ============================================================================
def trigram_surprisal(model, tokens):
    total = 0.0
    history = ["<s>", "<s>"]
    for tok in tokens:
        w1,w2 = history[-2], history[-1]
        tri = model.get("trigrams",{}).get((w1,w2,tok),0)
        bi  = model.get("bigrams", {}).get((w1,w2),0)
        uni = model.get("unigrams",{}).get(tok,0)
        total_uni = model.get("total_unigrams",1)
        if tri > 0:   p = tri/bi if bi>0 else 1e-10
        elif uni > 0: p = uni/total_uni
        else:         p = 1e-10
        total += -math.log2(max(p,1e-10))
        history.append(tok)
    return total

def dep_length(sent, order):
    idx_to_pos = {idx:pos for pos,idx in enumerate(order)}
    return sum(abs(idx_to_pos[w.idx]-idx_to_pos[w.head])
               for w in sent.words
               if w.head>0 and not w.is_root()
               and w.idx in idx_to_pos and w.head in idx_to_pos)

def info_status_score(sent, order):
    if sent.root_idx is None: return 0.0
    children = sorted(sent.get_children(sent.root_idx), key=lambda w: w.idx)
    preverbal = [w for w in children if w.idx<sent.root_idx and w.upos!="PUNCT"]
    if len(preverbal)<2: return 0.0
    idx_to_pos = {idx:pos for pos,idx in enumerate(order)}
    score = 0.0
    for w in preverbal:
        subtree_size = len(get_subtree(sent,w.idx))
        pos = idx_to_pos.get(w.idx,0)
        score += (len(order)-pos)/max(subtree_size,1)
    return score

def load_vad_lexicon(path):
    raw = defaultdict(list)
    with open(path,"r",encoding="utf-8") as f:
        sample = f.readline()
        delim = "\t" if "\t" in sample else " "
        f.seek(0)
        for line in f:
            parts = [p.strip() for p in line.strip().split(delim) if p.strip()]
            if len(parts)<5: parts=line.strip().split()
            if len(parts)<5: continue
            if parts[0].lower()=="english" or parts[4]=="Hindi": continue
            try:
                v=(float(parts[1])-0.5)*2.0
                a=float(parts[2])
                w=parts[4].split()[0]
                raw[w].append((v,a))
            except: continue
    return {w:(sum(s[0] for s in sc)/len(sc),
               sum(s[1] for s in sc)/len(sc))
            for w,sc in raw.items()}

def score_emotion(words, lexicon):
    vals,aros=[],[]
    for i,w in enumerate(words):
        form=w.form.strip()
        if form not in lexicon: continue
        v,a=lexicon[form]
        if i>0 and words[i-1].form.strip() in NEGATION: v=-v
        if i>0 and words[i-1].form.strip() in INTENSIFIERS: a=min(1.0,a*1.2)
        vals.append(v); aros.append(a)
    if not vals: return 0.0,0.5,0.0
    return sum(vals)/len(vals),sum(aros)/len(aros),len(vals)/max(len(words),1)

def get_construction(sent):
    if sent.root_idx is None: return "UNKNOWN"
    try: preverbal=sent.get_preverbal_constituents()
    except: return "UNKNOWN"
    subj=obj=iobj=None
    for i,w in enumerate(preverbal):
        if w.deprel in SUBJECT_RELS and subj is None: subj=i
        elif w.deprel=="obj" and obj is None: obj=i
        elif w.deprel=="iobj" and iobj is None: iobj=i
    if subj is None: return "UNKNOWN"
    if obj  is not None and obj  < subj: return "DOSV"
    if iobj is not None and iobj < subj: return "IOSV"
    return "SOV"

def sig_mark(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""

# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n"+"="*70)
    print(" PUD FULL PIPELINE: MTP-II WITH ADAPTIVE LSTM")
    print("="*70)

    # Load models
    print(f"\n[1] Loading models (device: {device})...")
    with open(VOCAB,"rb") as f: vocab=pickle.load(f)
    base_model=VanillaLSTM(len(vocab)).to(device)
    base_model.load_state_dict(torch.load(LSTM_PT,map_location=device))
    base_model.eval()
    with open(TRIGRAM,"rb") as f: trigram=pickle.load(f)
    lexicon=load_vad_lexicon(VAD_LEX)
    print(f"  Vocab: {len(vocab):,}  Lexicon: {len(lexicon):,}  LSTM: ready")

    # Parse & filter PUD
    print("\n[2] Parsing and filtering Hindi-PUD...")
    parser=UDParser()
    all_sents=parser.parse_file(PUD_FILE,verbose=False)
    sents=filter_pud(all_sents)
    print(f"  {len(all_sents):,} total → {len(sents):,} after filtering")

    # Construction distribution
    ctypes=[get_construction(s) for s in sents]
    dist=pd.Series(ctypes).value_counts()
    total=len(ctypes)
    print("\n  Construction distribution:")
    for ct in ["SOV","DOSV","IOSV","UNKNOWN"]:
        n=dist.get(ct,0)
        print(f"    {ct:>8s}: {n:>4} ({100*n/total:.1f}%)")

    # Build dataset with ALL features
    print("\n[3] Computing features (trigram + DL + IS + LSTM + adaptive)...")
    ordered_ids=[s.sent_id for s in sents]
    sent_lookup={s.sent_id:s for s in sents}
    rows=[]
    n_ref=0

    for i,sent in enumerate(sents):
        if sent.root_idx is None: continue
        children=sent.get_children(sent.root_idx)
        preverbal=sorted([w for w in children
                          if w.idx<sent.root_idx and w.upos!="PUNCT"],
                         key=lambda w:w.idx)
        if len(preverbal)<2: continue
        blocks=[sorted(get_subtree(sent,pw.idx)) for pw in preverbal]
        all_pre=set(idx for b in blocks for idx in b)
        post_v=[w.idx for w in sent.words if w.idx not in all_pre]
        ref_order=[idx for b in blocks for idx in b]+post_v

        # Context tokens (preceding sentence)
        context_tokens=None
        if i>0:
            prev=sent_lookup.get(ordered_ids[i-1])
            if prev: context_tokens=[w.form for w in prev.words]

        # Emotion of preceding sentence
        if i>0 and context_tokens:
            prev=sent_lookup.get(ordered_ids[i-1])
            if prev: e_val,e_aro,e_cov=score_emotion(prev.words,lexicon)
            else: e_val,e_aro,e_cov=0.0,0.5,0.0
        else: e_val,e_aro,e_cov=0.0,0.5,0.0

        # Adapt model to context
        if context_tokens:
            adapted=adapt_model(base_model,context_tokens,vocab,lr=2.0)
        else:
            adapted=base_model

        ctype=get_construction(sent)
        n_ref+=1

        variants=generate_variants(sent)
        for var_order in variants:
            ref_tokens=[sent.words[idx-1].form for idx in ref_order if 0<idx<=len(sent.words)]
            var_tokens=[sent.words[idx-1].form for idx in var_order if 0<idx<=len(sent.words)]

            ref_tri=trigram_surprisal(trigram,ref_tokens)
            var_tri=trigram_surprisal(trigram,var_tokens)
            ref_dl=dep_length(sent,ref_order)
            var_dl=dep_length(sent,var_order)
            ref_is=info_status_score(sent,ref_order)
            var_is=info_status_score(sent,var_order)
            ref_lstm=get_surprisal(base_model,ref_tokens,vocab)
            var_lstm=get_surprisal(base_model,var_tokens,vocab)
            ref_adap=get_surprisal(adapted,ref_tokens,vocab)
            var_adap=get_surprisal(adapted,var_tokens,vocab)

            for label,sign in [(1,1),(0,-1)]:
                rows.append({
                    "sent_id":           sent.sent_id,
                    "label":             label,
                    "construction_type": ctype,
                    "dep_len_diff":      sign*(ref_dl-var_dl),
                    "info_status_diff":  sign*(ref_is-var_is),
                    "trigram_surp_diff": sign*(ref_tri-var_tri),
                    "lstm_surp_diff":    sign*(ref_lstm-var_lstm),
                    "adaptive_surp_diff":sign*(ref_adap-var_adap),
                    "prev_valence":      e_val,
                    "prev_arousal":      e_aro,
                })

    df=pd.DataFrame(rows)
    print(f"  {n_ref} sentences → {len(df):,} pairwise instances")

    # Emotion distribution
    print("\n[4] Emotion distribution...")
    ref_df=df.drop_duplicates("sent_id")
    vals=ref_df["prev_valence"].tolist()
    aros=ref_df["prev_arousal"].tolist()
    print(f"  Valence: [{min(vals):.3f}, {max(vals):.3f}]  mean={sum(vals)/len(vals):.3f}")
    print(f"  Arousal: [{min(aros):.3f}, {max(aros):.3f}]  mean={sum(aros)/len(aros):.3f}")

    # Classification
    print("\n[5] Classification (10-fold CV)...")
    y=df["label"].values.astype(int)
    feats_A=["dep_len_diff","info_status_diff","trigram_surp_diff","lstm_surp_diff","adaptive_surp_diff"]
    feats_B=feats_A+["prev_valence","prev_arousal"]

    adp_c=df["adaptive_surp_diff"]-df["adaptive_surp_diff"].mean()
    val_c=df["prev_valence"]-df["prev_valence"].mean()
    aro_c=df["prev_arousal"]-df["prev_arousal"].mean()
    df["val_x_adp_c"]=val_c*adp_c
    df["aro_x_adp_c"]=aro_c*adp_c
    feats_C=feats_B+["val_x_adp_c","aro_x_adp_c"]

    def cv_pred(X,y,k=10):
        skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
        preds=np.zeros(len(y),dtype=int)
        for tr,te in skf.split(X,y):
            clf=LogisticRegression(max_iter=1000)
            clf.fit(X[tr],y[tr])
            preds[te]=clf.predict(X[te])
        return preds

    pA=cv_pred(df[feats_A].values,y)
    pB=cv_pred(df[feats_B].values,y)
    pC=cv_pred(df[feats_C].values,y)
    accA=(pA==y).mean(); accB=(pB==y).mean(); accC=(pC==y).mean()

    def mcn(y,p1,p2):
        c1,c2=(p1==y),(p2==y)
        n01=np.sum(~c1&c2); n10=np.sum(c1&~c2)
        return mcnemar([[0,n01],[n10,0]],exact=False,correction=True).pvalue

    print(f"\n  Model A (5 features)         : {accA:.4%}")
    print(f"  Model B (+emotion)            : {accB:.4%}  (Δ {accB-accA:+.4%})")
    print(f"  Model C (+interactions)       : {accC:.4%}  (Δ {accC-accB:+.4%})")
    print(f"\n  McNemar B vs A (H1)          : p={mcn(y,pA,pB):.4g}")
    print(f"  McNemar C vs B (H2/H3)       : p={mcn(y,pB,pC):.4g}")

    print("\n  Interaction coefficients (centered):")
    Xc=sm.add_constant(df[feats_C].values)
    try:
        m=sm.Logit(y,Xc).fit(disp=0)
        names=["const"]+feats_C
        for i,nm in enumerate(names):
            if nm in ("val_x_adp_c","aro_x_adp_c"):
                b,t,pv=m.params[i],m.tvalues[i],m.pvalues[i]
                print(f"    {nm:>14s}: β={b:+.4f}  t={t:+.2f}  p={pv:.4g} {sig_mark(pv)}")
    except Exception as e:
        print(f"  (Coefficient fit failed: {e})")

    # Save
    df.to_csv(os.path.join(OUT_DIR,"pud_full_features.csv"),index=False)
    print(f"\n  Saved -> {OUT_DIR}/pud_full_features.csv")
    print("="*70+"\n")

if __name__ == "__main__":
    main()