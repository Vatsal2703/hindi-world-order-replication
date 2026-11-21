#!/usr/bin/env python3
# v1.7 (2026-11-03): thesis-result box now includes the PUD replication; no logic change.
# v1.6 (2026-11-02): script numbers in comments and messages updated; naming only, no logic change.
"""
toolkit/app.py  (v1.7)  -- v1.7: result box adds the third corpus (PUD); v1.6: real HUTB demo examples (original ranked 1st on gold parse); v1.4: altair_chart width fallback for Streamlit 1.50; v1.3: one-decimal bits in the ranking table, width="stretch" (Streamlit 1.50); v1.1: surprisal chart axis from 0, not clipped; v1.2: "root" label (root can be a predicate adjective, e.g. खुश)

Hindi Emotion & Processing-Difficulty Toolkit  --  MTP demo
Vatsal Agrawal, M.Tech (Data Engineering), IIT Jodhpur. Supervisor: Prof. Sidharth Ranjan.

Type any Hindi sentence (optionally with the sentence that came before it) and the
toolkit runs the thesis pipeline on it live:

  1. Emotion      valence / arousal / dominance from the NRC-VAD v2.1 Hindi lexicon,
                  with the thesis modifier rules (negation flips the head word's
                  valence and dominance; an intensifier raises the next word's
                  arousal x1.2). Same rules as scripts 20 / 23 (v1.3).
  2. Difficulty   dependency length (thesis definition), EMILLE trigram surprisal
                  (plain backoff, scripts 20 / 23) and retrained EMILLE LSTM
                  surprisal (emille_lstm_v2.pt, scripts 28 / 29), each placed
                  against the HUTB and DISCO corpus distributions.
  3. Word order   MTP-I: every reordering of the preverbal constituents (subtree
                  blocks, script 04 rules) is scored and ranked by a pairwise
                  logistic ranker trained on HUTB (toolkit/train_demo_ranker.py:
                  DL + information status + trigram + LSTM, 91.7% 10-fold CV).

Parser: Stanza Hindi (trained on UD Hindi-HDTB, the same treebank as HUTB).

RUN (from the repo root, in the env that has torch):
    pip install streamlit stanza          # once
    streamlit run toolkit/app.py
First start loads the models (about 30-60 s); after that each sentence takes ~1 s.
"""
import os, sys, math, json, random, pickle, itertools, re
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

VERSION = "1.7"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from parsers.ud_parser import Word, Sentence                       # noqa: E402
from features.basic_features import (calculate_dependency_length_temperley,  # noqa: E402
                                     calculate_information_status_score)
P = lambda *x: os.path.join(BASE, *x)
print(f"=== toolkit/app.py v{VERSION} ===")

st.set_page_config(page_title="Hindi Emotion & Difficulty Toolkit", layout="wide")

# =============================================================================
# 1. Emotion scoring -- identical rules to script 23 v1.3 / script 20 v2.6
# =============================================================================
NEGATION = {"नहीं", "न", "मत", "ना"}
INTENSIFIERS = {"बहुत", "अत्यधिक", "ज़्यादा", "ज्यादा", "बेहद", "अत्यंत", "काफी", "इतना"}

def load_vad_lexicon(path):
    raw = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        sample = f.readline(); delim = "\t" if "\t" in sample else " "; f.seek(0)
        for line in f:
            parts = [p.strip() for p in line.strip().split(delim) if p.strip()]
            if len(parts) < 5:
                parts = line.strip().split()
                if len(parts) < 5: continue
            if parts[0].lower() == "english" or parts[4] == "Hindi": continue
            try:
                v = (float(parts[1]) - 0.5) * 2.0; a = float(parts[2])
                d = (float(parts[3]) - 0.5) * 2.0; w = parts[4].split()[0]
            except (ValueError, IndexError):
                continue
            raw[w].append((v, a, d))
    return {w: tuple(sum(s[k] for s in sc) / len(sc) for k in range(3)) for w, sc in raw.items()}

def score_emotion(toks, lex, heads):
    """Returns sentence (V, A, D, coverage) and a per-word table."""
    flipped = set()
    for i, t in enumerate(toks):
        if t in NEGATION:
            h = heads[i]
            if h and 1 <= h <= len(toks) and h - 1 != i:
                flipped.add(h - 1)
    rows, vals, aros, doms = [], [], [], []
    for i, t in enumerate(toks):
        note = ""
        if t in NEGATION:
            rows.append((t, None, None, None, "negator: flips its head word")); continue
        if t in INTENSIFIERS:
            rows.append((t, None, None, None, "intensifier: arousal of next word x1.2")); continue
        if t not in lex:
            rows.append((t, None, None, None, "not in lexicon")); continue
        v, a, d = lex[t]
        if i in flipped:
            v, d = -v, -d; note = "negated: V and D flipped"
        if i > 0 and toks[i - 1] in INTENSIFIERS:
            a = min(1.0, a * 1.2); note = (note + "; " if note else "") + "intensified"
        vals.append(v); aros.append(a); doms.append(d)
        rows.append((t, round(v, 3), round(a, 3), round(d, 3), note))
    table = pd.DataFrame(rows, columns=["word", "valence", "arousal", "dominance", "note"])
    if not vals:
        return (0.0, 0.5, 0.0, 0.0), table
    return (float(np.mean(vals)), float(np.mean(aros)), float(np.mean(doms)), len(vals) / len(toks)), table

# =============================================================================
# 2. Surprisal -- trigram (scripts 20/23 backoff) and LSTM (scripts 28/29)
# =============================================================================
def tri_prob(m, w1, w2, w3, tot):
    n, dn = m["trigrams"].get((w1, w2, w3), 0), m["bigrams"].get((w1, w2), 0)
    if n > 0 and dn > 0: return n / dn
    n, dn = m["bigrams"].get((w2, w3), 0), m["unigrams"].get(w2, 0)
    if n > 0 and dn > 0: return n / dn
    u = m["unigrams"].get(w3, 0)
    return u / tot if u > 0 else 1e-10

def trigram_bits(m, toks, tot):
    out, h = [], ["<s>", "<s>"]
    for t in toks:
        out.append(-math.log2(min(max(tri_prob(m, h[-2], h[-1], t, tot), 1e-10), 1.0))); h.append(t)
    return out

@st.cache_resource(show_spinner="Loading LSTM ...")
def load_lstm():
    import torch, torch.nn as nn
    vocab = pickle.load(open(P("data/models/emille_vocab.pkl"), "rb"))
    class VanillaLSTM(nn.Module):
        def __init__(self, V, e=200, h=200, L=2):
            super().__init__()
            self.embedding = nn.Embedding(V, e, padding_idx=0)
            self.lstm = nn.LSTM(e, h, num_layers=L, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(h, V)
        def forward(self, x):
            out, _ = self.lstm(self.embedding(x)); return self.fc(out)
    model = VanillaLSTM(len(vocab))
    model.load_state_dict(torch.load(P("data/models/emille_lstm_v2.pt"), map_location="cpu"))
    model.eval()
    return model, vocab

def lstm_bits(seqs):
    """seqs: list of token lists of equal or unequal length -> list of per-word bit lists."""
    import torch
    model, vocab = load_lstm()
    UNK, S = vocab["<UNK>"], vocab["<s>"]
    out = []
    with torch.no_grad():
        for toks in seqs:
            ids = [vocab.get(t, UNK) for t in toks]
            x = torch.tensor([[S] + ids[:-1]])
            lp = torch.log_softmax(model(x)[0], -1)
            got = lp[torch.arange(len(ids)), torch.tensor(ids)]
            out.append((-got / math.log(2)).tolist())
    return out

@st.cache_resource(show_spinner="Loading trigram model (about 2 GB, ~20 s) ...")
def load_trigram():
    m = pickle.load(open(P("data/models/trigram_model_emille.pkl"), "rb"))
    return m, max(sum(m["unigrams"].values()), 1)

@st.cache_resource(show_spinner="Loading emotion lexicon ...")
def load_lex():
    return load_vad_lexicon(P("data/processed/Hindi-NRC-VAD-Lexicon.txt"))

@st.cache_resource(show_spinner="Loading Stanza Hindi parser ...")
def load_parser():
    import stanza
    try:
        return stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                               download_method=stanza.DownloadMethod.REUSE_RESOURCES, verbose=False)
    except Exception:
        stanza.download("hi", verbose=False)
        return stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse", verbose=False)

@st.cache_resource
def load_ranker():
    return json.load(open(P("toolkit/ranker.json"), encoding="utf-8"))

@st.cache_resource(show_spinner="Loading corpus reference values ...")
def load_corpora():
    h = pd.read_csv(P("data/results/MTP2_refs_with_emille_freq.csv"), encoding="utf-8-sig")
    h = h[h.is_reference == 1]
    sc = pd.read_csv(P("data/results/lstm_v2_scores.csv"), encoding="utf-8-sig")
    hl = sc[sc.corpus == "HUTB"].set_index("key").lstm_v2_pw
    dl = sc[sc.corpus == "DISCO"].copy(); dl["key"] = pd.to_numeric(dl.key)
    dl = dl.set_index("key").lstm_v2_pw
    H = pd.DataFrame({"valence": h.tgt_valence.values, "arousal": h.tgt_arousal.values,
                      "dominance": h.tgt_dominance.values, "dl": (h.dep_len / h.n_words).values,
                      "trigram": (h.trigram_surprisal / h.n_words).values,
                      "lstm": h.sent_id.map(hl).values})
    d = pd.read_csv(P("data/results/disco_affect_scored.csv"), encoding="utf-8-sig")
    ddl = pd.read_csv(P("data/disco/disco_dl.csv"), encoding="utf-8-sig").set_index("Sentence Number").dl
    D = pd.DataFrame({"valence": d.val.values, "arousal": d.aro.values, "dominance": d.dom.values,
                      "dl": d["Sentence Number"].map(ddl).values, "trigram": d.surprisal.values,
                      "lstm": d["Sentence Number"].map(dl).values})
    return {"HUTB (written, 2,779)": H, "DISCO (task utterances, 3,171)": D}

# =============================================================================
# 3. Parsing and word-order variants (script 04 rules)
# =============================================================================
def parse(text):
    doc = load_parser()(text)
    if not doc.sentences:
        return None
    s = doc.sentences[0]
    words = [Word(w.id, w.text, w.lemma or w.text, w.upos, w.xpos, w.feats, w.head, w.deprel, None, None)
             for w in s.words]
    return Sentence(words, sent_id="input", text=" ".join(w.form for w in words))

def subtree(idx, sent):
    seen, stack = set(), [idx]
    while stack:
        c = stack.pop()
        if c in seen: continue
        seen.add(c); stack += [w.idx for w in sent.words if w.head == c]
    return sorted(seen)

def make_variants(sent, max_nonref=99, seed=42):
    root = sent.root_word
    if root is None:
        return []
    pre = sorted([w for w in sent.words if w.head == root.idx and w.idx < root.idx and w.deprel != "punct"],
                 key=lambda w: w.idx)
    if len(pre) < 2:
        return []
    blocks = [[w for w in sent.words if w.idx in set(subtree(d.idx, sent))] for d in pre]
    in_blocks = {w.idx for b in blocks for w in b}
    fixed = [w for w in sent.words if w.idx not in in_blocks]
    ref_str = " ".join(w.form for w in sent.words)
    seen, out = set(), []
    for perm in itertools.permutations(blocks):
        toks = [w for b in perm for w in b] + fixed
        s = " ".join(w.form for w in toks)
        if s in seen: continue
        seen.add(s)
        out.append({"order": [w.idx for w in toks], "text": s, "is_reference": s == ref_str})
    if not any(v["is_reference"] for v in out):          # e.g. punctuation before the verb:
        out.insert(0, {"order": [w.idx for w in sent.words],  # keep the typed order as the reference
                       "text": ref_str, "is_reference": True})
    ref = [v for v in out if v["is_reference"]]
    non = [v for v in out if not v["is_reference"]]
    if len(non) > max_nonref:
        non = random.Random(seed).sample(non, max_nonref)
    return ref + non

def pct(series, x):
    s = pd.Series(series).dropna()
    return 100.0 * (s < x).mean() if len(s) else float("nan")

# =============================================================================
# UI
# =============================================================================
st.title("Hindi Emotion & Processing-Difficulty Toolkit")
st.caption(f"v{VERSION} · MTP, M.Tech Data Engineering, IIT Jodhpur · "
           "Does the emotional tone of a Hindi sentence track how hard it is to process?")

EXAMPLES = {
    "(type your own)": ("", ""),
    # Real HUTB sentences (with the sentence before them) whose original order the ranker
    # puts 1st on the gold treebank parse; live results use the Stanza parse and can differ.
    "HUTB train-s11422 · 24 orders": ("इसके अलावा संघ भी लगातार दबाव बनाए हुए है ।",
        "मुलाकात के दौरान वाजपेयी ने मोदी को सभी को साथ लेकर चलने की हिदायत दी ।"),
    "HUTB train-s1813 · 24 orders": ("उन्होंने इस संबंध में विस्तृत ब्योरा देने से इंकार कर दिया ।",
        "वहीं पार्टी के दो वरिष्ठ नेता प्रमोद महाजन और पूर्व अध्यक्ष वेंकैया नायडू ने जोशी का बचाव करते हुए कहा कि वह पूरी तरह से निर्दोष हैं और यह सारी कवायद पार्टी की छवि को धूमिल करने का प्रयास है ।"),
    "HUTB train-s9426 · hope after grief": ("उन्होंने महाजन के जल्द स्वस्थ्य हो जाने की आशा व्यक्त की है ।",
        "उन्होंने रेखा को भरोसा दिलाया है कि दुख की इस घड़ी में वह उनके परिवार के साथ हैं ।"),
    "HUTB train-s9275 · IO-fronted": ("जबकि पिछले चुनाव में कांग्रेस को १० सीटें मिली थीं ।",
        "इस तरह उसे पूर्ण बहुमत से २० सीटें ज्यादा मिली हैं ।"),
    "Negation + intensifier (emotion)": ("मैं आज बहुत खुश नहीं हूँ ।", ""),
    "Made-up sentence (word order)": ("पुलिस ने कल शाम दिल्ली में दो चोरों को पकड़ा ।",
                                       "दिल्ली में कल कई चोरियाँ हुईं ।"),
}
with st.sidebar:
    st.header("Input")
    ex = st.selectbox("Example", list(EXAMPLES))
    sent_txt = st.text_area("Hindi sentence", value=EXAMPLES[ex][0], height=90)
    prev_txt = st.text_area("Previous sentence (optional, used for information status)",
                            value=EXAMPLES[ex][1], height=70)
    go = st.button("Analyse", type="primary", width="stretch")
    st.divider()
    st.markdown("**Thesis findings (both corpora, length and frequency controlled)**\n\n"
                "- Longer dependencies → lower dominance (β = −0.090 HUTB, −0.087 DISCO)\n"
                "- Higher LSTM surprisal → higher valence (β = +0.108, +0.094)\n"
                "- Trigram surprisal effects flip sign between corpora\n\n"
                "**MTP-I:** replication of Ranjan et al. (EMNLP 2022), 94.62% vs 95.06%.")

if not (go or sent_txt.strip()):
    st.info("Pick an example or type a Hindi sentence in the sidebar, then press **Analyse**.")
    st.stop()
if not sent_txt.strip():
    st.stop()

sent = parse(sent_txt.strip())
if sent is None or not sent.words:
    st.error("The parser found no words in this input."); st.stop()
ctx = parse(prev_txt.strip()) if prev_txt.strip() else None
toks = [w.form for w in sent.words]
heads = [w.head for w in sent.words]
n = len(toks)

lex = load_lex()
tri, tot = load_trigram()
(V, A, D, cov), emo_table = score_emotion(toks, lex, heads)
tri_w = trigram_bits(tri, toks, tot)
lstm_w = lstm_bits([toks])[0]
dl_thesis = sum(abs(w.idx - w.head) for w in sent.words if w.head != 0) / n
corp = load_corpora()

st.subheader("Parsed sentence")
st.markdown(" ".join(f"`{w.form}`" for w in sent.words))
st.caption(f"{n} tokens · head of the sentence (root): **{sent.root_word.form if sent.root_word else '—'}**")

tab1, tab2, tab3 = st.tabs(["1 · Emotion", "2 · Processing difficulty", "3 · Word order (MTP-I)"])

# ---- Tab 1: Emotion ---------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valence (−1 … +1)", f"{V:+.3f}", help="unpleasant → pleasant")
    c2.metric("Arousal (0 … 1)", f"{A:.3f}", help="calm → excited")
    c3.metric("Dominance (−1 … +1)", f"{D:+.3f}", help="controlled → in control")
    c4.metric("Lexicon coverage", f"{cov:.0%}")
    st.markdown("**Word-level scores** (NRC-VAD v2.1 Hindi; sentence score = mean over matched words)")
    st.dataframe(emo_table, width="stretch", hide_index=True)
    st.markdown("**Where this sentence sits in each corpus** (percentile: share of corpus sentences below it)")
    st.dataframe(pd.DataFrame({k: {"valence": f"{pct(df.valence, V):.0f}%", "arousal": f"{pct(df.arousal, A):.0f}%",
                                   "dominance": f"{pct(df.dominance, D):.0f}%"} for k, df in corp.items()}),
                 width="stretch")

# ---- Tab 2: Difficulty ------------------------------------------------------
with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dependency length / word", f"{dl_thesis:.2f}")
    c2.metric("Trigram surprisal / word (bits)", f"{np.mean(tri_w):.2f}")
    c3.metric("LSTM surprisal / word (bits)", f"{np.mean(lstm_w):.2f}")
    st.markdown("**Per-word surprisal** (bits; higher = less predictable)")
    import altair as alt                    # explicit axis from 0 so tall bars are not clipped
    _lab = [f"{i+1:02d}. {t}" for i, t in enumerate(toks)]
    _bars = pd.DataFrame({"word": _lab * 2, "model": ["LSTM"] * n + ["trigram"] * n,
                          "bits": list(lstm_w) + list(tri_w)})
    _ymax = float(max(max(lstm_w), max(tri_w))) * 1.08
    _chart = (alt.Chart(_bars).mark_bar().encode(
        x=alt.X("word:N", sort=_lab, title=None, axis=alt.Axis(labelAngle=-45)),
        xOffset="model:N",
        y=alt.Y("bits:Q", scale=alt.Scale(domain=[0, _ymax]), title="surprisal (bits)"),
        color=alt.Color("model:N", scale=alt.Scale(range=["#1f5fbf", "#8fc1f2"]), title=None),
        tooltip=["word", "model", alt.Tooltip("bits:Q", format=".2f")]).properties(height=340))
    try:
        st.altair_chart(_chart, width="stretch")
    except TypeError:                      # Streamlit versions without `width` here
        st.altair_chart(_chart, use_container_width=True)
    st.markdown("**Dependency parse** (distance = |position − head position|)")
    st.dataframe(pd.DataFrame([{"#": w.idx, "word": w.form, "lemma": w.lemma, "POS": w.upos, "relation": w.deprel,
                                "head": (sent.words[w.head - 1].form if w.head else "ROOT"),
                                "distance": (abs(w.idx - w.head) if w.head else 0)} for w in sent.words]),
                 width="stretch", hide_index=True)
    st.markdown("**Percentile in each corpus**")
    st.dataframe(pd.DataFrame({k: {"dependency length": f"{pct(df.dl, dl_thesis):.0f}%",
                                   "trigram surprisal": f"{pct(df.trigram, np.mean(tri_w)):.0f}%",
                                   "LSTM surprisal": f"{pct(df.lstm, np.mean(lstm_w)):.0f}%"}
                               for k, df in corp.items()}), width="stretch")
    st.info("Thesis result: in all three corpora (HUTB, DISCO, PUD), sentences with longer dependencies have "
            "lower dominance. Sentences with higher LSTM surprisal have higher valence in HUTB and DISCO, but not "
            "in PUD. Effects are small (about 0.1 SD per SD) and correlational; one sentence cannot confirm or "
            "refute them.")

# ---- Tab 3: Word order ------------------------------------------------------
with tab3:
    variants = make_variants(sent)
    if not variants:
        st.warning("This sentence has fewer than two constituents before the verb, so there is nothing to reorder. "
                   "Try a sentence with a subject, an object and/or adverbials before the verb.")
    else:
        R = load_ranker(); w = np.array(R["weights"])
        wm = {x.idx: x.form for x in sent.words}
        seqs = [[wm[i] for i in v["order"]] for v in variants]
        lstm_tot = [sum(b) for b in lstm_bits(seqs)]
        rows = []
        for v, s_toks, lt in zip(variants, seqs, lstm_tot):
            f = np.array([calculate_dependency_length_temperley(sent, v["order"]),
                          calculate_information_status_score(sent, v["order"], ctx),
                          sum(trigram_bits(tri, s_toks, tot)), lt])
            rows.append({"order": v["text"], "original": "✓" if v["is_reference"] else "",
                         "dep. length": int(f[0]), "info status": int(f[1]),
                         "trigram bits": round(f[2], 1), "LSTM bits": round(f[3], 1),
                         "score": float(w @ f)})
        T = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        T.index = T.index + 1
        rank = int(T.index[T.original == "✓"][0])
        nvar = len(T) - 1
        if rank == 1:
            st.success(f"The original order is ranked **1st** of {len(T)} orders ({nvar} generated variants).")
        else:
            st.warning(f"The original order is ranked **{rank}** of {len(T)}. The model prefers: "
                       f"**{T.loc[1, 'order']}**")
        st.dataframe(T.style.format({"score": "{:.2f}", "trigram bits": "{:.1f}", "LSTM bits": "{:.1f}"}), width="stretch", height=min(420, 38 + 35 * len(T)))
        st.markdown("**Why:** contribution of each feature to the score (weight × value), original vs the best "
                    "alternative order")
        alt = T[T.original != "✓"].iloc[0]; ref = T[T.original == "✓"].iloc[0]
        names = ["dep. length", "info status", "trigram bits", "LSTM bits"]
        st.dataframe(pd.DataFrame({"weight": w, "original": [w[i] * ref[c] for i, c in enumerate(names)],
                                   "best alternative": [w[i] * alt[c] for i, c in enumerate(names)]},
                                  index=names).round(2), width="stretch")
        acc = R["cv_accuracy"]
        st.caption(f"Ranker: logistic regression on reference-minus-variant differences, trained on "
                   f"{R['n_sentences']:,} HUTB sentences ({R['n_pairs']:,} pairs); 10-fold CV accuracy "
                   f"{acc['all four']}% (trigram alone {acc['trigram alone']}%, LSTM alone {acc['LSTM alone']}%, "
                   f"DL alone {acc['DL alone']}%). The full MTP-I model (94.62%) also uses PCFG and "
                   "discourse-adapted surprisal, which need the whole document and are not computed live.")
st.caption(f"toolkit v{VERSION}")
