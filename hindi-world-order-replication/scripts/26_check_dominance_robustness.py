#!/usr/bin/env python3
# v1.3 (2026-11-02): renumbered from 34_check_dominance_robustness.py; naming only, no logic change.
# v1.2 (2026-11-02): renamed from 34_dominance_robustness.py; naming only, no logic change.
"""
Script 26: Is the dominance-difficulty effect real, and is it just syntax? — v1.3

Follows the 26 Aug 2026 supervisor direction (difficulty WITHIN the 2,779 HUTB
reference sentences; hypothesis: higher difficulty -> lower dominance) and the
1 Sep result (dominance falls across trigram-surprisal tertiles, d = -0.40).

It answers two questions in one place, so the result is reproducible from the
repo (the 1 Sep script and the old per-word diagnostic, now extra_diag/diagnose_perword.py, were removed in the 1 Oct clean-up):

  STEP 1  Artefact checks
    1a  Re-score emotion from the gold HDTB trees with script 20's rules and
        check it matches tgt_dominance in MTP2_raw_perword.csv.
    1b  Tertile table (1 Sep analysis): mean dominance in low / mid / high
        per-word trigram-surprisal tertiles, Cohen's d high vs low.
    1c  Content-word dominance: score only NOUN/PROPN/VERB/ADJ/ADV, so
        function words such as ने (dominance -0.51) cannot drive the effect.

  STEP 2  Syntax controls — does low dominance just stand in for structure?
    Nested OLS models, emotion as outcome (same orientation as script 24):
      dom_z ~ difficulty_z + log_len_z + zipf_z                       M0 base
            + passive + ergative                                      M1 voice/case
            + C(construction) + info_status + n_preverbal             M2 order
            + n_clauses                                               M3 complexity
            + val_z                                                   M4 + valence
    difficulty in {trigram surprisal, LSTM surprisal, dependency length}
    (all per word, from MTP2_raw_perword.csv), dominance in {all words,
    content words}. HC3 SEs; 1,000 bootstrap resamples on M3.
    "change_vs_M0" = beta / beta(M0) - 1: negative = the controls absorb part
    of the effect, positive = the effect gets stronger once syntax is held
    fixed (a suppressor, e.g. ने lowers dominance but also lowers surprisal).
    Reading: if beta stays negative and significant at M3, the effect is not
    explained by these structures; if it drops to ~0, the finding becomes
    "difficult constructions carry low-control meaning".

Syntactic features (from the gold reference tree, root = main verb):
  passive        root has an aux:pass or nsubj:pass dependent, or Voice=Pass
  ergative       sentence contains ने as a case marker
  construction   SOV / DOSV / IOSV / UNKNOWN (ref_construction_type, script 20)
  info_status    given-new score of the reference order (script 20 rule)
  n_preverbal    number of preverbal root dependents (excl. punctuation)
  n_clauses      count of advcl / acl / acl:relcl / ccomp / xcomp / conj(VERB)
  (transitivity and verbal vs nominal predicate were also checked, but every
  sentence in the filtered set is a transitive verbal clause — the MTP-I
  filtering conditions require subject + object — so they are constants and
  are reported only in the descriptive table.)

Frequency control: mean wordfreq Zipf ('hi') over Devanagari tokens, as in
script 24. The 1 Sep analysis used Shabd ZIPF norms; pass --shabd FILE
(CSV/TSV with a word column and a zipf column) to use those instead.

Input : data/processed/replication_filtered_sentences.pkl  (gold HDTB trees)
        data/results/MTP2_raw_perword.csv                  (script 20)
        data/results/MTP2_master_sheet.csv                 (sentence text)
        data/processed/Hindi-NRC-VAD-Lexicon.txt
Output: data/results/dominance_robustness_results.csv      (all model rows)
        data/results/dominance_robustness_tertiles.csv
        data/results/dominance_robustness_sentences.csv    (per-sentence features)
        data/results/dominance_robustness_report.txt

Usage : python scripts/26_check_dominance_robustness.py [--shabd FILE] [--boot N]
Needs : pandas, numpy, statsmodels, wordfreq (same as script 24)

Version history
  v1.0  first version (5 Oct 2026)
  v1.1  emotion scoring follows script 20 v2.6: negators/intensifiers are modifiers
        only (not scored as emotion words); negation flips the word it attaches to
        in the gold tree. Re-run after `20_build_hutb_raw_dataset.py --emotion-only`.
"""
VERSION = "v1.3"

import os, re, sys, pickle, argparse, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
from parsers.ud_parser import Sentence, Word  # noqa: F401  (needed to unpickle)

ap = argparse.ArgumentParser()
ap.add_argument("--shabd", default=None, help="Shabd frequency file (word + zipf columns)")
ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (default 1000)")
args = ap.parse_args()

SENTS_IN = ROOT / "data/processed/replication_filtered_sentences.pkl"
PW_IN    = ROOT / "data/results/MTP2_raw_perword.csv"
MS_IN    = ROOT / "data/results/MTP2_master_sheet.csv"
VAD_IN   = ROOT / "data/processed/Hindi-NRC-VAD-Lexicon.txt"
OUT      = ROOT / "data/results"
OUT.mkdir(parents=True, exist_ok=True)

BANNER = f"SCRIPT 34: DOMINANCE ROBUSTNESS + SYNTAX CONTROLS ({VERSION})"
print("=" * 78 + f"\n {BANNER}\n" + "=" * 78)
for p in (SENTS_IN, PW_IN, MS_IN, VAD_IN):
    if not p.exists():
        sys.exit(f"ERROR: missing required file: {p}")

# ---------------------------------------------------------------------------
# Emotion scoring — identical rules to script 20 (load_vad_lexicon/score_emotion)
# ---------------------------------------------------------------------------
NEGATION     = {"नहीं", "न", "मत", "ना"}
INTENSIFIERS = {"बहुत", "अत्यधिक", "ज़्यादा", "ज्यादा", "बेहद", "अत्यंत", "काफी", "इतना"}
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

def load_vad_lexicon(path):
    raw = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        sample = f.readline(); delim = "\t" if "\t" in sample else " "; f.seek(0)
        for line in f:
            parts = [p.strip() for p in line.strip().split(delim) if p.strip()]
            if len(parts) < 5:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
            if parts[0].lower() == "english" or parts[4] == "Hindi":
                continue
            try:
                v = (float(parts[1]) - 0.5) * 2.0
                a = float(parts[2])
                d = (float(parts[3]) - 0.5) * 2.0
                w = parts[4].split()[0]
            except (ValueError, IndexError):
                continue
            raw[w].append((v, a, d))
    return {w: tuple(np.mean([s[k] for s in sc]) for k in range(3)) for w, sc in raw.items()}

def score_emotion(words, lex, keep=None):
    """script 20 v2.6 score_emotion; `keep` restricts which words are scored
    (negation scope still uses the full sentence). Negators / intensifiers are
    modifiers only (never scored themselves); negation flips valence and
    dominance of the word the negator's head points to in the gold tree."""
    forms = [w.form.strip() for w in words]
    pos = {int(w.idx): i for i, w in enumerate(words)}
    flipped = set()
    for i, f in enumerate(forms):
        if f in NEGATION:
            try:
                h = pos.get(int(words[i].head))
            except (TypeError, ValueError):
                h = None
            if h is not None and h != i:
                flipped.add(h)
    vals, aros, doms = [], [], []
    for i, w in enumerate(words):
        form = forms[i]
        if form in NEGATION or form in INTENSIFIERS or form not in lex or (keep is not None and not keep(w)):
            continue
        v, a, d = lex[form]
        if i in flipped:
            v, d = -v, -d
        if i > 0 and forms[i - 1] in INTENSIFIERS:
            a = min(1.0, a * 1.2)
        vals.append(v); aros.append(a); doms.append(d)
    if not vals:
        return 0.0, 0.5, 0.0, 0
    return float(np.mean(vals)), float(np.mean(aros)), float(np.mean(doms)), len(vals)

# ---------------------------------------------------------------------------
# Syntactic features from the gold tree
# ---------------------------------------------------------------------------
CLAUSE_RELS = {"advcl", "acl", "acl:relcl", "ccomp", "xcomp"}

def info_status_ref(sent):
    """script 20 info_status_score on the reference order (GIVEN = PRON)."""
    pre = [w for w in sent.get_preverbal_constituents()]
    if len(pre) < 2:
        return 0
    g1, g2 = pre[0].upos == "PRON", pre[1].upos == "PRON"
    return 1 if (g1 and not g2) else (-1 if (g2 and not g1) else 0)

def syntax_features(sent):
    r = sent.root_idx
    root = next((w for w in sent.words if w.idx == r), None)
    kids = sent.get_children(r) if r is not None else []
    passive = any(k.deprel in ("aux:pass", "nsubj:pass") for k in kids) or \
              (root is not None and "Voice=Pass" in str(root.feats))
    return dict(
        passive=int(passive),
        ergative=int(any(w.form == "ने" and w.deprel == "case" for w in sent.words)),
        info_status=info_status_ref(sent),
        n_preverbal=sum(1 for k in kids if k.idx < (r or 0) and k.upos != "PUNCT"),
        n_clauses=sum(1 for w in sent.words
                      if w.deprel in CLAUSE_RELS or (w.deprel == "conj" and w.upos == "VERB")),
        transitive=int(any(k.deprel in ("obj", "iobj") for k in kids)),
        nonverbal_pred=int(root is None or root.upos != "VERB"),
    )

# ---------------------------------------------------------------------------
# Build the per-sentence table
# ---------------------------------------------------------------------------
print("\n[1] Loading data...")
with open(SENTS_IN, "rb") as f:
    sents = pickle.load(f)
lex = load_vad_lexicon(VAD_IN)
pw = pd.read_csv(PW_IN, encoding="utf-8-sig")
ms = pd.read_csv(MS_IN, encoding="utf-8-sig")
ref = pw[pw.is_reference == 1].merge(ms[["sent_id", "sentence"]], on="sent_id", how="left")
print(f"    {len(sents):,} gold trees | {len(ref):,} reference rows | lexicon {len(lex):,} words")

rows = []
for s in sents:
    _, _, dom_all, _ = score_emotion(s.words, lex)
    v_c, a_c, dom_c, n_c = score_emotion(s.words, lex, keep=lambda w: w.upos in CONTENT_UPOS)
    rows.append(dict(sent_id=s.sent_id, dom_rescored=dom_all, dom_content=dom_c,
                     val_content=v_c, n_content_matched=n_c, **syntax_features(s)))
df = ref.merge(pd.DataFrame(rows), on="sent_id", how="inner")

# frequency control
DEV = re.compile(r"[ऀ-ॿ]+")
if args.shabd:
    sh = pd.read_csv(args.shabd, sep=None, engine="python")
    wcol = next(c for c in sh.columns if c.lower() in ("word", "words", "token", "hindi"))
    zcol = next(c for c in sh.columns if "zipf" in c.lower())
    freq = dict(zip(sh[wcol].astype(str), sh[zcol].astype(float)))
    fz = lambda w: freq.get(w, np.nan)
    FREQ_SRC = f"Shabd ({Path(args.shabd).name})"
else:
    from wordfreq import zipf_frequency
    fz = lambda w: zipf_frequency(w, "hi")
    FREQ_SRC = "wordfreq Zipf ('hi')"
df["zipf"] = df.sentence.map(lambda s: np.nanmean([fz(w) for w in DEV.findall(str(s))] or [np.nan]))

df = df.rename(columns={"tgt_valence": "val", "tgt_arousal": "aro", "tgt_dominance": "dom",
                        "trigram_surprisal_per_word": "trigram", "lstm_surprisal_per_word": "lstm",
                        "dep_len_per_word": "dl", "ref_construction_type": "construction"})
df["log_len"] = np.log(df.n_words)
need = ["dom", "dom_content", "val", "trigram", "lstm", "dl", "log_len", "zipf"]
n0 = len(df)
df = df.dropna(subset=need).reset_index(drop=True)
print(f"    analysis set: {len(df):,} sentences ({n0 - len(df)} dropped for missing values)")
print(f"    frequency control: {FREQ_SRC}")

# ---------------------------------------------------------------------------
# STEP 1a — rescoring sanity check
# ---------------------------------------------------------------------------
diff = (df.dom - df.dom_rescored).abs()
match_rate = (diff < 1e-3).mean()
print(f"\n[1a] Re-scored dominance matches MTP2_raw_perword: {match_rate:.1%} "
      f"(max |diff| {diff.max():.4f})")

# ---------------------------------------------------------------------------
# STEP 1b — tertile table
# ---------------------------------------------------------------------------
def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp

tert_rows = []
for diffm in ["trigram", "lstm", "dl"]:
    t = pd.qcut(df[diffm], 3, labels=["low", "mid", "high"])
    for emo in ["dom", "dom_content", "val", "aro"]:
        g = df.groupby(t, observed=True)[emo]
        lo, hi = df.loc[t == "low", emo], df.loc[t == "high", emo]
        from scipy.stats import ttest_ind
        tt = ttest_ind(hi, lo, equal_var=False)
        tert_rows.append(dict(difficulty=diffm, emotion=emo,
                              mean_low=g.mean()["low"], mean_mid=g.mean()["mid"],
                              mean_high=g.mean()["high"],
                              monotonic_decrease=bool(g.mean()["low"] > g.mean()["mid"] > g.mean()["high"]),
                              d_high_vs_low=cohen_d(hi, lo), t=tt.statistic, p=tt.pvalue,
                              n_per_tertile=int(len(lo))))
tert = pd.DataFrame(tert_rows)
r = tert[(tert.difficulty == "trigram") & (tert.emotion == "dom")].iloc[0]
print(f"[1b] Trigram tertiles, dominance: {r.mean_low:+.3f} -> {r.mean_mid:+.3f} -> "
      f"{r.mean_high:+.3f}   d = {r.d_high_vs_low:+.2f}, p = {r.p:.1e}")
r = tert[(tert.difficulty == "trigram") & (tert.emotion == "dom_content")].iloc[0]
print(f"[1c] Content-word dominance:      {r.mean_low:+.3f} -> {r.mean_mid:+.3f} -> "
      f"{r.mean_high:+.3f}   d = {r.d_high_vs_low:+.2f}, p = {r.p:.1e}")

# ---------------------------------------------------------------------------
# STEP 2 — nested models with syntax controls
# ---------------------------------------------------------------------------
for c in ["dom", "dom_content", "val", "trigram", "lstm", "dl", "log_len", "zipf",
          "info_status", "n_preverbal", "n_clauses"]:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

CONTROLS = {
    "M0_base":       "log_len_z + zipf_z",
    "M1_voice_case": "log_len_z + zipf_z + passive + ergative",
    "M2_order":      "log_len_z + zipf_z + passive + ergative + C(construction) + info_status_z + n_preverbal_z",
    "M3_complexity": "log_len_z + zipf_z + passive + ergative + C(construction) + info_status_z + n_preverbal_z"
                     " + n_clauses_z",
}
CONTROLS["M4_plus_valence"] = CONTROLS["M3_complexity"] + " + val_z"

print(f"\n[2] Nested models (emotion ~ difficulty + controls), bootstrap {args.boot} on M3...")
rng = np.random.default_rng(34)
boot_idx = [rng.integers(0, len(df), len(df)) for _ in range(args.boot)]
res = []
for emo in ["dom", "dom_content"]:
    for diffm in ["trigram", "lstm", "dl"]:
        term = f"{diffm}_z"
        base_beta = None
        for name, ctrl in CONTROLS.items():
            form = f"{emo}_z ~ {term} + {ctrl}"
            m = smf.ols(form, df).fit(cov_type="HC3")
            b = m.params[term]
            if name == "M0_base":
                base_beta = b
            row = dict(emotion=emo, difficulty=diffm, model=name, beta=b, se=m.bse[term],
                       t=m.tvalues[term], p=m.pvalues[term],
                       change_vs_M0=b / base_beta - 1, r2=m.rsquared, n=int(m.nobs),
                       boot_lo=np.nan, boot_hi=np.nan, boot_sign_consistency=np.nan)
            if name == "M3_complexity" and args.boot > 0:
                X, y = m.model.exog, m.model.endog          # reuse the design matrix
                j = m.model.exog_names.index(term)
                bs = np.array([np.linalg.lstsq(X[ix], y[ix], rcond=None)[0][j] for ix in boot_idx])
                row.update(boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5),
                           boot_sign_consistency=max((bs > 0).mean(), (bs < 0).mean()))
            res.append(row)
        last = [x for x in res if x["emotion"] == emo and x["difficulty"] == diffm]
        print(f"    {emo:<12s} {diffm:<8s} " +
              "  ".join(f"{x['model'][:2]} {x['beta']:+.3f}{'*' if x['p'] < .05 else ' '}" for x in last))
res = pd.DataFrame(res)

# What the controls themselves do: syntax -> dominance and syntax -> difficulty
desc = []
for feat in ["passive", "ergative", "transitive", "nonverbal_pred"]:
    for lvl in (0, 1):
        s = df[df[feat] == lvl]
        desc.append(dict(feature=feat, level=lvl, n=len(s), dom=s.dom.mean(),
                         dom_content=s.dom_content.mean(), trigram=s.trigram.mean(), dl=s.dl.mean()))
for lvl, s in df.groupby("construction"):
    desc.append(dict(feature="construction", level=lvl, n=len(s), dom=s.dom.mean(),
                     dom_content=s.dom_content.mean(), trigram=s.trigram.mean(), dl=s.dl.mean()))
desc = pd.DataFrame(desc)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
res.to_csv(OUT / "dominance_robustness_results.csv", index=False)
tert.to_csv(OUT / "dominance_robustness_tertiles.csv", index=False)
keep = ["sent_id", "sentence", "n_words", "val", "aro", "dom", "dom_content", "n_content_matched",
        "trigram", "lstm", "dl", "zipf", "construction", "passive", "ergative", "info_status",
        "n_preverbal", "n_clauses", "transitive", "nonverbal_pred"]
df[keep].to_csv(OUT / "dominance_robustness_sentences.csv", index=False, encoding="utf-8-sig")

def verdict(sub):
    m0 = sub[sub.model == "M0_base"].iloc[0]; m3 = sub[sub.model == "M3_complexity"].iloc[0]
    if m3.p < .05 and np.sign(m3.beta) == np.sign(m0.beta):
        return f"SURVIVES (M0 {m0.beta:+.3f} -> M3 {m3.beta:+.3f}, {m3.change_vs_M0:+.0%} vs M0)"
    return f"EXPLAINED BY SYNTAX (M0 {m0.beta:+.3f} -> M3 {m3.beta:+.3f}, p = {m3.p:.2f})"

L = [BANNER, "=" * len(BANNER), "",
     f"Sentences: {len(df):,} HUTB references | frequency control: {FREQ_SRC}",
     f"Re-scored dominance matches MTP2_raw_perword.csv for {match_rate:.1%} of sentences", "",
     "STEP 1 — tertiles of difficulty (mean emotion low / mid / high; d = high vs low)"]
for _, x in tert.iterrows():
    L.append(f"  {x.difficulty:<8s} {x.emotion:<12s} {x.mean_low:+.3f} {x.mean_mid:+.3f} {x.mean_high:+.3f}"
             f"   d = {x.d_high_vs_low:+.2f}  p = {x.p:.1e}{'  monotonic' if x.monotonic_decrease else ''}")
L += ["", "STEP 2 — beta of difficulty on dominance (z units), nested controls; * p < .05"]
for (emo, diffm), sub in res.groupby(["emotion", "difficulty"], sort=False):
    L.append(f"  {emo:<12s} {diffm:<8s} " + "  ".join(
        f"{x.model.split('_')[0]} {x.beta:+.3f}{'*' if x.p < .05 else ' '}" for _, x in sub.iterrows()))
    m3 = sub[sub.model == "M3_complexity"].iloc[0]
    if not np.isnan(m3.boot_lo):
        L.append(f"  {'':<21s} M3 bootstrap 95% CI [{m3.boot_lo:+.3f}, {m3.boot_hi:+.3f}], "
                 f"sign consistency {m3.boot_sign_consistency:.1%}")
    L.append(f"  {'':<21s} -> {verdict(sub)}")
L += ["", "Syntactic features: mean dominance and difficulty by level"]
for _, x in desc.iterrows():
    L.append(f"  {x.feature:<15s} {str(x.level):<8s} n={x.n:<5d} dom {x.dom:+.3f}  dom_content "
             f"{x.dom_content:+.3f}  trigram {x.trigram:6.2f}  DL {x.dl:5.2f}")
L += ["", "Models: M0 length + frequency | M1 + passive + ergative | M2 + construction + "
      "info status + n preverbal | M3 + n clauses | "
      "M4 + valence", "Emotion is a text-content proxy (NRC-VAD), not reader mood."]
(OUT / "dominance_robustness_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

print(f"\nWrote {OUT}/dominance_robustness_{{results,tertiles,sentences}}.csv and _report.txt")
print(f"Done — {BANNER}")
