#!/usr/bin/env python3
# v1.5 (2026-11-02): renumbered from 32_regress_difficulty_emotion.py; naming only, no logic change.
# v1.4 (2026-11-02): renamed from 32_difficulty_parameters_both_corpora.py; naming only, no logic change.
"""
Script 24: Surprisal and dependency length vs valence / arousal / dominance —
           HUTB and DISCO side by side — v1.5

Step 2 of 3 in the DISCO pipeline (23 -> 24 -> 25; run all with
scripts/run_disco_pipeline.sh).

Model, per corpus and per emotion dimension E in {valence, arousal, dominance}:
    E_z ~ surprisal_z + DL_z + log_len_z + zipf_z  [+ C(domain) for DISCO]
  spec "joint"          : both parameters together (each holding the other fixed)
  spec "surprisal_only" : surprisal alone
  spec "dl_only"        : DL alone
All variables z-scored; OLS with HC3 SEs + 1,000 bootstrap resamples.
Script 25 copies the DISCO "joint" rows into the master sheet.

Difficulty parameters (both per word, same definitions as script 20):
  surprisal : trigram surprisal / word, EMILLE trigram model, trigram ->
              bigram -> unigram backoff (script 20 v2.5).
  DL        : sum over words of |position(head) - position(dependent)|,
              divided by number of tokens (punctuation included, as in
              script 20 / MTP2_raw_perword.csv).
    HUTB  : gold UD Hindi-HDTB trees (already in MTP2_raw_perword.csv).
    DISCO : read from data/disco/disco_dl.csv — fluent sentences parsed once
            (30 Sep 2026) with a spaCy parser trained on UD Hindi-HDTB train
            (HDTB test UAS 91.5 / LAS 86.8). The parser itself is no longer
            kept in the repo; disco_dl.csv is the cached result. To re-parse,
            train a spaCy tagger+parser on UD_Hindi-HDTB and pass
            --parser MODEL_DIR after deleting disco_dl.csv.
            DISCO is spoken-style commands with many English loanwords, so
            its parses are out-of-domain and noisier than HUTB's gold trees.

Emotion: NRC-VAD v2.1 Hindi, script 20 rules (rescaling, negation,
intensifier). HUTB values from MTP2_raw_perword.csv; DISCO values from
disco_affect_scored.csv (script 23).

Input : data/results/MTP2_raw_perword.csv, data/results/MTP2_master_sheet.csv
        data/results/disco_affect_scored.csv          (script 23 output)
        data/disco/disco_dl.csv                       (cached DISCO parses + DL)
Output: data/results/difficulty_params_results.csv    (read by script 25)
        data/results/raw_correlations.csv             (uncontrolled Pearson r, both corpora)

Usage : python scripts/24_regress_difficulty_emotion.py [ROOT] [OUT] [--parser MODEL_DIR] [--raw-only]

Version history
  v1.0  models + tertile table + text report; parser shipped in repo
  v1.1  1 Oct clean-up: only difficulty_params_results.csv is written
        (difficulty_params_tertiles.csv and difficulty_params_report.txt
        removed); parser no longer in repo, disco_dl.csv is the cached
        parse. Model results are unchanged.
  v1.2  also writes data/results/raw_correlations.csv: uncontrolled Pearson r (and p) of
        surprisal and DL with valence / arousal / dominance, plus surprisal~DL, for
        HUTB and DISCO (same definition as the live CORREL formulas in script 25's
        sheet). Regression results are unchanged.
  v1.3  --raw-only flag: write raw_correlations.csv and stop (seconds, no models/bootstrap).
"""
VERSION = "v1.5"

import re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from wordfreq import zipf_frequency

warnings.filterwarnings("ignore")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PARSER = sys.argv[sys.argv.index("--parser") + 1] if "--parser" in sys.argv else None
if PARSER in args: args.remove(PARSER)
ROOT = Path(args[0]) if args else Path(".")
OUT = Path(args[1]) if len(args) > 1 else ROOT / "data/results"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 1000
print(f"=== Script 24: surprisal + DL vs valence/arousal/dominance, HUTB & DISCO {VERSION} ===")

DEV = re.compile(r"[ऀ-ॿ]+")
PUNCT = re.compile(r"([।?!,.|])")
def tokenize(s):  # identical to script 23
    return PUNCT.sub(r" \1 ", str(s)).split()
def mean_zipf(s):
    t = DEV.findall(str(s))
    return np.mean([zipf_frequency(w, "hi") for w in t]) if t else np.nan

# ---------------- HUTB ------------------------------------------------------
pw = pd.read_csv(ROOT / "data/results/MTP2_raw_perword.csv", encoding="utf-8-sig")
ms = pd.read_csv(ROOT / "data/results/MTP2_master_sheet.csv", encoding="utf-8-sig")
h = pw[pw.is_reference == 1].merge(ms[["sent_id", "sentence"]], on="sent_id", how="left")
h = pd.DataFrame({"val": h.tgt_valence, "aro": h.tgt_arousal, "dom": h.tgt_dominance,
                  "surprisal": h.trigram_surprisal_per_word, "dl": h.dep_len_per_word,
                  "log_len": np.log(h.n_words), "zipf": h.sentence.map(mean_zipf)}).dropna()
print(f"HUTB : {len(h)} reference sentences")

# ---------------- DISCO -----------------------------------------------------
d = pd.read_csv(ROOT / "data/results/disco_affect_scored.csv")
dl_path = ROOT / "data/disco/disco_dl.csv"
if not dl_path.exists():
    if PARSER is None:
        sys.exit(f"ERROR: {dl_path} missing — restore it, or re-parse with --parser <spaCy model dir> (see header)")
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load(PARSER)
    rows = []
    for i, s in zip(d["Sentence Number"], d.fluent):
        toks = tokenize(s)
        doc = nlp(Doc(nlp.vocab, words=toks))
        total = sum(abs(t.i - t.head.i) for t in doc if t.head.i != t.i)
        rows.append((i, len(toks), total, total / max(len(toks), 1),
                     " ".join(f"{t.i+1}:{t.text}->{t.head.i+1 if t.head.i!=t.i else 0}:{t.dep_}" for t in doc)))
    dl_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Sentence Number", "n_tokens", "dep_len", "dl", "parse"]).to_csv(dl_path, index=False)
    print(f"parsed {len(rows)} DISCO fluent sentences -> {dl_path}")
dl = pd.read_csv(dl_path)
d = d.merge(dl[["Sentence Number", "dl"]], on="Sentence Number", how="left")
d = d[["val", "aro", "dom", "surprisal", "dl", "log_len", "zipf", "domain"]].dropna()
print(f"DISCO: {len(d)} utterances")

# ---------------- raw (uncontrolled) correlations ---------------------------
from scipy.stats import pearsonr
raw_rows = []
for corpus, df in [("HUTB", h), ("DISCO", d)]:
    for par in ["surprisal", "dl"]:
        for emo in ["val", "aro", "dom"]:
            r_, p_ = pearsonr(df[par], df[emo])
            raw_rows.append(dict(corpus=corpus, parameter=par, emotion=emo, pearson_r=r_, p=p_, n=len(df)))
    r_, p_ = pearsonr(df["surprisal"], df["dl"])
    raw_rows.append(dict(corpus=corpus, parameter="surprisal", emotion="dl", pearson_r=r_, p=p_, n=len(df)))
raw = pd.DataFrame(raw_rows)
raw.round(5).to_csv(OUT / "raw_correlations.csv", index=False)
print(f"saved {OUT / 'raw_correlations.csv'}")
print("\nRaw Pearson r (no controls):")
for corpus in ["HUTB", "DISCO"]:
    t = raw[(raw.corpus == corpus) & (raw.emotion != "dl")].pivot(index="parameter", columns="emotion", values="pearson_r")
    print(f"  {corpus} (n = {int(raw[raw.corpus == corpus].n.iloc[0])})")
    print(t[["val", "aro", "dom"]].round(3).to_string())
    print(f"    surprisal ~ DL: {raw[(raw.corpus == corpus) & (raw.emotion == 'dl')].pearson_r.iloc[0]:+.3f}")
if "--raw-only" in sys.argv:
    print(f"=== done {VERSION} (--raw-only: models skipped) ===")
    sys.exit(0)

# ---------------- models ----------------------------------------------------
def z(df, cols):
    df = df.copy()
    for c in cols: df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    return df
COLS = ["val", "aro", "dom", "surprisal", "dl", "log_len", "zipf"]
SPECS = {"joint": ["surprisal_z", "dl_z"], "surprisal_only": ["surprisal_z"], "dl_only": ["dl_z"]}

rows = []
for corpus, df, fe in [("HUTB", h, ""), ("DISCO", d, " + C(domain)")]:
    df = z(df, COLS).reset_index(drop=True)
    rng = np.random.default_rng(32)
    idx = [rng.integers(0, len(df), len(df)) for _ in range(N_BOOT)]
    r_sd = df[["surprisal", "dl"]].corr().iloc[0, 1]
    for emo in ["val", "aro", "dom"]:
        for spec, terms in SPECS.items():
            form = f"{emo}_z ~ {' + '.join(terms)} + log_len_z + zipf_z{fe}"
            m = smf.ols(form, df).fit(cov_type="HC3")
            bs = np.array([smf.ols(form, df.iloc[ix]).fit().params[terms].values for ix in idx])
            for j, t in enumerate(terms):
                rows.append(dict(corpus=corpus, emotion=emo, spec=spec, parameter=t.replace("_z", ""),
                                 beta=m.params[t], se=m.bse[t], t=m.tvalues[t], p=m.pvalues[t],
                                 boot_lo=np.percentile(bs[:, j], 2.5), boot_hi=np.percentile(bs[:, j], 97.5),
                                 boot_sign_consistency=max((bs[:, j] > 0).mean(), (bs[:, j] < 0).mean()),
                                 n=int(m.nobs), r_surprisal_dl=r_sd))
    print(f"  {corpus} done (r surprisal~DL = {r_sd:+.3f})")

res = pd.DataFrame(rows)
res.round(5).to_csv(OUT / "difficulty_params_results.csv", index=False)
print(f"saved {OUT / 'difficulty_params_results.csv'}")
pd.set_option("display.width", 220)
print(res[["corpus", "emotion", "spec", "parameter", "beta", "p", "boot_lo", "boot_hi", "boot_sign_consistency"]]
      .round(4).to_string(index=False))
print(f"=== done {VERSION} ===")
