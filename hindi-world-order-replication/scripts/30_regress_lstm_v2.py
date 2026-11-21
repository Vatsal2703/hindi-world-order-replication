#!/usr/bin/env python
# v1.5 (2026-11-02): renumbered from 39_regress_lstm_v2.py; naming only, no logic change.
# v1.4 (2026-11-02): renamed from 39_lstm_v2_regressions.py; naming only, no logic change.
"""
30_regress_lstm_v2.py  (v1.5)

Redoes the surprisal -> emotion analysis with the RETRAINED LSTM (script 28/29)
in BOTH corpora, next to the trigram measure, so the surprisal comparison is
finally like-for-like.

For each corpus (HUTB references, DISCO fluent utterances) and each emotion
(valence, arousal, dominance), z-scored OLS with HC3 standard errors:

  M_tri : emotion ~ trigram surprisal + DL + controls
  M_lstm: emotion ~ LSTM-v2 surprisal + DL + controls
  M_both: emotion ~ trigram + LSTM-v2 + DL + controls

Controls: HUTB  = log length + EMILLE log frequency (as script 27)
          DISCO = log length + mean Zipf frequency (wordfreq) + task domain

INPUTS
  data/results/lstm_v2_scores.csv            (script 29)
  data/results/MTP2_refs_with_emille_freq.csv (HUTB)
  data/results/disco_affect_scored.csv        (DISCO emotion + trigram, script 23)
  data/disco/disco_dl.csv                     (DISCO parse, DL)
  data/disco/Hindi.xlsx                       (DISCO task domain)
OUTPUT
  data/results/lstm_v2_regressions.csv

RUN:  python scripts/30_regress_lstm_v2.py
      python scripts/30_regress_lstm_v2.py --kn
          v1.2: --kn also scores both corpora with the SMOOTHED Kneser-Ney
          trigram from MTP-I (data/models/trigram_model_blind.pkl, script 07)
          and fits M_kn: emotion ~ KN trigram + DL + controls. Tests whether
          the sign flips of the unsmoothed trigram come from unseen words.
          Extra outputs: data/results/kn_trigram_scores.csv,
                         data/results/kn_trigram_regressions.csv
          (lstm_v2_regressions.csv is not changed by --kn).
      python scripts/30_regress_lstm_v2.py --disfl
          v1.3: DISCO only. Does the emotion of the fluent version differ for
          utterances the speaker produced disfluently? Disfluency measures from
          script 23: disfluent (0/1), deletion (share of tokens removed),
          repair (correction or false start, 0/1). One measure per model:
          emotion ~ measure + LSTM + DL + log length + Zipf + domain.
          CAVEAT: PRESTO/DISCO disfluencies were elicited from contributors
          following instructions targeting the phenomenon, not recorded from
          spontaneous speech. Output: data/results/disco_disfluency_regressions.csv
"""
import os, re, sys, argparse
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

VERSION = "1.5"
print(f"=== 30_regress_lstm_v2.py v{VERSION} ===")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--kn", action="store_true", help="v1.2: add the Kneser-Ney trigram analysis")
ap.add_argument("--disfl", action="store_true", help="v1.3: DISCO disfluency vs emotion")
ARGS = ap.parse_args()
BASE = ARGS.base
R = lambda *p: os.path.join(BASE, *p)

def z(s):
    return (s - s.mean()) / s.std()

def find(cols, pattern, exclude=None):
    c = [x for x in cols if re.search(pattern, x, re.I) and not (exclude and re.search(exclude, x, re.I))]
    c.sort(key=len)
    return c[0] if c else None

scores = pd.read_csv(R("data/results/lstm_v2_scores.csv"), encoding="utf-8-sig")

# ---------------------------------------------------------------- HUTB ----
h = pd.read_csv(R("data/results/MTP2_refs_with_emille_freq.csv"), encoding="utf-8-sig")
h = h[h.is_reference == 1].copy()
hs = scores[scores.corpus == "HUTB"][["key", "lstm_v2_pw"]].rename(columns={"key": "sent_id"})
h = h.merge(hs, on="sent_id", how="inner")
H = pd.DataFrame({
    "valence": h.tgt_valence, "arousal": h.tgt_arousal, "dominance": h.tgt_dominance,
    "tri": h.trigram_surprisal / h.n_words, "lstm": h.lstm_v2_pw,
    "dl": h.dep_len / h.n_words, "loglen": np.log(h.n_words), "freq": h.em_logf,
}).dropna()
print(f"HUTB rows: {len(H):,}")

# --------------------------------------------------------------- DISCO ----
sc = pd.read_csv(R("data/results/disco_affect_scored.csv"), encoding="utf-8-sig")
print("DISCO scored columns:", list(sc.columns))
key = find(sc.columns, r"^sentence.?number$|^sent.*(id|num)|^utt|^id$") or sc.columns[0]
cv = "val" if "val" in sc.columns else find(sc.columns, r"valence", r"prev|content")
ca = "aro" if "aro" in sc.columns else find(sc.columns, r"arousal", r"prev|content")
cd = "dom" if "dom" in sc.columns else find(sc.columns, r"dominance", r"prev|content")
ct = find(sc.columns, r"^surprisal$|trigram.*surp|^surp")
print(f"DISCO key={key}  valence={cv} arousal={ca} dominance={cd} trigram={ct}")
if not (cv and ca and cd):
    sys.exit("ERROR: could not find the DISCO emotion columns -- paste the column list above to Claude.")

dl = pd.read_csv(R("data/disco/disco_dl.csv"), encoding="utf-8-sig")[["Sentence Number", "dl", "parse"]]
dl = dl.rename(columns={"dl": "dl_pw", "parse": "parse_s"})
dom = pd.read_excel(R("data/disco/Hindi.xlsx"))[["Sentence Number", "Domain Label"]].drop_duplicates("Sentence Number")
dom = dom.rename(columns={"Domain Label": "domain_lbl"})
ds = scores[scores.corpus == "DISCO"][["key", "lstm_v2_pw", "n_tokens"]].rename(
    columns={"key": "Sentence Number", "n_tokens": "n_tok_v2"})

sc = sc.rename(columns={key: "Sentence Number"})
for t in (sc, ds, dl, dom):
    t["Sentence Number"] = pd.to_numeric(t["Sentence Number"], errors="coerce").astype("Int64")
D = sc.merge(ds, on="Sentence Number").merge(dl, on="Sentence Number").merge(dom, on="Sentence Number", how="left")

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None
PAT = re.compile(r"^(\d+):(.*)->(\d+):(.+)$")
def mean_zipf(parse):
    toks = [m.group(2) for m in (PAT.match(p) for p in str(parse).split(" ")) if m]
    return np.mean([zipf_frequency(t, "hi") for t in toks]) if toks else np.nan
D["freq"] = D["zipf"] if "zipf" in D.columns else D.parse_s.map(mean_zipf)

DD = pd.DataFrame({
    "valence": D[cv], "arousal": D[ca], "dominance": D[cd],
    "tri": D[ct] if ct else np.nan, "lstm": D.lstm_v2_pw, "dl": D.dl_pw,
    "loglen": np.log(D.n_tok_v2), "freq": D.freq, "domain": (D["domain"] if "domain" in D.columns else D.domain_lbl).fillna("NA").astype(str),
}).dropna(subset=["valence", "arousal", "dominance", "lstm", "dl", "loglen", "freq"])
print(f"DISCO rows: {len(DD):,}")



# ------------------------------------------- v1.3: DISCO disfluency ----
if ARGS.disfl:
    X = DD.copy()
    for c in ["disfluent", "deletion", "repair"]:
        X[c] = D.loc[X.index, c].values
    print(f"\nDISCO disfluency: disfluent {X.disfluent.mean():.2%} of utterances, mean deletion {X.deletion.mean():.3f}, repair {X.repair.mean():.2%}")
    for c in ["valence", "arousal", "dominance", "lstm", "dl", "loglen", "freq", "deletion"]:
        X[c + "_z"] = z(X[c])
    rr = []
    for emo in ["valence", "arousal", "dominance"]:
        for meas, term in (("disfluent", "disfluent"), ("deletion", "deletion_z"), ("repair", "repair")):
            m = smf.ols(f"{emo}_z ~ {term} + lstm_z + dl_z + loglen_z + freq_z + C(domain)", X).fit(cov_type="HC3")
            lo, hi = m.conf_int().loc[term]
            rr.append(("DISCO", emo, meas, len(X), round(m.params[term], 4), round(lo, 4), round(hi, 4), round(m.pvalues[term], 4)))
    dfr = pd.DataFrame(rr, columns=["corpus", "emotion", "measure", "n", "beta", "ci_lo", "ci_hi", "p"])
    dfr.to_csv(R("data/results/disco_disfluency_regressions.csv"), index=False)
    print("disfluent/repair: difference in SD units (0/1 predictor); deletion: standardised beta")
    print(dfr.to_string(index=False))
    print("wrote", R("data/results/disco_disfluency_regressions.csv"))
    print(f"DONE v{VERSION}")
    sys.exit(0)

# ------------------------------------------------- v1.2: KN trigram ----
def kn_prob(w1, w2, w3, m):
    """Interpolated Kneser-Ney trigram probability, identical to scripts/09_score_trigram.py."""
    D_ = m["D"]
    uni_cont = m["unigram_continuation_count"].get(w3, 0)
    tb = m["total_bigram_types"]
    p_uni = uni_cont / tb if tb > 0 else 1e-10
    bc, bct = m["bigram_continuation_count"].get((w2, w3), 0), m["bigram_cont_total"].get(w2, 0)
    if bct > 0:
        p_bi = max(bc - D_, 0) / bct + D_ * m["bigram_unique_followers"].get(w2, 0) / bct * p_uni
    else:
        p_bi = p_uni
    tc, bi = m["trigrams"].get((w1, w2, w3), 0), m["bigrams"].get((w1, w2), 0)
    if bi > 0:
        p_tri = max(tc - D_, 0) / bi + D_ * m["bigram_followers_count"].get((w1, w2), 0) / bi * p_bi
    else:
        p_tri = p_bi
    return max(p_tri, 1e-10)

def kn_pw(tokens, m):
    """Per-token KN surprisal in bits; sentence-boundary tokens as in script 09, divided by word count."""
    t = ["<s>", "<s>"] + list(tokens) + ["</s>"]
    bits = sum(-np.log2(kn_prob(t[i - 2], t[i - 1], t[i], m)) for i in range(2, len(t)))
    return bits / max(len(tokens), 1)

if ARGS.kn:
    import pickle, time
    t0 = time.time()
    KN = pickle.load(open(R("data/models/trigram_model_blind.pkl"), "rb"))
    print(f"KN model loaded in {time.time() - t0:.0f}s (D = {KN['D']})")
    h["kn"] = h["sentence"].astype(str).str.split().map(lambda tk: kn_pw(tk, KN))
    H["kn"] = h.loc[H.index, "kn"].values if len(h) == len(H) else h["kn"].reindex(H.index).values
    D["kn"] = D.parse_s.map(lambda ps: kn_pw([m_.group(2) for m_ in (PAT.match(x) for x in str(ps).split(" ")) if m_], KN))
    DD["kn"] = D.loc[DD.index, "kn"].values
    pd.concat([pd.DataFrame({"corpus": "HUTB", "key": h.sent_id, "kn_pw": h.kn}),
               pd.DataFrame({"corpus": "DISCO", "key": D["Sentence Number"], "kn_pw": D.kn})]).to_csv(
        R("data/results/kn_trigram_scores.csv"), index=False, encoding="utf-8-sig")
    print(f"KN scoring done in {time.time() - t0:.0f}s")
    for nm, df in (("HUTB", H), ("DISCO", DD)):
        print(f"  {nm}: mean bits/token  unsmoothed {df.tri.mean():.2f}  KN {df.kn.mean():.2f}  LSTM {df.lstm.mean():.2f};"
              f"  r(unsmoothed, KN) = {df.tri.corr(df.kn):+.2f}  r(KN, LSTM) = {df.kn.corr(df.lstm):+.2f}  r(unsmoothed, LSTM) = {df.tri.corr(df.lstm):+.2f}")

# ------------------------------------------------------------- models ----
rows = []
def run(corpus, df):
    df = df.copy()
    zc = ["valence", "arousal", "dominance", "lstm", "dl", "loglen", "freq"] + (["tri"] if df.tri.notna().all() else []) + (["kn"] if "kn" in df else [])
    for c in zc:
        df[c + "_z"] = z(df[c])
    ctrl = "loglen_z + freq_z" + (" + C(domain)" if "domain" in df else "")
    specs = {"LSTMv2": "lstm_z + dl_z", }
    if "tri_z" in df:
        specs["trigram"] = "tri_z + dl_z"
        specs["both"] = "tri_z + lstm_z + dl_z"
    if "kn_z" in df:
        specs["trigramKN"] = "kn_z + dl_z"
        specs["KN+LSTM"] = "kn_z + lstm_z + dl_z"
    for emo in ["valence", "arousal", "dominance"]:
        for name, rhs in specs.items():
            m = smf.ols(f"{emo}_z ~ {rhs} + {ctrl}", df).fit(cov_type="HC3")
            for term in [t for t in ["tri_z", "kn_z", "lstm_z", "dl_z"] if t in rhs.split(" + ")]:
                rows.append((corpus, emo, name, term.replace("_z", ""), len(df),
                             round(m.params[term], 4), round(m.pvalues[term], 4)))

run("HUTB", H)
run("DISCO", DD)
out = pd.DataFrame(rows, columns=["corpus", "emotion", "model", "predictor", "n", "beta", "p"])
if ARGS.kn:   # v1.2: keep the original table unchanged; KN models go to their own file
    out[out.model.isin(["trigramKN", "KN+LSTM"])].to_csv(R("data/results/kn_trigram_regressions.csv"), index=False)
    out[~out.model.isin(["trigramKN", "KN+LSTM"])].to_csv(R("data/results/lstm_v2_regressions_check.csv"), index=False)
else:
    out.to_csv(R("data/results/lstm_v2_regressions.csv"), index=False)

pd.set_option("display.width", 160)
for corpus in ["HUTB", "DISCO"]:
    print(f"\n===== {corpus}  (standardised beta, p) =====")
    t = out[out.corpus == corpus].copy()
    t["cell"] = t.beta.map("{:+.3f}".format) + " (p=" + t.p.map("{:.3f}".format) + ")"
    print(t.pivot_table(index=["emotion", "predictor"], columns="model", values="cell", aggfunc="first").to_string())
print("\nwrote", R("data/results/kn_trigram_regressions.csv") if ARGS.kn else R("data/results/lstm_v2_regressions.csv"))
print(f"DONE v{VERSION}")
