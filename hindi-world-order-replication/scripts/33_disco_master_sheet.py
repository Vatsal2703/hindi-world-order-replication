#!/usr/bin/env python3
"""
Script 33: DISCO Hindi master sheet (Excel) — v1.1

Step 3 of 3 in the DISCO pipeline (31 -> 32 -> 33; run all with
scripts/run_disco_pipeline.sh). Takes a few seconds.

One row per DISCO Hindi utterance, laid out like MTP2_master_sheet.xlsx:
emotion (valence / arousal / dominance), trigram surprisal and dependency
length for the fluent sentence, plus the disfluency annotation.
A second sheet ("correlations") relates surprisal and DL to V/A/D with live
CORREL formulas, plus the controlled regression betas from script 32.

Input : data/results/disco_affect_scored.csv   (script 31: emotion, surprisal)
        data/disco/disco_dl.csv                (script 32: dependency length)
        data/results/difficulty_params_results.csv (script 32: betas)
Output: data/results/DISCO_master_sheet.xlsx

Sheet "Sheet1"       : one row per utterance (3,171 rows) — ids, disfluent and
                       fluent text, translation, disfluency type, domain,
                       n_words, valence, arousal, dominance, coverage,
                       n_matched, dep_len(+per word), trigram_surprisal(+per
                       word), disfluent flag.
Sheet "correlations" : raw Pearson correlations (live CORREL formulas) of
                       surprisal and DL with V/A/D; surprisal-DL correlation;
                       controlled betas (DISCO, spec "joint") from script 32.

Version history
  v1.0  first version
  v1.1  1 Oct clean-up: header/notes updated (pipeline step numbering; the
        notes row no longer pins a script-32 version). Output unchanged.
"""
VERSION = "v1.1"

import sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/results"
print(f"=== Script 33: DISCO master sheet {VERSION} ===")

d = pd.read_csv(ROOT / "data/results/disco_affect_scored.csv")
dl = pd.read_csv(ROOT / "data/disco/disco_dl.csv")
d = d.merge(dl[["Sentence Number", "n_tokens", "dep_len", "dl"]], on="Sentence Number", how="left")
sheet = pd.DataFrame({
    "sent_id": d["Sentence Number"].map(lambda i: f"disco-hi-{int(i):04d}"),
    "disfluent_sentence": d.disfl,
    "fluent_sentence": d.fluent,
    "english_translation": d["English Translation of Fluent Sentence"],
    "disfluency_type": d.dtype,
    "domain": d.domain,
    "n_words": d.n_fluent,
    "valence": d.val.round(4),
    "arousal": d.aro.round(4),
    "dominance": d.dom.round(4),
    "coverage": d.coverage.round(4),
    "n_matched": (d.coverage * d.n_fluent).round().astype(int),
    "dep_len": d.dep_len.astype(int),
    "dep_len_per_word": d.dl.round(4),
    "trigram_surprisal": (d.surprisal * d.n_fluent).round(4),
    "trigram_surprisal_per_word": d.surprisal.round(4),
    "disfluent": d.disfluent.astype(int),
})
n = len(sheet)
print(f"rows: {n}")

wb = Workbook()
ws = wb.active
ws.title = "Sheet1"
ws.append(list(sheet.columns))
for row in sheet.itertuples(index=False):
    ws.append(list(row))
bold = Font(name="Arial", bold=True)
for c in ws[1]:
    c.font = bold
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.font = Font(name="Arial")
widths = {"A": 15, "B": 45, "C": 45, "D": 40, "E": 9, "F": 20}
for col in range(1, ws.max_column + 1):
    L = get_column_letter(col)
    ws.column_dimensions[L].width = widths.get(L, 13)
ws.freeze_panes = "B2"

# ---------------- correlations sheet ---------------------------------------
cs = wb.create_sheet("correlations")
col = {name: get_column_letter(i + 1) for i, name in enumerate(sheet.columns)}
rng = lambda name: f"Sheet1!${col[name]}$2:${col[name]}${n + 1}"
cs["A1"] = f"DISCO Hindi: difficulty parameters vs emotion (N = {n}) — script 33 {VERSION}"
cs["A1"].font = Font(name="Arial", bold=True, size=12)
cs["A3"] = "Pearson correlation (live formulas on Sheet1)"
cs["A3"].font = bold
cs.append(["", "valence", "arousal", "dominance"])
for c in cs[4]: c.font = bold
for r, par in enumerate(["trigram_surprisal_per_word", "dep_len_per_word"], start=5):
    cs.cell(r, 1, par).font = Font(name="Arial")
    for j, emo in enumerate(["valence", "arousal", "dominance"], start=2):
        cell = cs.cell(r, j, f"=CORREL({rng(par)},{rng(emo)})")
        cell.number_format = "0.000"; cell.font = Font(name="Arial")
cs["A7"] = "Correlation between the two parameters"
cs["B7"] = f"=CORREL({rng('trigram_surprisal_per_word')},{rng('dep_len_per_word')})"
cs["B7"].number_format = "0.000"

res = pd.read_csv(ROOT / "data/results/difficulty_params_results.csv")
res = res[(res.corpus == "DISCO") & (res.spec == "joint")]
cs["A10"] = ("Controlled regression (script 32): emotion_z ~ surprisal_z + DL_z + log length + "
             "word frequency + task domain; standardised beta, HC3 p-value")
cs["A10"].font = bold
cs.append(["emotion", "surprisal beta", "surprisal p", "DL beta", "DL p"])
for c in cs[11]: c.font = bold
emo_name = {"val": "valence", "aro": "arousal", "dom": "dominance"}
for emo in ["val", "aro", "dom"]:
    s = res[(res.emotion == emo) & (res.parameter == "surprisal")].iloc[0]
    l = res[(res.emotion == emo) & (res.parameter == "dl")].iloc[0]
    cs.append([emo_name[emo], round(s.beta, 4), round(s.p, 4), round(l.beta, 4), round(l.p, 4)])
cs["A16"] = ("Values in rows 12-14 are copied from data/results/difficulty_params_results.csv "
             "(script 32); correlations in rows 5-7 are uncontrolled.")
cs["A17"] = ("Surprisal: EMILLE trigram model, per word. DL: sum |head - dependent| per word, "
             "from a spaCy parser trained on UD Hindi-HDTB (test UAS 91.5).")
cs["A18"] = ("Emotion: NRC-VAD v2.1 Hindi on the fluent sentence (script 26 rules). "
             f"9 of 3,180 utterances dropped (no Devanagari words for frequency).")
for r in (16, 17, 18): cs[f"A{r}"].font = Font(name="Arial", italic=True, size=9)
cs.column_dimensions["A"].width = 32
for L in "BCDE": cs.column_dimensions[L].width = 15

OUT.mkdir(parents=True, exist_ok=True)
wb.save(OUT / "DISCO_master_sheet.xlsx")
print(f"saved {OUT / 'DISCO_master_sheet.xlsx'}")
print(f"=== done {VERSION} ===")
