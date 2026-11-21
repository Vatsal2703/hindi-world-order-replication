# hindi-world-order-replication

M.Tech (Data Engineering) thesis, IIT Jodhpur, by Vatsal Agrawal. Supervisor: Prof. Sidharth Ranjan.

README version 2.3 (2 Nov 2026). On 2 Nov 2026 the scripts were renamed to one convention, `NN_verb_object.py`, and renumbered without gaps (01–15 MTP-I, 16–19 first MTP-II direction, 20–32 final MTP-II, 33 report figures); each script's header names its old file. Every script prints its version in its banner or header.

| Part | Question | Main result |
|---|---|---|
| **MTP-I** | Replication of Ranjan, van Schijndel, Agarwal & Rajkumar (2022), *Discourse Context Predictability Effects in Hindi Word Order* (EMNLP). Can cognitive features (surprisal, dependency length, information status) tell the attested Hindi word order from its grammatical reorderings? | Full model 94.62% (paper 95.06%), 10-fold CV, 178,722 pairs |
| **MTP-II** | Does the emotional tone of a Hindi sentence (valence, arousal, dominance) track how hard it is to process? Tested on the Hindi-Urdu Treebank (HUTB, 2,779 sentences) and the DISCO Hindi corpus (3,171 utterances). | Longer dependencies go with lower dominance (β = −0.090 HUTB, −0.087 DISCO). Higher LSTM surprisal goes with higher valence (β = +0.108, +0.094). Trigram-surprisal effects flip sign between corpora, also with Kneser-Ney smoothing. Both main effects survive Holm correction. A third corpus (Hindi PUD, 1,000 sentences) replicates the dominance effect (−0.100) but not the valence effect (+0.029, n.s.). |
| **Toolkit** | Interactive demo of both parts on any typed Hindi sentence | `toolkit/`; see Section 6 |

---

## 1. Repository layout

```
scripts/            numbered pipeline scripts (run in the order of Section 4)
  extra_diag/       diagnostics not needed to reproduce the results (36, 40, diagnose_perword)
  run_disco_pipeline.sh shortcut for the DISCO side (scripts 23 -> 24 -> 25)
src/                shared code: CoNLL-U parser, variant generator, DL / information-status features, emotion helpers
toolkit/            demo app (Streamlit) + its word-order ranker
tools/berkeley-parser/  Berkeley parser jar (PCFG surprisal, script 13)
_archive/           superseded scripts and files moved out on 2 Nov 2026 (git-ignored; list in _archive/MANIFEST.txt)
data/
  disco/            DISCO Hindi (Hindi.xlsx) + cached dependency parses (disco_dl.csv)
  lexicon/          NRC-VAD v2.1 lexicon files
  processed/        parsed sentences, variants, tokenised EMILLE, Hindi NRC-VAD list
  models/           trigram and LSTM language models
  features/         pairwise feature tables (MTP-I)
  results/          all result tables (CSV/XLSX)
data_pud/           Hindi-PUD test treebank (third corpus, script 32)
requirements.txt
```

## 2. Setup

```
conda activate base                 # any Python 3.9+ env
pip install -r requirements.txt
```

* **torch** is needed for the LSTM scripts (08, 10, 11, 37, 38) and for the toolkit. On Apple-silicon Macs the scripts use the `mps` GPU automatically.
* **Java** is needed for the Berkeley PCFG parser (script 13).
* Run every command from the repository root, the folder that contains `scripts/` and `data/`.

## 3. Data

| Resource | Where it goes | Source |
|---|---|---|
| UD Hindi-HDTB treebank (HUTB) | `data/raw/UD_Hindi-HDTB-master/`, or pass the folder to script 01 | https://github.com/UniversalDependencies/UD_Hindi-HDTB |
| Hindi-PUD test set | `data_pud/hi_pud-ud-test.conllu` (included) | https://github.com/UniversalDependencies/UD_Hindi-PUD |
| EMILLE written Hindi corpus (language-model training) | raw text → `data/processed/emille_written_tokenized.pkl` (script 06) | EMILLE corpus (licensed via ELRA; not redistributed here) |
| NRC-VAD Lexicon v2.1 (Hindi) | `data/lexicon/`, `data/processed/Hindi-NRC-VAD-Lexicon.txt` | Saif Mohammad, NRC (research licence) |
| DISCO Hindi | `data/disco/Hindi.xlsx` (included) | https://github.com/vineet2104/DISCO (Bhat, Jyothi & Bhattacharyya, Findings of EMNLP 2023) |

**Large files.** Files above GitHub's size limit are listed in `.gitignore`: the trigram models, the tokenised EMILLE corpus, all-sentence pickles and PCFG scores. They are shared separately:

> **Large-file download:** _[add Google Drive link here]_. Unzip into the repository root so the paths above are restored.

The toolkit and every MTP-II result can be reproduced from the included files plus `trigram_model_emille.pkl` and `emille_lstm_v2.pt`.

## 4. Pipeline

### 4.1 MTP-I: replication of Ranjan et al. (2022)

| Step | Script | Output |
|---|---|---|
| 1 | `01_parse_hutb.py [UD_HDTB_folder]`, `02_prepare_sentences.py` | parsed HUTB sentences (`valid_sentences.pkl`) |
| 2 | `03_filter_paper_conditions.py` | sentences meeting the paper's conditions (`replication_filtered_sentences.pkl`) |
| 3 | `04_generate_variants.py` | permutations of preverbal subtree blocks, at most 99 per sentence: 93,306 variants from 2,779 sentences (`all_variants_final.pkl`) |
| 4 | `05_score_dl_infostatus.py` | dependency length + information status (context = previous sentence) |
| 5 | `06_parse_emille.py`, `07_train_trigram_kn.py` | EMILLE corpus, Kneser-Ney trigram model (HUTB test sentences held out) |
| 6 | `08_train_lstm.py` (original LSTM) or `28_retrain_lstm.py` (retrained LSTM, see 4.3) | LSTM language model |
| 7 | `09_score_trigram.py`, `10_score_lstm.py`, `11_score_adaptive_lstm.py`, `12_score_lexrept.py`, `13_score_pcfg.py` | trigram, LSTM, adaptive-LSTM, lexical-repetition and PCFG surprisal per variant |
| 8 | `14_merge_features.py` | pairwise feature table, reference minus variant (`all_features_final.csv`, 178,722 rows) |
| 9 | `15_classify_word_order.py` | paper Tables 2–4: regressions, 10-fold CV accuracy, McNemar tests, VIF (`table4_results.csv`) |

**Which LSTM.** The reported MTP-I table (94.62%) uses the original LSTM:

```
python scripts/10_score_lstm.py --model data/models/emille_base_lstm.pt
python scripts/14_merge_features.py
python scripts/15_classify_word_order.py
```

Script 10's default is the retrained LSTM (`emille_lstm_v2.pt`). With it, the LSTM-alone row is 86.29% (paper 94.01%) and the full model is 94.66%. Both versions are kept: `final_scored_variants_v2lstm.pkl`, `all_features_final_v2lstm.csv` and `table4_results_v2lstm.csv`.

### 4.2 MTP-II, first direction (preference study, superseded)

Scripts 16 to 19 score the emotion of the *preceding* sentence and build the HUTB master sheet (19 needs 16 → 17 → 18). The supervisor redirected the study on 26 Aug 2026 to the difficulty question below. The other scripts of that first direction and the bounded-rationality side analysis were moved to `_archive/` on 2 Nov 2026.

### 4.3 MTP-II, final analysis: emotion vs processing difficulty

| Step | Script | Output |
|---|---|---|
| 1 | `20_build_hutb_raw_dataset.py` (`--emotion-only` to refresh emotion only) | HUTB per-sentence emotion, surprisal and DL (`MTP2_raw_dataset_full.csv`, `MTP2_raw_perword.csv`) |
| 2 | `21_train_trigram_emille.py` (`22_compare_trigram_backoff.py` compares backoff schemes) | `trigram_model_emille.pkl` |
| 3 | `23_score_disco.py` | DISCO emotion, trigram surprisal and Zipf frequency (`disco_affect_scored.csv`) |
| 4 | `24_regress_difficulty_emotion.py` (`--raw-only` for raw correlations) | main regressions, both corpora (`difficulty_params_results.csv`, `raw_correlations.csv`) |
| 5 | `25_build_disco_master_sheet.py`, `19_build_hutb_master_sheet.py` | Excel master sheets |
| 6 | `26_check_dominance_robustness.py`, `27_check_frequency_control.py` | robustness checks (tertiles, nested models, content words, frequency control) |
| 7 | `28_retrain_lstm.py [--resume --epochs N --decay 0.7]` | retrained LSTM `emille_lstm_v2.pt` (held-out perplexity 156.1; unigram baseline 927) |
| 8 | `29_score_lstm_v2.py` | LSTM surprisal for HUTB and DISCO (`lstm_v2_scores.csv`) |
| 9 | `30_regress_lstm_v2.py` | trigram vs LSTM, both corpora: **final results table** (`lstm_v2_regressions.csv`) |
| 10 | `30_regress_lstm_v2.py --kn` | same models with the Kneser-Ney trigram (`kn_trigram_regressions.csv`, `kn_trigram_scores.csv`) |
| 11 | `30_regress_lstm_v2.py --disfl` | DISCO disfluency vs emotion, exploratory (`disco_disfluency_regressions.csv`) |
| 12 | `31_correct_multiple_comparisons.py` | Holm and FDR correction of the 18 primary tests (`mtp2_corrected.csv`) |
| 13 | `32_analyze_pud.py` | third corpus, Hindi PUD; run in conda base for the LSTM part (`data_pud/results/pud_mtp2_regressions.csv`) |
| 14 | `33_make_report_figures.py` | report figures and tables (`data/report/`) |

DISCO shortcut: `bash scripts/run_disco_pipeline.sh full` (23 → 24 → 25).

## 5. How the measures are computed

* **Emotion.** Each sentence gets the mean NRC-VAD (Hindi) valence, arousal and dominance over its matched words. Valence and dominance are rescaled to (−1)-(1); arousal stays 0-1. Negators (नहीं, न, मत, ना) and intensifiers (बहुत, बेहद, …) are modifiers and are not scored themselves. A negator flips the valence and dominance of its dependency head. An intensifier multiplies the next word's arousal by 1.2, capped at 1.0.
* **Trigram surprisal.** From the EMILLE trigram model, backing off trigram → bigram → unigram (floor 10⁻¹⁰). −log₂ p is summed over the sentence and divided by the token count.
* **LSTM surprisal.** From a 2-layer, 200-dimensional LSTM trained on EMILLE, with the same per-word convention as the trigram.
* **Dependency length.** The sum of |position − head position| over words, punctuation included, divided by the token count. MTP-I uses Temperley's version: intervening words, a sentence total.
* **Regression (MTP-II).** Z-scored OLS with HC3 standard errors. Controls: log length and frequency (HUTB: EMILLE log frequency; DISCO: mean Zipf frequency plus task domain).
* **Classification (MTP-I).** Joachims pairwise set-up: (reference − variant, label 1) and (variant − reference, label 0). Logistic regression without intercept, 10-fold CV.

## 6. Toolkit

```
streamlit run toolkit/app.py
```

Type a Hindi sentence, optionally with the previous sentence, to get three tabs:

1. emotion per word and per sentence;
2. dependency length, trigram and LSTM surprisal with corpus percentiles;
3. every reordering of the preverbal constituents, ranked by a model trained on HUTB (91.73% 10-fold CV).

Details in `toolkit/README.md`; `toolkit/train_demo_ranker.py` rebuilds the ranker.

## 7. Where each reported number comes from

| Result | File | Script |
|---|---|---|
| MTP-I accuracy table | `data/results/table4_results.csv` | 15 |
| MTP-II main table (trigram vs LSTM, both corpora) | `data/results/lstm_v2_regressions.csv` | 30 |
| MTP-II first trigram regressions, bootstrap | `data/results/difficulty_params_results.csv` | 24 |
| Raw correlations | `data/results/raw_correlations.csv` | 24 `--raw-only` |
| Robustness | `data/results/dominance_robustness_*.csv` | 26, 27 |
| Kneser-Ney trigram check | `data/results/kn_trigram_regressions.csv` | 30 `--kn` |
| Multiple-comparison correction | `data/results/mtp2_corrected.csv` | 31 |
| Third corpus (PUD) | `data_pud/results/pud_mtp2_regressions.csv` | 32 |
| Disfluency (exploratory) | `data/results/disco_disfluency_regressions.csv` | 30 `--disfl` |
| Toolkit ranker | `toolkit/ranker.json` | `toolkit/train_demo_ranker.py` |

## 8. Known issues

* **Original LSTM.** The LSTM from `08_train_lstm.py` is partly context-insensitive: shuffled sentences got the same score in 119 of 300 checks (`scripts/extra_diag/diagnose_lstm_context.py`). The retrained LSTM fixes this (8 of 300), but it does not reproduce the paper's LSTM-alone accuracy. The reason is unexplained; a larger model is likely needed.
* **Lexicon.** NRC-VAD is a translated, context-free word list. Some function words get scores (ने: valence −0.83), and inflected forms are missed (no lemmatisation).
* **Parses.** DISCO dependency lengths and toolkit parses come from automatic parsers (spaCy HDTB model, UAS 91.5; Stanza), not gold trees.
* **PCFG scores.** `pcfg_scores.pkl` is not in the repository. Script 14 carries the PCFG column over from the previous `all_features_final.csv` (checked row by row).
* **Interpretation.** All MTP-II effects are small (about 0.1 SD per SD) and correlational.

## 9. References

* Ranjan, S., van Schijndel, M., Agarwal, S., & Rajkumar, R. (2022). Discourse Context Predictability Effects in Hindi Word Order. *EMNLP 2022*.
* Ferré, P., et al. (2025). Affect-as-information account (see the final report for the full reference).
* Mohammad, S. M. NRC VAD Lexicon v2.1.
* Bhat, V., Jyothi, P., & Bhattacharyya, P. (2023). DISCO: A Large Scale Human Annotated Corpus for Disfluency Correction in Indo-European Languages. *Findings of EMNLP 2023*.
* Universal Dependencies: UD Hindi-HDTB and UD Hindi-PUD treebanks.
* Qi, P., et al. (2020). Stanza: A Python NLP Toolkit for Many Human Languages. *ACL 2020 System Demonstrations*.
