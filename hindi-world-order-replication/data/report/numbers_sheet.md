# Verified numbers for slides, report and viva (v1.1, 3 Nov 2026)

Every number below comes from a results file in the repo (shown in brackets). If a result is rerun, take the new value from the file, not from this sheet.

## Data
- HUTB reference sentences: **2,779**. **2,753** are in the MTP-I classification, because 26 drop out when the LSTM scores are merged with the trigram feature file (script 14). [all_variants_final.pkl, all_features_final.csv]
- MTP-I word-order variants: **93,306**, at most 99 per sentence. Pairwise rows: **178,722**. [script 14 output]
- DO-fronted: 145 sentences, 6,622 rows. IO-fronted: 82 sentences, 5,080 rows. [script 15 output]
- DISCO Hindi: **3,171** utterances scored (3,180 minus 9 with no Devanagari word). [disco_affect_scored.csv]
- Hindi PUD (third corpus): **1,000** sentences, gold UD trees. [data_pud/results/pud_mtp2_scores.csv]
- NRC-VAD v2.1 Hindi lexicon: 11,937 entries.
- EMILLE: about 14.3 million tokens. LSTM training set: 978,504 sentences; validation: 19,969 sentences; vocabulary: 32,478.

## MTP-I (replication of Ranjan et al. 2022) [table4_results.csv]
- Full model (base1 + g): **94.62%** against the paper's **95.06%** (−0.44 points).
- Single features, ours vs paper: IS 52.77 vs 51.84; DL 61.77 vs 62.31; PCFG 89.22 vs 86.86; lexical repetition 91.92 vs 90.07; trigram 92.21 vs 91.18; LSTM 90.27 vs 94.01; adaptive LSTM 90.49 vs 94.06.
- Fronted subsets, full model: DO **84.75** vs 81.06; IO **94.69** vs 89.65. Both are higher than the paper.
- Regression on full data [table2_regression.csv]:
  - every surprisal β is negative: PCFG −0.385, trigram −0.158, adaptive LSTM −0.109, LSTM −0.051, lexical repetition −0.028;
  - IS +0.069; DL −0.017.
- Retrained LSTM instead of the original [table4_results_v2lstm.csv]: LSTM alone 86.29%; full model 94.66%.

## LSTM retraining (MTP-II)
- Original LSTM: shuffled order scored the same as the original in **119 of 300** sentences, so it was partly context-insensitive. [script extra_diag/diagnose_lstm_context.py]
- Retrained: held-out perplexity **156.1** against a unigram baseline of **927** (8 epochs in total). Same score in 8 of 300 sentences.

## MTP-II main results [lstm_v2_regressions.csv]
Standardised β, controls for length and frequency, HC3 SE.

| Effect | HUTB | DISCO |
|---|---|---|
| Dominance ~ dependency length | **−0.090** (p < .001) | **−0.087** (p < .001) |
| Valence ~ LSTM surprisal | **+0.108** (p < .001) | **+0.094** (p < .001) |
| Dominance ~ LSTM surprisal | −0.023 (n.s.) | −0.028 (n.s.) |
| Valence ~ trigram surprisal | +0.035 (n.s.) | −0.146 (p < .001) |
| Dominance ~ trigram surprisal | −0.195 | +0.129 (sign flips) |
| Arousal ~ trigram surprisal | −0.150 | +0.177 (sign flips) |

- With an earlier LSTM checkpoint (perplexity 163.9), valence ~ LSTM was +0.134 / +0.113: same sign.
- Raw correlations are all below |0.21|. DL per word correlates +0.71 with sentence length.
- n-gram vs LSTM surprisal correlate +0.64 in HUTB, −0.08 in DISCO, +0.35 to +0.37 in PUD. DISCO mean bits per word: n-gram 14.5, LSTM 9.9.

## Checks added 31 Oct – 1 Nov
**Multiple-comparison correction** [mtp2_corrected.csv, script 31]
- 18 primary tests: 11 significant before correction, all 11 after Holm and after FDR.
- Holm-adjusted p: dominance ~ DL .004 (HUTB), .002 (DISCO); valence ~ LSTM .001 (HUTB), < .001 (DISCO).
- All 42 model rows: 21 of 25 survive Holm; valence ~ LSTM 4 of 4; dominance ~ DL 4 of 6 (the two HUTB trigram-model rows, β = −0.070, Holm p = .094, FDR p = .009).

**Kneser-Ney trigram** [kn_trigram_regressions.csv, script 30 --kn]
- Same sign flips as the unsmoothed trigram: dominance −0.233 (HUTB) vs +0.122 (DISCO); arousal −0.130 vs +0.172; valence −0.032 (n.s.) vs −0.155.
- Unsmoothed and KN trigram correlate +0.93 (HUTB), +0.99 (DISCO). Smoothing is not the cause of the flips.

**Third corpus, Hindi PUD** [data_pud/results/pud_mtp2_regressions.csv, script 32]
| Effect | HUTB | DISCO | PUD |
|---|---|---|---|
| Dominance ~ dependency length | −0.090 | −0.087 | **−0.100** (p = .029) |
| Valence ~ LSTM surprisal | +0.108 | +0.094 | **+0.029** (p = .416, n.s.; 95% CI −0.041 to +0.099) |
| Dominance ~ trigram surprisal | −0.195 | +0.129 | +0.119 (p = .153) |
| Arousal ~ trigram surprisal | −0.150 | +0.177 | +0.120 (p = .133) |
| Valence ~ dependency length | +0.008 | −0.036 | −0.092 (p = .032) |

**DISCO disfluency, exploratory** [disco_disfluency_regressions.csv, script 30 --disfl]
- Disfluent vs fluent utterances (fluent version scored; LSTM, DL, length, Zipf, domain controlled): valence −0.181 SD (p = .001), dominance −0.124 SD (p = .019); share of words deleted → dominance −0.048 (p = .009).
- 89% of utterances are disfluent (347 fluent). Only the valence difference survives Holm across these 9 tests (adjusted p = .010).
- Disfluencies were elicited in PRESTO, not spontaneous.

**HUTB robustness** [dominance_robustness_report.txt, script 26]
- Trigram-surprisal tertiles: dominance +0.104 → +0.090 → +0.067, Cohen's d = −0.40.
- Full nested model (M3): trigram β = −0.254, 95% bootstrap CI [−0.307, −0.205]; content words only −0.221.

## Toolkit [toolkit/ranker.json]
- Ranker, 10-fold CV: all four features **91.73%**; trigram 90.34%; LSTM 86.29%; DL 61.77%; IS 52.77%.
- Validation:
  - emotion scores identical to the thesis on 300 HUTB sentences;
  - trigram surprisal identical on 200 sentences;
  - variant counts match script 04 (04_generate_variants.py) on 299 of 300 sentences.
- Demo examples, original order ranked 1st (live parse): train-s11422 (1st of 24), train-s1813 (1st of 24), train-s9426 (1st of 6), train-s9275 (1st of 24).
