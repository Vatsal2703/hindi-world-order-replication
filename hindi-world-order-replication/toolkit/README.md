# Hindi Emotion & Processing-Difficulty Toolkit (v1.0)

Demo toolkit for the MTP. Type a Hindi sentence (and optionally the sentence before it) and it shows:

1. **Emotion**: valence, arousal and dominance per word and for the sentence (NRC-VAD v2.1 Hindi, thesis modifier rules), with the sentence's percentile in HUTB and DISCO.
2. **Processing difficulty**: dependency length, trigram surprisal and LSTM surprisal, per word and per sentence, with corpus percentiles and the parse.
3. **Word order (MTP-I)**: all reorderings of the constituents before the verb, ranked by a pairwise model trained on HUTB. It shows whether the original order comes first and which features decided the ranking.

## Run

From the repo root, in the conda env that has torch (`base`):

```
pip install streamlit stanza
streamlit run toolkit/app.py
```

A browser tab opens. The first run downloads the Stanza Hindi parser once (needs internet); after that it runs offline. Model loading takes about 30–60 s, then each sentence takes about 1 s.

## Files

| File | What it is |
|---|---|
| `app.py` | The toolkit (Streamlit) |
| `train_demo_ranker.py` | Trains the word-order ranker; writes `ranker.json` (no torch needed) |
| `ranker.json` | Ranker weights and 10-fold CV accuracy |

## What it reuses from the pipeline

| Part | Source in the repo |
|---|---|
| Emotion rules | script 23 v1.3 / script 20 v2.6 |
| Trigram surprisal | `data/models/trigram_model_emille.pkl`, same backoff as scripts 20/23 |
| LSTM surprisal | `data/models/emille_lstm_v2.pt` (scripts 28/29) |
| Variant generation | script 04 rules (permute preverbal subtree blocks, at most 99 variants) |
| Dependency length and information status | `src/features/basic_features.py` (script 05) |
| Corpus reference values | `MTP2_refs_with_emille_freq.csv`, `disco_affect_scored.csv`, `disco_dl.csv`, `lstm_v2_scores.csv` |

The emotion and trigram code was checked against the thesis values on 300 and 200 HUTB sentences, using the gold trees: the differences were 0. Variant counts matched script 04 for 299 of 300 sentences; the one difference is from random sampling when a sentence has more than 99 variants.

## Ranker

The ranker is a logistic regression without an intercept, fitted on reference-minus-variant differences of four features: dependency length, information status, trigram surprisal and LSTM surprisal. It was trained on 2,753 HUTB sentences (178,722 pairs), with 91.73% 10-fold CV accuracy (trigram alone 90.34%, LSTM alone 86.29%).

The full MTP-I model (94.62%) also uses PCFG, lexical-repetition and adaptive-LSTM surprisal. Those need a Berkeley parser and the whole document, so they can't be computed live for one typed sentence.

## Limits

Typed sentences are parsed by Stanza (trained on UD Hindi-HDTB), not gold trees, so dependency length and variants carry parser error. Emotion is lexicon-based and ignores context.
