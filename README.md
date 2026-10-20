# Banking77 intent classification: calibrated, class-balanced TF-IDF vs standard baseline

Motivation: The paper "Experimental evaluation of machine learning models for goal-oriented customer service chatbot with pipeline architecture" (EA Kamaru Zaman et al., 2026) compares machine learning models for goal-oriented customer-service chatbots, where intent classification errors directly affect routing. Accurate and well-calibrated intent confidences matter as much as raw accuracy, because a chatbot must decide when to hand off to a human. This experiment checks a simple, cheap version of that idea on a public banking dataset.

Question: Can a class-balanced, calibrated text classifier beat a standard TF-IDF logistic-regression intent model on macro-F1 and expected calibration error for banking customer-service queries?

Dataset: Banking77 (77 banking intents), downloaded as CSV from the PolyAI task-specific-datasets GitHub repository, using its official train/test split. This is not the paper's own dataset; the paper's model and data were not specified.

Baseline: TF-IDF word unigrams and bigrams with unweighted multinomial logistic regression. This stands in for the paper's unspecified intent model.

Improved model: word plus character n-gram TF-IDF with class-balanced logistic regression, C chosen by cross-validation on the training data, and temperature scaling fitted on held-out training folds to improve confidence reliability.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the Banking77 test set (3,080 queries), a standard TF-IDF logistic-regression intent classifier reached 85.3% macro-F1 and 45.3% ECE, while my class-balanced, temperature-scaled word+char model reached 91.7% macro-F1 and 1.5% ECE.

| Metric | Value |
|---|---|
| baseline_score | 0.8531 |
| baseline_accuracy | 0.8545 |
| baseline_ece | 0.4531 |
| improved_score | 0.9171 |
| improved_accuracy | 0.9169 |
| improved_ece | 0.01482 |
| improved_uncalibrated_ece | 0.09025 |
| improvement | 0.06396 |
| ece_improvement | 0.4383 |
| cv_best_C | 10 |
| temperature | 0.6258 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Official PolyAI split used as the single split (test is about 23% of data, not a stratified random split). Both LR models are deterministic (lbfgs), so one run is reported with no seed variation. The final improved model is refit on the full training set while the temperature was fitted on out-of-fold logits from 80% fits, a common but approximate practice. ECE uses 15 equal-width bins. The paper's own model and dataset were not specified, so this is a standard-baseline comparison only. Runtime: 223s.

## Reference

This project is motivated by the following paper:

> EA Kamaru Zaman, NAN Mohd Isa, SNA Jawaddi (2026). *Experimental evaluation of machine learning models for goal-oriented customer service chatbot with pipeline architecture: EA Kamaru Zaman et al.*. Soft Computing. https://link.springer.com/article/10.1007/s00500-026-11411-x

Paper author: Prof. Azlan Ismail, Universiti Teknologi MARA (UiTM) (azlanismail@uitm.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
