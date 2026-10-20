"""Banking77 intent classification: TF-IDF+LR baseline vs class-balanced, temperature-scaled word+char model.

Both models share data loading, the word TF-IDF features, the evaluation function (macro-F1,
accuracy, ECE) and the same train/test split. The test set is used only once, at the end.
"""
import io, json, sys, time
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import softmax, log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
T0 = time.time()
BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
URLS = {"train": BASE + "train.csv", "test": BASE + "test.csv"}
SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets (banking_data)"


def fetch(url, retries=5):
    """Download a text file with timeout=60 and exponential backoff on HTTP 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  download failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def load(split):
    """Return (texts, labels) for the official Banking77 split."""
    df = pd.read_csv(io.StringIO(fetch(URLS[split])))
    tcol = "text" if "text" in df.columns else df.columns[0]
    lcol = "category" if "category" in df.columns else df.columns[1]
    return df[tcol].astype(str).str.strip().tolist(), df[lcol].astype(str).to_numpy()


def ece(proba, y, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf, correct = proba.max(1), (proba.argmax(1) == y).astype(float)
    edges, err = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def evaluate(name, proba, y):
    """Shared evaluation: macro-F1 (main), accuracy and ECE (secondary)."""
    pred = proba.argmax(1)
    return {f"{name}_score": f1_score(y, pred, average="macro"),
            f"{name}_accuracy": accuracy_score(y, pred),
            f"{name}_ece": ece(proba, y)}


print("[1/5] loading Banking77 (official train/test split)")
Xtr_txt, ytr_raw = load("train")
Xte_txt, yte_raw = load("test")
classes = np.unique(ytr_raw)
assert set(yte_raw) <= set(classes), "test labels unseen in training"
y_tr = np.searchsorted(classes, ytr_raw)
y_te = np.searchsorted(classes, yte_raw)
print(f"  train={len(y_tr)} test={len(y_te)} classes={len(classes)}")

# Shared word unigram+bigram TF-IDF (fit on training text only, no labels used)
word_vec = TfidfVectorizer(ngram_range=(1, 2))
Wtr = word_vec.fit_transform(Xtr_txt)
Wte = word_vec.transform(Xte_txt)

print("[2/5] baseline: word TF-IDF + unweighted multinomial LR (C=1)")
base = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
base.fit(Wtr, y_tr)
base_proba = base.predict_proba(Wte)

print("[3/5] improved: word + char TF-IDF, class-balanced LR, C chosen by 3-fold CV on train")
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100000)
Ftr = hstack([Wtr, char_vec.fit_transform(Xtr_txt)]).tocsr()
Fte = hstack([Wte, char_vec.transform(Xte_txt)]).tocsr()


def make_lr(C):
    return LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=SEED)


cv3 = StratifiedKFold(3, shuffle=True, random_state=SEED)
cv_f1 = {}
for C in [3, 10, 30]:  # small grid, selection uses training data only
    scores = [f1_score(y_tr[va], make_lr(C).fit(Ftr[tr], y_tr[tr]).predict(Ftr[va]), average="macro")
              for tr, va in cv3.split(Ftr, y_tr)]
    cv_f1[C] = float(np.mean(scores))
    print(f"  C={C}: CV macro-F1={cv_f1[C]:.4f}")
best_C = max(cv_f1, key=cv_f1.get)
print(f"  selected C={best_C}")

print("[4/5] temperature scaling on out-of-fold training logits (5-fold, held-out folds)")
oof = np.zeros((len(y_tr), len(classes)))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Ftr, y_tr):
    oof[va] = make_lr(best_C).fit(Ftr[tr], y_tr[tr]).decision_function(Ftr[va])


def nll(T):
    return -log_softmax(oof / T, axis=1)[np.arange(len(y_tr)), y_tr].mean()


T = float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)
print(f"  fitted temperature T={T:.3f}")

imp = make_lr(best_C).fit(Ftr, y_tr)  # refit on full training set
imp_logits = imp.decision_function(Fte)
imp_proba = softmax(imp_logits / T, axis=1)
imp_uncal = softmax(imp_logits, axis=1)

print("[5/5] evaluating both models once on the test set")
res = {}
res.update(evaluate("baseline", base_proba, y_te))
res.update(evaluate("improved", imp_proba, y_te))
res["improved_uncalibrated_ece"] = ece(imp_uncal, y_te)
res["improvement"] = res["improved_score"] - res["baseline_score"]  # higher F1 is better
res["ece_improvement"] = res["baseline_ece"] - res["improved_ece"]  # positive = improved has lower ECE
res["cv_best_C"] = float(best_C)
res["temperature"] = T
for k, v in res.items():
    print(f"  {k}: {v:.4f}")

if not all(np.isfinite(v) for v in res.values()):
    print("Non-finite metric produced; failing.")
    sys.exit(1)

# Optional chart: baseline vs improved
import os
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, metric, title in [(axes[0], "score", "Macro-F1 (higher better)"),
                          (axes[1], "ece", "ECE (lower better)")]:
    vals = [res[f"baseline_{metric}"], res[f"improved_{metric}"]]
    ax.bar(["baseline", "improved"], vals, color=["grey", "steelblue"])
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("figures/baseline_vs_improved.png", dpi=120)

results = {
    "question": "Can a class-balanced, calibrated text classifier beat a standard TF-IDF logistic-regression "
                "intent model on macro-F1 and expected calibration error for banking customer-service queries?",
    "method": "Banking77 official train/test split; baseline TF-IDF word 1-2gram + unweighted multinomial LR; "
              "improved word+char TF-IDF + class-balanced LR (C by 3-fold CV on train) + temperature scaling "
              "fitted on 5-fold out-of-fold training logits. Test set used once.",
    "dataset": {
        "name": "Banking77 (PolyAI)",
        "source_url": SOURCE,
        "n_rows": int(len(y_tr) + len(y_te)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "same_as_paper": False,
        "baseline_is_papers_method": False,
    },
    "metric_name": "macro-F1",
    "higher_is_better": True,
    "baseline": "Standard TF-IDF (word unigrams+bigrams) with unweighted multinomial logistic regression, C=1.",
    "improved": "Word+char_wb(2-5) TF-IDF, class-balanced logistic regression with C chosen by CV on train, "
                "and temperature scaling of softmax probabilities fitted on held-out training folds.",
    "metrics": res,
    "notes": "Official PolyAI split used as the single split (test is about 23% of data, not a stratified random "
             "split). Both LR models are deterministic (lbfgs), so one run is reported with no seed variation. "
             "The final improved model is refit on the full training set while the temperature was fitted on "
             "out-of-fold logits from 80% fits, a common but approximate practice. ECE uses 15 equal-width bins. "
             "The paper's own model and dataset were not specified, so this is a standard-baseline comparison only. "
             f"Runtime: {time.time() - T0:.0f}s.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Done in {time.time() - T0:.0f}s; wrote results.json")
