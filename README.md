# Mastering Search Spaces and Callbacks in sklearn-genetic-opt

> Companion repository for the technical tutorial: **[Mastering Search Spaces and Callbacks in sklearn-genetic-opt: Faster, Smarter Hyperparameter Tuning](./article.md)**

This repository provides a hands-on guide and runnable Python implementation exploring how to leverage **`sklearn-genetic-opt`** for efficient evolutionary hyperparameter optimization.

---

## 📖 Read the Full Tutorial

👉 **[Read the Full Article (article.md)](./article.md)**

### Key Topics Covered
- **Moving Beyond Grid and Random Search**: How evolutionary algorithms balance exploration and exploitation across non-linear hyperparameter surfaces.
- **Designing Custom Search Spaces**: Defining `Integer`, `Continuous` (uniform vs. log-uniform), and `Categorical` distributions with `sklearn_genetic.space`.
- **Intelligent Early Stopping**: Preventing wasted compute using `ConsecutiveStopping`, `ProgressBar`, and `TimerStopping`.
- **API Specifics**: Understanding `ConsecutiveStopping` comparison logic (`metric <= past N values`), population mean tracking (`metric="fitness"`) vs. generation champions (`metric="fitness_best"`), and graceful completion in `TimerStopping`.
- **Measured Compute Savings**: How early stopping saved over 53% of candidate evaluation rounds and cross-validation fits compared to the planned search budget.

---

## 🚀 Quickstart & Code

Run the accompanying benchmark script locally:

```bash
# 1. Install dependencies
pip install scikit-learn sklearn-genetic-opt

# 2. Run the tutorial benchmark
python tutorial_code.py
```

### Repository Structure

- [**`article.md`**](./article.md): The complete technical article.
- [**`tutorial_code.py`**](./tutorial_code.py): Executable script running `GASearchCV` with mixed distributions and early stopping callbacks on the Wisconsin Breast Cancer diagnostic dataset.

---

## 📊 Benchmark Summary

| Method | CV Calls / Work | CV ROC-AUC | Test Accuracy | Test ROC-AUC | Compute Workload |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RandomizedSearchCV** | 83 candidates (249 fold fits) | 0.9897 | 95.80% | 0.9941 | Matched Baseline Workload |
| **GASearchCV (Early Stopped)** | 83 candidates (249 fold fits) | **0.9900** | **95.80%** | **0.9943** | **~53.6% estimated reduction vs. planned budget** |

*See [article.md](./article.md) for full methodology, parameter spaces, and detailed classification reports.*
