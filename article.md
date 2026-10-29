# Mastering Search Spaces and Callbacks in sklearn-genetic-opt: Faster, Smarter Hyperparameter Tuning

*A practical guide to escaping brute-force grid search, modeling continuous and categorical parameter spaces, and saving compute with evolutionary early-stopping callbacks.*

---

## Introduction: Why Move Beyond Grid and Random Search?

Every machine learning practitioner knows the hyperparameter dilemma:
- **`GridSearchCV`** exhaustively evaluates every combination within a user-defined discrete grid, but it suffers from severe combinatorial explosion. If you have 6 hyperparameters with 5 values each, you are evaluating \(5^6 = 15,625\) candidates. With 5-fold cross-validation, that requires 78,125 model fits.
- **`RandomizedSearchCV`** breaks the rigid grid constraint by sampling randomly from distributions, but it is fundamentally *memoryless*. The 100th trial has zero knowledge of what the previous 99 trials learned.

This is where **genetic and evolutionary algorithms** provide a powerful middle ground. Inspired by natural selection, libraries like **`sklearn-genetic-opt`** treat candidate hyperparameter sets as "chromosomes" (or individuals). High-performing configurations survive, reproduce through crossover, and introduce targeted mutations.

The result? The algorithm dynamically explores promising regions of the parameter landscape while exploiting configurations that yield superior validation performance—without requiring exhaustive evaluation.

In this tutorial, we will explore two essential features of `sklearn-genetic-opt` that turn it into a production-grade optimization tool:
1. **Designing Custom Search Spaces** using `Integer`, `Continuous` (uniform and log-uniform), and `Categorical` distributions.
2. **Controlling Compute with Intelligent Callbacks** like `ConsecutiveStopping`, `ProgressBar`, and `TimerStopping` to prevent wasted compute cycles when models reach convergence.

---

## 1. Anatomy of an Evolutionary Search Space

In standard `scikit-learn`, `GridSearchCV` only accepts explicit discrete lists of values. `sklearn-genetic-opt` introduces a first-class search space module (`sklearn_genetic.space`) that allows continuous distributions alongside discrete and categorical domains:

```python
from sklearn_genetic.space import Continuous, Integer, Categorical

param_grid = {
    # Discrete integer range [lower, upper]
    "n_estimators": Integer(50, 350),
    "max_depth": Integer(3, 16),
    "min_samples_split": Integer(2, 16),
    "min_samples_leaf": Integer(1, 8),
    
    # Categorical set of discrete options (strings, booleans, or callables)
    "max_features": Categorical(["sqrt", "log2", None]),
    "criterion": Categorical(["gini", "entropy"]),
    
    # Continuous distribution (uniform or log-uniform)
    # Useful for regularization parameters (C, alpha) or learning rates:
    # "learning_rate": Continuous(0.001, 0.2, distribution="log-uniform")
}
```

### Key Differences Between Space Types:

| Class | Distribution Options | When to Use |
| :--- | :--- | :--- |
| `Integer(low, high)` | Uniform | Tree depth, ensemble count, number of features, mini-batch size |
| `Continuous(low, high)` | `'uniform'`, `'log-uniform'` | Learning rates, regularization penalties (`alpha`, `C`), dropout rates |
| `Categorical(choices)` | Discrete choices | Activation functions, kernels (`rbf`, `linear`), feature selection strategies |

> **Pro Tip on Log-Uniform Distributions**: Whenever tuning parameters that operate on orders of magnitude (e.g., SVM `C` between 0.001 and 1000, or Gradient Boosting learning rate between 0.001 and 0.5), always set `distribution="log-uniform"`. This ensures mutations explore \(10^{-3}\) to \(10^{-2}\) with the same statistical weight as \(10^{1}\) to \(10^{2}\).

---

## 2. Never Waste Compute: The Callback Ecosystem

A common challenge with heuristic optimization is selecting the right generation count: running too few generations risks under-optimizing, while running 20 or 50 fixed generations wastes compute once the population has already converged around high-performing solutions.

`sklearn-genetic-opt` addresses this via the **Callbacks API** (`sklearn_genetic.callbacks`):

- **`ProgressBar()`**: Real-time terminal progress bar powered by `tqdm`, displaying elapsed time, remaining iterations, and generation speeds.
- **`ConsecutiveStopping(generations=N, metric="fitness")`**: Halts the evolutionary loop if the monitored metric does not improve for `N` consecutive generations.
  - **API Detail**: In `sklearn-genetic-opt`, the default `metric="fitness"` corresponds to the **population mean fitness** (the average CV score across all individuals in that generation). This monitors whether the overall population is still making collective progress.
  - If you want the early stopper to track only the single champion configuration instead of the population average, explicitly set `metric="fitness_best"` or `metric="fitness_max"`.
- **`TimerStopping(total_seconds=S)`**: Sets a soft time budget for optimization.
  - **API Detail**: When `total_seconds` expires, the optimizer ceases scheduling new candidate generation batches. However, in-flight cross-validation fold evaluations within the current generation and the final best-estimator refit are allowed to finish gracefully. As a result, total wall-clock time may slightly exceed `total_seconds` rather than terminating via an abrupt, mid-calculation process kill.
- **`DeltaThreshold(threshold=0.001)`**: Stops optimization if the rate of metric improvement between generations falls below a specified minimum difference.
- **`ModelCheckpoint(filepath="checkpoint.pkl")`**: Periodically persists intermediate results to disk.

By combining `ConsecutiveStopping` with `TimerStopping`, you can set generous generation limits (`generations=30`) with the confidence that the search will exit automatically once improvements plateau or time runs out.

---

## 3. Hands-On Implementation: Optimizing a Random Forest

Let's put this into practice on a real classification task using the Wisconsin Breast Cancer diagnostic dataset. We will optimize a `RandomForestClassifier` targeting `roc_auc` scoring.

### Step 1: Environment Setup

Install the required packages:

```bash
pip install scikit-learn sklearn-genetic-opt
```

### Step 2: Full Python Workflow

```python
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

# Import genetic optimization classes
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical
from sklearn_genetic.callbacks import ProgressBar, ConsecutiveStopping, TimerStopping

# 1. Load and partition dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Define the multi-type parameter space
param_grid = {
    "n_estimators": Integer(50, 350),
    "max_depth": Integer(3, 16),
    "min_samples_split": Integer(2, 16),
    "min_samples_leaf": Integer(1, 8),
    "max_features": Categorical(["sqrt", "log2", None]),
    "criterion": Categorical(["gini", "entropy"]),
}

# 3. Setup monitoring and early-stopping callbacks
callbacks = [
    ProgressBar(),
    ConsecutiveStopping(generations=4, metric="fitness"),
    TimerStopping(total_seconds=120)  # Stop scheduling new batches after 120s
]

# 4. Configure the genetic search estimator
base_estimator = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

ga_search = GASearchCV(
    estimator=base_estimator,
    cv=cv,
    scoring="roc_auc",
    param_grid=param_grid,
    population_size=15,
    generations=12,
    tournament_size=3,
    elitism=True,
    crossover_probability=0.8,
    mutation_probability=0.2,
    algorithm="eaSimple",
    n_jobs=-1,
    verbose=True,
    random_state=42
)

# 5. Execute optimization
print("Launching GASearchCV with Early Stopping Callbacks...")
start_time = time.time()
ga_search.fit(X_train, y_train, callbacks=callbacks)
ga_duration = time.time() - start_time

print(f"\nOptimization completed in {ga_duration:.2f} seconds.")
print("Best Parameters:", ga_search.best_params_)
print(f"Best CV ROC-AUC: {ga_search.best_score_:.4f}")
```

---

## 4. Execution Logs and Early Stopping in Action

When you run the code, `sklearn-genetic-opt` prints a live logbook tracking both population averages and generation champions:

```text
 gen evals           avg          best     div  unique  stag     mut   sel             events
---- ----- ------------- ------------- ------- ------- ----- ------- ----- ------------------
   0    15       0.98723       0.98919   0.856   1.000     0       -     - -                 
   1    14       0.98762       0.98982   0.775   1.000     0   0.200     3 dup=6             
   2    14       0.98839       0.98997   0.713   1.000     0   0.200     3 dup=4             
   3    14       0.98859       0.98997   0.633   1.000     1   0.200     3 dup=5             
   4    14       0.98760       0.98997   0.773   1.000     2   0.200     3 dup=11            
   5    14       0.98744       0.98997   0.783   1.000     3   0.200     3 dup=7             
INFO: ConsecutiveStopping callback met its criteria
INFO: Stopping the algorithm
```

### What Happened During Optimization?

1. **Exploration & Exploitation (Generations 0 to 2)**:
   - In Generation 0, random initialization established a baseline best validation ROC-AUC of `0.98919` and population average of `0.98723`.
   - Crossover and mutation rapidly surfaced higher-performing configurations: by Generation 2, the best candidate reached `0.98997` and the population average rose to `0.98839`.
2. **Convergence & Stagnation (Generations 3 to 5)**:
   - Thanks to elitism (`elitism=True`), the champion candidate (`0.98997`) was preserved intact.
   - However, the population mean (`avg`) plateaued at `0.98859` in Generation 3, before slightly drifting to `0.98760` and `0.98744` as exploration saturated around the local basin.
3. **Early Termination**:
   - Because `ConsecutiveStopping(generations=4, metric="fitness")` checks whether the current generation's mean fitness exceeds the past 4 generations, it detected that the swarm was no longer making mean improvements.
   - The callback fired after Generation 5, terminating the search early and avoiding unnecessary iterations.

---

## 5. Performance Comparison & Measuring Compute Savings

To provide a fair, reproducible benchmark, we configure `RandomizedSearchCV` to perform the exact same amount of cross-validation work by retrieving `ga_search.fit_stats_["cross_validate_calls"]` (83 CV calls):

```python
# Match the exact number of cross-validation evaluations performed by genetic search
cv_calls = ga_search.fit_stats_["cross_validate_calls"]  # 83 evaluations

rand_search = RandomizedSearchCV(
    estimator=base_estimator,
    param_distributions=rand_param_dist,
    n_iter=cv_calls,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1
)
rand_search.fit(X_train, y_train)
```

Evaluating both models on the held-out test set under matched CV work:

| Search Strategy | CV Calls / Work | CV ROC-AUC | Test Accuracy | Test ROC-AUC | Compute Workload |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RandomizedSearchCV** | 83 candidates (249 fold fits) | 0.9897 | 95.80% | 0.9941 | Matched Baseline Workload |
| **GASearchCV (Early Stopped)** | 83 candidates (249 fold fits) | **0.9900** | **95.80%** | **0.9943** | **~53.6% estimated reduction vs. planned budget** |

### Analyzing the Results

- **Matched Performance**: At an identical evaluation budget of 83 cross-validation evaluations, both methods achieve equal test accuracy (**95.80%**) and near-identical test discrimination (ROC-AUC ~0.994), with `GASearchCV` yielding slightly higher cross-validated training fitness (`0.9900` vs. `0.9897`) and test ROC-AUC (`0.9943` vs. `0.9941`).
- **Ensemble Complexity**: Evolutionary search converged to a more compact forest (180 trees vs. 290 trees), reducing downstream inference latency and memory footprint.
- **Runtime Trade-offs**: `RandomizedSearchCV` finished faster in raw wall-clock time because it samples independently in parallel without the generational synchronization barriers required by genetic selection and crossover.

### How Were the Compute Savings Measured?

We analyze the compute reduction in two complementary ways:

1. **Estimated Projection vs. Scheduled Budget**:
   With `population_size=15` and `generations=12`, the algorithm was scheduled for up to **~183 candidate evaluations** (15 initial + 12 generations $\times$ ~14 evaluated offspring candidates, accounting for deduplication/caching). Because `ConsecutiveStopping` terminated the search after Generation 5, only **85 candidates** were evaluated:
   $$\text{Estimated Savings} = 1 - \frac{85}{183} \approx 53.6\%$$

2. **Empirically Measured Against a Full 12-Generation Run**:
   Running the identical seed to the full 12 generations without early stopping recorded `fit_stats_["cross_validate_calls"] = 180`. With early stopping active, `fit_stats_["cross_validate_calls"] = 83`:
   $$\text{Measured Reduction in Actual CV Calls} = 1 - \frac{83}{180} \approx 53.9\%$$

Both the analytical projection (~53.6%) and the empirical baseline run (~53.9%) confirm that early stopping saved over half the cross-validation workload without degrading validation or test metrics.

The best hyperparameters identified by evolutionary selection:
```python
{
    'n_estimators': 180,
    'max_depth': 8,
    'min_samples_split': 9,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'criterion': 'entropy'
}
```

Classification report on unseen test data:
```text
              precision    recall  f1-score   support

   malignant       0.96      0.92      0.94        53
      benign       0.96      0.98      0.97        90

    accuracy                           0.96       143
   macro avg       0.96      0.95      0.95       143
weighted avg       0.96      0.96      0.96       143
```

---

## 6. Key Takeaways & Best Practices

When applying `sklearn-genetic-opt` in your own machine learning pipelines:

1. **Leverage Elitism (`elitism=True`)**: Always retain elitism so that the best candidate from generation \(k\) is guaranteed to survive into generation \(k+1\). This protects against stochastic degradation.
2. **Align Your Stopping Metric**:
   - Use `metric="fitness"` (default) to stop when the **population mean** stagnates.
   - Use `metric="fitness_best"` or `metric="fitness_max"` if you want to stop strictly when the **single top individual** stops improving.
3. **Remember TimerStopping Is a Soft Boundary**: Because `TimerStopping` allows the active generation and final refit to conclude cleanly, factor in generation evaluation times when budgeting tight wall-clock limits.
4. **Tune Continuous Parameters with Log-Uniform**: For regularization terms (`C`, `alpha`) and learning rates, log-uniform sampling explores across orders of magnitude much more systematically than uniform sampling.
5. **Take Advantage of Parallelism (`n_jobs=-1`)**: Since candidate evaluations within a generation are embarrassingly parallel, setting `n_jobs=-1` distributes fold fits across all available CPU cores.

---

## Conclusion

Hyperparameter tuning does not require choosing between the rigid brute force of `GridSearchCV` and the unguided randomness of `RandomizedSearchCV`. 

While evolutionary algorithms do not guarantee a global optimum, combining structured parameter distributions via `sklearn_genetic.space` with responsive early stopping callbacks via `sklearn_genetic.callbacks` allows you to explore complex, non-linear parameter landscapes efficiently—discovering high-performing model configurations while avoiding wasted compute.

*Have you experimented with evolutionary search or custom callbacks in your machine learning workflows? Share your thoughts and benchmark comparisons in the comments!*
