"""
Practical Guide to Designing Advanced Search Spaces and Callbacks in sklearn-genetic-opt
Companion executable code.
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

# Import evolutionary components from sklearn-genetic-opt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer, Categorical
from sklearn_genetic.callbacks import ProgressBar, ConsecutiveStopping, TimerStopping

def main():
    print("=" * 70)
    print("sklearn-genetic-opt: Search Space & Callbacks Hands-On Tutorial")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[Step 1] Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]} | Target classes: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # 2. Define Mixed Evolutionary Search Space
    print("\n[Step 2] Defining Evolutionary Search Space...")
    param_grid = {
        "n_estimators": Integer(50, 350),
        "max_depth": Integer(3, 16),
        "min_samples_split": Integer(2, 16),
        "min_samples_leaf": Integer(1, 8),
        "max_features": Categorical(["sqrt", "log2", None]),
        "criterion": Categorical(["gini", "entropy"]),
    }

    for param, dist in param_grid.items():
        print(f"  - {param}: {dist}")

    # 3. Setup Callbacks
    print("\n[Step 3] Configuring Callbacks for Monitoring & Early Stopping...")
    callbacks = [
        ProgressBar(),
        ConsecutiveStopping(generations=4, metric="fitness"),
        TimerStopping(total_seconds=120)  # Soft time budget
    ]
    print("  - ProgressBar(): Visual progress bar")
    print("  - ConsecutiveStopping(generations=4, metric='fitness'): Halts when current mean fitness is <= each of the previous 4 generation values")
    print("  - TimerStopping(total_seconds=120): Stops scheduling new batches after 120s; allows in-flight batch and refit to finish")

    # 4. Initialize and Fit GASearchCV
    print("\n[Step 4] Running Genetic Hyperparameter Optimization (GASearchCV)...")
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

    start_ga = time.time()
    ga_search.fit(X_train, y_train, callbacks=callbacks)
    duration_ga = time.time() - start_ga

    print(f"\n[GASearchCV Completed in {duration_ga:.2f} seconds]")
    print(f"  Best Cross-Validated ROC-AUC : {ga_search.best_score_:.4f}")
    print(f"  Best Hyperparameters found   : {ga_search.best_params_}")

    # Test set evaluation
    best_rf_ga = ga_search.best_estimator_
    y_pred_ga = best_rf_ga.predict(X_test)
    y_prob_ga = best_rf_ga.predict_proba(X_test)[:, 1]
    acc_ga = accuracy_score(y_test, y_pred_ga)
    roc_ga = roc_auc_score(y_test, y_prob_ga)
    print(f"  Test Accuracy : {acc_ga * 100:.2f}%")
    print(f"  Test ROC-AUC  : {roc_ga:.4f}")

    # 5. Baseline Comparison: RandomizedSearchCV with matched CV work
    # Use cross_validate_calls from fit_stats_ to count actual cross-validation evaluations
    cv_calls = ga_search.fit_stats_["cross_validate_calls"]
    print(f"\n[Step 5] Running Baseline RandomizedSearchCV for Comparison ({cv_calls} matched CV calls)...")
    rand_param_dist = {
        "n_estimators": list(range(50, 351, 10)),
        "max_depth": list(range(3, 17)),
        "min_samples_split": list(range(2, 17)),
        "min_samples_leaf": list(range(1, 9)),
        "max_features": ["sqrt", "log2", None],
        "criterion": ["gini", "entropy"],
    }
    rand_search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=rand_param_dist,
        n_iter=cv_calls,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    start_rand = time.time()
    rand_search.fit(X_train, y_train)
    duration_rand = time.time() - start_rand

    best_rf_rand = rand_search.best_estimator_
    y_pred_rand = best_rf_rand.predict(X_test)
    y_prob_rand = best_rf_rand.predict_proba(X_test)[:, 1]
    acc_rand = accuracy_score(y_test, y_pred_rand)
    roc_rand = roc_auc_score(y_test, y_prob_rand)

    # 6. Summary Comparison Table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON (MATCHED CV WORK)")
    print("=" * 80)
    print(f"{'Method':<24} | {'CV Calls':<10} | {'Time (s)':<10} | {'CV ROC-AUC':<12} | {'Test Acc':<10} | {'Test ROC-AUC':<12}")
    print("-" * 80)
    print(f"{'RandomizedSearchCV':<24} | {cv_calls:<10} | {duration_rand:<10.2f} | {rand_search.best_score_:<12.4f} | {acc_rand*100:<9.2f}% | {roc_rand:<12.4f}")
    print(f"{'GASearchCV (Early Stopped)':<24} | {cv_calls:<10} | {duration_ga:<10.2f} | {ga_search.best_score_:<12.4f} | {acc_ga*100:<9.2f}% | {roc_ga:<12.4f}")
    print("=" * 80)
    print("\nDetailed Classification Report (GASearchCV Best Model):")
    print(classification_report(y_test, y_pred_ga, target_names=data.target_names))

if __name__ == "__main__":
    main()
