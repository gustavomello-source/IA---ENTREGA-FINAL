```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Optimization script for XGBoost hyperparameters using RandomizedSearchCV.

It loads the dataset, applies preprocessing and PCA (if enabled),
calculates the scale_pos_weight, and performs hyperparameter optimization.

The best parameters are saved, and a comparison with the baseline is generated,
including a boxplot of F1-macro scores.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.experiment.config.config_reader import ConfigReader
from src.experiment.data_handling.dimensionality_reducer import DimensionalityReducer
from src.experiment.data_handling.preprocessor import Preprocessor


# Configs

EXPERIMENT_NAME = "experiment_20260827_170446"

OUTPUT_DIR = Path("optimize_xgboost_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("XGBoost Hyperparameter Optimization")
    print("=" * 70)

    print("\n[1/8] Loading configuration...")

    config_reader = ConfigReader()
    config_reader.read_config("./config.ini")

    data_config = config_reader.get_section("DATA")
    data_path = data_config.get("data_path")
    target_col = data_config.get("target_column")

    print(
        f"[2/8] Loading dataset and experiment IDs "
        f"from experiment {EXPERIMENT_NAME}..."
    )

    df = pd.read_csv(data_path)

    exp_path = Path("report") / EXPERIMENT_NAME / "data_ids"

    train_ids = pd.read_csv(exp_path / "train_ids.csv")["id"].values
    test_ids = pd.read_csv(exp_path / "test_ids.csv")["id"].values

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]

    y_train = y.loc[train_ids]
    y_test = y.loc[test_ids]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print(
        "[3/8] Applying preprocessing "
        "(fit on training data, transform on test data)..."
    )

    preprocessing_config = config_reader.get_section("PREPROCESSING")

    preprocessor = Preprocessor(config=preprocessing_config)

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    print(
        f"  After preprocessing: "
        f"Train {X_train_processed.shape}, "
        f"Test {X_test_processed.shape}"
    )

    pca_config = config_reader.get_section("PCA")
    reducer = DimensionalityReducer(config=pca_config)

    if reducer.enabled:
        print("[4/8] Applying PCA...")

        X_train_processed = reducer.fit_transform(X_train_processed)
        X_test_processed = reducer.transform(X_test_processed)

        print(
            f"  After PCA: "
            f"Train {X_train_processed.shape}, "
            f"Test {X_test_processed.shape}"
        )
    else:
        print("[4/8] PCA disabled, skipping...")

    print("[5/8] Calculating scale_pos_weight (auto)...")

    counts = y_train.value_counts()

    n_positive = int(counts.get(1, 0))
    n_negative = int(y_train.shape[0] - n_positive)

    scale_pos_weight = (
        float(n_negative / n_positive)
        if n_positive > 0
        else 1.0
    )

    print(
        f"  scale_pos_weight: {scale_pos_weight:.4f} "
        f"(neg={n_negative}, pos={n_positive})"
    )

    print("[6/8] Running RandomizedSearchCV (50 iterations, 3-fold CV)...")
    print("  This may take approximately 15-20 minutes...")

    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 1.5, 2, 5],
    }

    base_estimator = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=1,
        n_jobs=-1,
        eval_metric="logloss",
    )

    random_search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_macro",
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    random_search.fit(X_train_processed, y_train)

    best_params = random_search.best_params_

    print("\n  Best parameters found:")

    for k, v in best_params.items():
        print(f"    {k}: {v}")

    print(
        f"  Best F1-macro (CV): "
        f"{random_search.best_score_:.4f}"
    )

    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results.to_csv(
        OUTPUT_DIR / "gridsearch_results.csv",
        index=False,
    )

    print("\n[7/8] Running 10 runs with the best parameters...")

    optimized_runs_dir = OUTPUT_DIR / "optimized_runs"
    optimized_runs_dir.mkdir(exist_ok=True)

    optimized_f1_scores = []

    for run in range(1, 11):
        params = {
            **best_params,
            "scale_pos_weight": scale_pos_weight,
            "random_state": run,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)

        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro",
        )

        optimized_f1_scores.append(f1_macro)

        metrics = {
            "run": run,
            "seed": run,
            "f1_macro": f1_macro,
            "params": params,
        }

        with open(
            optimized_runs_dir / f"run_{run}.json",
            "w",
        ) as f:
            json.dump(
                metrics,
                f,
                indent=2,
                default=str,
            )

        print(
            f"  Run {run}/10: "
            f"F1-macro = {f1_macro:.4f}"
        )

    optimized_mean = np.mean(optimized_f1_scores)
    optimized_std = np.std(
        optimized_f1_scores,
        ddof=1,
    )

    print("\n[8/8] Loading baseline and generating comparison...")

    baseline_csv = (
        Path("report")
        / EXPERIMENT_NAME
        / "comparison"
        / "metrics_per_run.csv"
    )

    baseline_df = pd.read_csv(baseline_csv)

    baseline_xgb = baseline_df[
        baseline_df["model"] == "XGBoost"
    ]

    baseline_f1_scores = baseline_xgb["f1_macro"].values

    baseline_mean = np.mean(baseline_f1_scores)
    baseline_std = np.std(
        baseline_f1_scores,
        ddof=1,
    )

    improvement = (
        (optimized_mean - baseline_mean)
        / baseline_mean
    ) * 100

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Baseline:   F1-macro = "
        f"{baseline_mean:.4f} +/- {baseline_std:.4f}"
    )

    print(
        f"Optimized:  F1-macro = "
        f"{optimized_mean:.4f} +/- {optimized_std:.4f}"
    )

    print(f"Improvement: {improvement:+.2f}%")

    print("=" * 70)

    with open(
        OUTPUT_DIR / "optimization_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "XGBoost Hyperparameter Optimization Report\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            "Baseline Configuration (from config.ini):\n"
        )
        f.write("  n_estimators: 100\n")
        f.write("  max_depth: 6\n")
        f.write("  learning_rate: 0.1\n")
        f.write("  subsample: 0.8\n")
        f.write("  colsample_bytree: 0.8\n\n")

        f.write(
            "Baseline Performance (10 runs):\n"
        )
        f.write(
            f"  F1-macro: "
            f"{baseline_mean:.4f} +/- {baseline_std:.4f}\n\n"
        )

        f.write("Optimized Configuration:\n")

        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")

        f.write(
            "\nOptimized Performance (10 runs):\n"
        )
        f.write(
            f"  F1-macro: "
            f"{optimized_mean:.4f} +/- {optimized_std:.4f}\n\n"
        )

        f.write(
            f"Improvement: {improvement:+.2f}%\n"
        )

    comparison_data = pd.DataFrame(
        {
            "F1-macro": (
                list(baseline_f1_scores)
                + optimized_f1_scores
            ),
            "Model": (
                ["Baseline"] * len(baseline_f1_scores)
                + ["Optimized"] * len(optimized_f1_scores)
            ),
        }
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=comparison_data,
        x="Model",
        y="F1-macro",
        palette=["lightblue", "lightgreen"],
    )

    sns.stripplot(
        data=comparison_data,
        x="Model",
        y="F1-macro",
        color="black",
        alpha=0.5,
        size=6,
    )

    plt.title(
        "XGBoost: Baseline vs Optimized Hyperparameters",
        fontsize=14,
        fontweight="bold",
    )

    plt.ylabel(
        "F1-macro Score",
        fontsize=12,
    )

    plt.xlabel(
        "Configuration",
        fontsize=12,
    )

    for i, (label, scores) in enumerate(
        [
            ("Baseline", baseline_f1_scores),
            ("Optimized", optimized_f1_scores),
        ]
    ):
        mean_val = np.mean(scores)

        plt.text(
            i,
            mean_val,
            f"μ={mean_val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "xgboost_optimization_comparison.png",
        dpi=150,
    )

    print(
        f"\nFiles saved to: "
        f"{OUTPUT_DIR.absolute()}"
    )

    print("  - best_params.json")
    print("  - gridsearch_results.csv")
    print("  - optimized_runs/ (10 JSON files)")
    print("  - xgboost_optimization_comparison.png")
    print("  - optimization_report.txt")


if __name__ == "__main__":
    main()
```
