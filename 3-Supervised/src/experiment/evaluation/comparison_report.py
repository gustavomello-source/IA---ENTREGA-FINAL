"""
Comparison report generation for the experiment pipeline.

The :class:`ComparisonReport` aggregates metrics across all model runs,
computes mean/std statistics, builds comparison tables, identifies the best
model, and writes summary artifacts to the report folder.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


class ComparisonReport:
    """
    Aggregate and report model comparison results from multiple runs.

    Attributes:
        output_folder (Path): Directory where comparison artifacts are saved.
        logger (Any): Optional logger for progress messages.
        per_run_data (dict[str, list[dict]]): Per-run metrics for each model.
        metrics_table (pd.DataFrame | None): Aggregated comparison table (mean/std).
    """

    def __init__(self, output_folder: Path, logger: Any = None) -> None:
        """
        Initialize the comparison report.

        Args:
            output_folder (Path): Output directory (e.g.,
                ``report/experiment_[timestamp]/comparison/``).
            logger (Any): Optional logger for progress messages.
        """
        self.output_folder = output_folder
        self.logger = logger
        self.per_run_data: dict[str, list[dict[str, Any]]] = {}
        self.metrics_table: pd.DataFrame | None = None

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def add_model_runs(self, model_name: str, run_metrics: list[dict[str, Any]]) -> None:
        """
        Add all runs for a model to the comparison.

        Args:
            model_name (str): Model name.
            run_metrics (list[dict[str, Any]]): List of metrics dicts, one per run.
        """
        self.per_run_data[model_name] = run_metrics

    def _compute_aggregated_metrics(
        self, run_metrics: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Compute mean and std across runs for numeric metrics.

        Args:
            run_metrics (list[dict[str, Any]]): List of per-run metric dicts.

        Returns:
            dict[str, float]: Aggregated metrics with keys like
                ``accuracy_mean``, ``accuracy_std``, etc.
        """
        # Metrics to aggregate
        metric_keys = [
            "accuracy",
            "precision_minority",
            "recall_minority",
            "f1_minority",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
            "false_positives",
            "false_negatives",
            "total_errors",
            "train_time",
            "predict_time",
        ]

        aggregated = {}
        for key in metric_keys:
            values = [m[key] for m in run_metrics if key in m]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)

        return aggregated

    def save(self) -> None:
        """
        Build aggregated and per-run tables, write all comparison artifacts.

        Writes:
        - ``comparison/metrics_table_aggregated.csv``: Mean ± std per model
        - ``comparison/metrics_per_run.csv``: Long-format table with all runs
        - ``comparison/summary.txt``: Text summary with best model
        """
        if not self.per_run_data:
            self._log("No model runs to compare. Skipping comparison save.")
            return

        # Build aggregated table (mean/std)
        aggregated_rows = []
        for model_name, run_metrics in self.per_run_data.items():
            agg = self._compute_aggregated_metrics(run_metrics)
            agg["model"] = model_name
            agg["n_runs"] = len(run_metrics)
            aggregated_rows.append(agg)

        self.metrics_table = pd.DataFrame(aggregated_rows).set_index("model")

        # Build per-run table (long format)
        per_run_rows = []
        for model_name, run_metrics in self.per_run_data.items():
            for metrics in run_metrics:
                row = {
                    "model": model_name,
                    "run": metrics["run"],
                    "seed": metrics["seed"],
                    "accuracy": metrics["accuracy"],
                    "precision_minority": metrics["precision_minority"],
                    "recall_minority": metrics["recall_minority"],
                    "f1_minority": metrics["f1_minority"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "roc_auc": metrics["roc_auc"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    "total_errors": metrics["total_errors"],
                    "train_time": metrics["train_time"],
                    "predict_time": metrics["predict_time"],
                }
                per_run_rows.append(row)

        per_run_table = pd.DataFrame(per_run_rows)

        # Save aggregated table
        agg_csv_path = self.output_folder / "metrics_table_aggregated.csv"
        self.metrics_table.to_csv(agg_csv_path)
        self._log(f"Aggregated metrics table saved to {agg_csv_path}.")

        # Save per-run table
        per_run_csv_path = self.output_folder / "metrics_per_run.csv"
        per_run_table.to_csv(per_run_csv_path, index=False)
        self._log(f"Per-run metrics table saved to {per_run_csv_path}.")

        # Identify best model by primary metric (f1_macro_mean)
        best_model, best_score = self.get_best_model(metric="f1_macro_mean")

        # Write summary
        summary_path = self.output_folder / "summary.txt"
        with summary_path.open("w") as f:
            f.write("Model Comparison Summary (Multi-Run)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Model (by F1 macro mean): {best_model}\n")
            f.write(f"  F1 Score (macro, mean): {best_score:.4f}\n\n")
            f.write("Per-Model Aggregated Metrics (mean ± std):\n")
            f.write("-" * 60 + "\n")
            for model_idx, row in self.metrics_table.iterrows():
                f.write(f"\nModel: {model_idx} ({int(row['n_runs'])} runs)\n")
                f.write(
                    f"  Accuracy:           {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}\n"
                )
                f.write(
                    f"  Precision (minor):  {row['precision_minority_mean']:.4f} ± {row['precision_minority_std']:.4f}\n"
                )
                f.write(
                    f"  Recall (minor):     {row['recall_minority_mean']:.4f} ± {row['recall_minority_std']:.4f}\n"
                )
                f.write(
                    f"  F1 (minor):         {row['f1_minority_mean']:.4f} ± {row['f1_minority_std']:.4f}\n"
                )
                f.write(
                    f"  F1 (macro):         {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f}\n"
                )
                f.write(
                    f"  ROC-AUC:            {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}\n"
                )
                f.write(
                    f"  Train Time (s):     {row['train_time_mean']:.2f} ± {row['train_time_std']:.2f}\n"
                )
                f.write(
                    f"  Predict Time (s):   {row['predict_time_mean']:.4f} ± {row['predict_time_std']:.4f}\n"
                )
                f.write(
                    f"  False Positives:    {row['false_positives_mean']:.1f} ± {row['false_positives_std']:.1f}\n"
                )
                f.write(
                    f"  False Negatives:    {row['false_negatives_mean']:.1f} ± {row['false_negatives_std']:.1f}\n"
                )
                f.write(
                    f"  Total Errors:       {row['total_errors_mean']:.1f} ± {row['total_errors_std']:.1f}\n"
                )
        self._log(f"Comparison summary saved to {summary_path}.")

    def get_best_model(self, metric: str = "f1_macro_mean") -> tuple[str, float]:
        """
        Identify the best model by a given metric.

        Args:
            metric (str): Metric name (column in the metrics table).

        Returns:
            tuple[str, float]: Best model name and its metric value.
        Raises:
            ValueError: If no metrics have been added.
        """
        if self.metrics_table is None or self.metrics_table.empty:
            raise ValueError("No model metrics available for comparison.")
        best_idx = self.metrics_table[metric].idxmax()
        best_value = float(self.metrics_table.loc[best_idx, metric])
        return str(best_idx), best_value
