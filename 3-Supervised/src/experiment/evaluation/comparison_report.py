"""
Comparison report generation for the experiment pipeline.

The :class:`ComparisonReport` aggregates metrics across all fitted models,
builds a comparison table, identifies the best model, and writes summary
artifacts to the report folder.
"""

from pathlib import Path
from typing import Any

import pandas as pd


class ComparisonReport:
    """
    Aggregate and report model comparison results.

    Attributes:
        output_folder (Path): Directory where comparison artifacts are saved.
        logger (Any): Optional logger for progress messages.
        metrics_data (list[dict]): Collected metrics for all models.
        metrics_table (pd.DataFrame | None): Comparison table (built on save).
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
        self.metrics_data: list[dict[str, Any]] = []
        self.metrics_table: pd.DataFrame | None = None

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def add_model_metrics(self, model_name: str, metrics: dict[str, Any]) -> None:
        """
        Add a model's metrics to the comparison.

        Args:
            model_name (str): Model name.
            metrics (dict[str, Any]): Metrics dictionary from
                :meth:`ModelEvaluator.evaluate`.
        """
        self.metrics_data.append(metrics)

    def save(self) -> None:
        """
        Build the comparison table and write all comparison artifacts.

        Writes:
        - ``comparison/metrics_table.csv``
        - ``comparison/metrics_table.json``
        - ``comparison/summary.txt``
        """
        if not self.metrics_data:
            self._log("No model metrics to compare. Skipping comparison save.")
            return

        # Build comparison table
        table_rows = []
        for metrics in self.metrics_data:
            table_rows.append(
                {
                    "model": metrics["model_name"],
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
                }
            )
        self.metrics_table = pd.DataFrame(table_rows).set_index("model")

        # Save CSV and JSON
        csv_path = self.output_folder / "metrics_table.csv"
        json_path = self.output_folder / "metrics_table.json"
        self.metrics_table.to_csv(csv_path)
        self.metrics_table.to_json(json_path, orient="index", indent=2)
        self._log(f"Comparison table saved to {csv_path} and {json_path}.")

        # Identify best model by primary metric (f1_macro)
        best_model, best_score = self.get_best_model(metric="f1_macro")

        # Write summary
        summary_path = self.output_folder / "summary.txt"
        with summary_path.open("w") as f:
            f.write("Model Comparison Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Model (by F1 macro class): {best_model}\n")
            f.write(f"  F1 Score (macro): {best_score:.4f}\n\n")
            f.write("Per-Model Metrics:\n")
            f.write("-" * 50 + "\n")
            for _, row in self.metrics_table.iterrows():
                f.write(f"\nModel: {row.name}\n")
                f.write(f"  Accuracy:           {row['accuracy']:.4f}\n")
                f.write(f"  Precision (minor):  {row['precision_minority']:.4f}\n")
                f.write(f"  Recall (minor):     {row['recall_minority']:.4f}\n")
                f.write(f"  F1 (minor):         {row['f1_minority']:.4f}\n")
                f.write(f"  F1 (macro):         {row['f1_macro']:.4f}\n")
                f.write(f"  ROC-AUC:            {row['roc_auc']:.4f}\n")
                f.write(f"  False Positives:    {int(row['false_positives'])}\n")
                f.write(f"  False Negatives:    {int(row['false_negatives'])}\n")
                f.write(f"  Total Errors:       {int(row['total_errors'])}\n")
        self._log(f"Comparison summary saved to {summary_path}.")

    def get_best_model(self, metric: str = "f1_macro") -> tuple[str, float]:
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
