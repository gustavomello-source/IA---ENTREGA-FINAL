"""
Visualization module for multi-run model comparison.

The :class:`ComparisonPlots` generates boxplots and bar charts to visualize
metric distributions across multiple training runs for each model.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ComparisonPlots:
    """
    Generate comparison plots for multi-run model evaluation.

    Attributes:
        output_folder (Path): Directory where plots are saved.
        logger (Any): Optional logger for progress messages.
    """

    def __init__(self, output_folder: Path, logger: Any = None) -> None:
        """
        Initialize the plotting module.

        Args:
            output_folder (Path): Output directory (e.g.,
                ``report/experiment_[timestamp]/comparison/plots/``).
            logger (Any): Optional logger for progress messages.
        """
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def plot_metric_boxplot(
        self,
        per_run_data: dict[str, list[dict[str, Any]]],
        metric_name: str,
        metric_label: str,
        output_filename: str,
    ) -> None:
        """
        Create a boxplot comparing models for a single metric.

        Args:
            per_run_data (dict[str, list[dict]]): Per-run metrics for each model.
            metric_name (str): Metric key in the run dictionaries.
            metric_label (str): Human-readable label for the y-axis.
            output_filename (str): Output filename (e.g., "f1_macro_boxplot.png").
        """
        # Flatten data into long format for seaborn
        data_rows = []
        for model_name, runs in per_run_data.items():
            for run_metrics in runs:
                if metric_name in run_metrics:
                    data_rows.append(
                        {"Model": model_name, metric_label: run_metrics[metric_name]}
                    )

        if not data_rows:
            self._log(f"No data for metric '{metric_name}'. Skipping plot.")
            return

        df = pd.DataFrame(data_rows)

        # Create boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x="Model",
            y=metric_label,
            hue="Model",
            palette="Set2",
            legend=False,
        )
        plt.title(
            f"{metric_label} Distribution Across Runs", fontsize=14, weight="bold"
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric_label, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = self.output_folder / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._log(f"Boxplot saved: {output_path}")

    def plot_timing_comparison(
        self, per_run_data: dict[str, list[dict[str, Any]]]
    ) -> None:
        """
        Create side-by-side boxplots for train_time and predict_time.

        Args:
            per_run_data (dict[str, list[dict]]): Per-run metrics for each model.
        """
        # Flatten data
        train_rows = []
        predict_rows = []
        for model_name, runs in per_run_data.items():
            for run_metrics in runs:
                if "train_time" in run_metrics:
                    train_rows.append(
                        {"Model": model_name, "Time (s)": run_metrics["train_time"]}
                    )
                if "predict_time" in run_metrics:
                    predict_rows.append(
                        {"Model": model_name, "Time (s)": run_metrics["predict_time"]}
                    )

        if not train_rows and not predict_rows:
            self._log("No timing data available. Skipping timing plots.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Train time boxplot
        if train_rows:
            df_train = pd.DataFrame(train_rows)
            sns.boxplot(
                data=df_train,
                x="Model",
                y="Time (s)",
                hue="Model",
                ax=axes[0],
                palette="Set2",
                legend=False,
            )
            axes[0].set_title("Training Time Distribution", fontsize=13, weight="bold")
            axes[0].set_xlabel("Model", fontsize=11)
            axes[0].set_ylabel("Time (s)", fontsize=11)
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(axis="y", alpha=0.3, linestyle="--")

        # Predict time boxplot
        if predict_rows:
            df_predict = pd.DataFrame(predict_rows)
            sns.boxplot(
                data=df_predict,
                x="Model",
                y="Time (s)",
                hue="Model",
                ax=axes[1],
                palette="Set2",
                legend=False,
            )
            axes[1].set_title(
                "Prediction Time Distribution", fontsize=13, weight="bold"
            )
            axes[1].set_xlabel("Model", fontsize=11)
            axes[1].set_ylabel("Time (s)", fontsize=11)
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        output_path = self.output_folder / "timing_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._log(f"Timing comparison saved: {output_path}")

    def generate_all_plots(self, per_run_data: dict[str, list[dict[str, Any]]]) -> None:
        """
        Generate all comparison plots for the available metrics.

        Args:
            per_run_data (dict[str, list[dict]]): Per-run metrics for each model.
        """
        self._log("Generating comparison plots...")

        # Performance and error metrics
        metrics_to_plot = [
            ("f1_macro", "F1 Score (Macro)", "f1_macro_boxplot.png"),
            ("f1_minority", "F1 Score (Minority)", "f1_minority_boxplot.png"),
            ("accuracy", "Accuracy", "accuracy_boxplot.png"),
            (
                "precision_minority",
                "Precision (Minority)",
                "precision_minority_boxplot.png",
            ),
            ("recall_minority", "Recall (Minority)", "recall_minority_boxplot.png"),
            ("roc_auc", "ROC-AUC", "roc_auc_boxplot.png"),
            ("false_positives", "False Positives", "false_positives_boxplot.png"),
            ("false_negatives", "False Negatives", "false_negatives_boxplot.png"),
            ("total_errors", "Total Errors", "total_errors_boxplot.png"),
        ]

        for metric_name, metric_label, filename in metrics_to_plot:
            self.plot_metric_boxplot(per_run_data, metric_name, metric_label, filename)

        # Timing comparison
        self.plot_timing_comparison(per_run_data)

        self._log(f"All comparison plots saved to {self.output_folder}.")
