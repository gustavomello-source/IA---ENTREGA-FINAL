"""
Model evaluation for the experiment pipeline.

The :class:`ModelEvaluator` computes comprehensive metrics, confusion matrices,
and error analysis for a single model on the test set, with outputs saved to
per-model folders under the report directory.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.experiment.models.base_model import BaseModel


class ModelEvaluator:
    """
    Evaluate a single model on the test set.

    Computes standard classification metrics, confusion matrix, and identifies
    all misclassified samples for error analysis.

    Attributes:
        minority_class (Any): Label treated as the positive/minority class.
        logger (Any): Optional logger for progress messages.
    """

    def __init__(self, minority_class: Any, logger: Any = None) -> None:
        """
        Initialize the evaluator.

        Args:
            minority_class (Any): Minority class label for F1 computation.
            logger (Any): Optional logger for progress messages.
        """
        self.minority_class = minority_class
        self.logger = logger

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def evaluate(
        self,
        model: BaseModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """
        Compute comprehensive metrics for a model on the test set.

        Args:
            model (BaseModel): Fitted model instance.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.

        Returns:
            dict[str, Any]: Dictionary containing metrics, confusion matrix,
            and classification report.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Identify majority class (used for binary metrics)
        labels = sorted(y_test.unique())
        majority_class = labels[0] if labels[0] != self.minority_class else labels[1]

        # Compute metrics
        accuracy = float(accuracy_score(y_test, y_pred))

        precision_minority = float(
            precision_score(
                y_test, y_pred, pos_label=self.minority_class, zero_division=0
            )
        )
        recall_minority = float(
            recall_score(y_test, y_pred, pos_label=self.minority_class, zero_division=0)
        )
        f1_minority = float(
            f1_score(y_test, y_pred, pos_label=self.minority_class, zero_division=0)
        )

        precision_macro = float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # ROC-AUC
        minority_idx = list(labels).index(self.minority_class)
        roc_auc = float(roc_auc_score(y_test, y_proba[:, minority_idx]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_normalized = confusion_matrix(
            y_test, y_pred, labels=labels, normalize="true"
        )

        # Classification report
        report = classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        # Error counts
        errors = y_pred != y_test
        false_positives = int(
            ((y_pred == majority_class) & (y_test == self.minority_class)).sum()
        )
        false_negatives = int(
            ((y_pred == self.minority_class) & (y_test == majority_class)).sum()
        )

        return {
            "model_name": model.name,
            "accuracy": accuracy,
            "precision_minority": precision_minority,
            "recall_minority": recall_minority,
            "f1_minority": f1_minority,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_normalized.tolist(),
            "classification_report": report,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_errors": int(errors.sum()),
            "labels": labels,
        }

    def plot_confusion_matrix(
        self,
        cm: np.ndarray | list,
        labels: list,
        model_name: str,
        output_path: Path,
    ) -> None:
        """
        Generate and save a confusion matrix heatmap.

        Args:
            cm (np.ndarray | list): Confusion matrix (raw counts).
            labels (list): Class labels in the same order as the matrix.
            model_name (str): Model name for the plot title.
            output_path (Path): Path to save the PNG file.
        """
        cm_array = np.array(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        self._log(f"Confusion matrix plot saved to {output_path}.")

    def analyze_errors(
        self,
        model: BaseModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        """
        Identify all misclassified test samples for error analysis.

        Splits errors into false positives (predicted minority, actual
        majority) and false negatives (predicted majority, actual minority).
        Each returned frame includes the original test index, true and
        predicted labels, per-class probabilities, and the full set of PCA
        component values for the sample.

        Args:
            model (BaseModel): Fitted model instance.
            X_test (pd.DataFrame): Test features (PCA components).
            y_test (pd.Series): Test target.

        Returns:
            dict[str, pd.DataFrame]: Mapping with keys ``"false_positives"``
            and ``"false_negatives"``.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        labels = sorted(y_test.unique())
        majority_class = labels[0] if labels[0] != self.minority_class else labels[1]

        base = pd.DataFrame(index=X_test.index)
        base.insert(0, "id", X_test.index)
        base["true_label"] = y_test.to_numpy()
        base["predicted_label"] = y_pred
        for class_idx, label in enumerate(labels):
            base[f"proba_class_{label}"] = y_proba[:, class_idx]
        # Attach all PCA component values
        features = X_test.reset_index(drop=True)
        features.index = X_test.index
        base = pd.concat([base, features], axis=1)

        y_pred_series = pd.Series(y_pred, index=X_test.index)

        fp_mask = (y_pred_series == self.minority_class) & (y_test == majority_class)
        fn_mask = (y_pred_series == majority_class) & (y_test == self.minority_class)

        false_positives = base.loc[fp_mask].reset_index(drop=True)
        false_negatives = base.loc[fn_mask].reset_index(drop=True)

        self._log(
            f"{model.name} error analysis: "
            f"{len(false_positives)} false positives, "
            f"{len(false_negatives)} false negatives."
        )

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
