"""
Defines :class:`PipelineContext`, that stores shared runtime state for an experiment,
including configuration, logger, experiment folder, selected stages, and artifacts
exchanged between stages.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.experiment.config.config_reader import ConfigReader
from src.experiment.data_handling.data_manager import DataManager


@dataclass
class PipelineContext:
    """
    Shared runtime state for an experiment pipeline.

    Attributes:
        config (ConfigReader): Configuration reader for the experiment.
        logger (Any): Logger instance for logging messages.
        experiment_folder (Path): Path to the experiment folder.
        data_manager (DataManager): Data manager for handling the dataset.
        preprocessor (Any): Fitted preprocessor produced by the pipeline.
        X_train_processed (Any): Transformed training features.
        X_test_processed (Any): Transformed testing features.
        minority_class (Any): Minority class label for the primary metric.
        dimensionality_reducer (Any): Fitted PCA reducer produced by the pipeline.
        pca_explained_variance (Any): Total variance retained by the PCA
            components (``None`` when PCA is disabled).
        model_runs (dict[str, list[dict[str, Any]]]): Mapping of model names to
            lists of run dictionaries. Each run dict contains 'run', 'seed',
            'model', and 'train_time' keys.
        fitted_models (dict[str, Any]): Mapping of model names to fitted model
            instances (first run of each model, for backward compatibility).
        best_model_name (str | None): Name of the best model determined by the
            comparison stage.
        comparison_metrics (Any): Comparison table (DataFrame) from the
            comparison stage.
    """

    config: ConfigReader
    logger: Any
    data_manager: DataManager
    experiment_folder: Path
    preprocessor: Any = field(default=None)
    X_train_processed: Any = field(default=None)
    X_test_processed: Any = field(default=None)
    minority_class: Any = field(default=None)
    dimensionality_reducer: Any = field(default=None)
    pca_explained_variance: Any = field(default=None)
    model_runs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    fitted_models: dict[str, Any] = field(default_factory=dict)
    best_model_name: str | None = field(default=None)
    comparison_metrics: Any = field(default=None)


def save_context_snapshot(context: PipelineContext) -> None:
    """
    Save a snapshot of the current context to a file.

    Args:
        context (PipelineContext): The pipeline context to snapshot.
    """
    snapshot_path = context.experiment_folder / "context_snapshot.txt"
    with Path.open(snapshot_path, "w") as f:
        f.write(f"Configuration: {context.config}\n")
        f.write(f"Experiment Folder: {context.experiment_folder}\n")
        f.write(f"Data Manager: {context.data_manager}\n")
        f.write(f"Context: {context}\n")
