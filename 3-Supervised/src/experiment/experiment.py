"""
Experiment orchestration for the pipeline.
"""

from src.experiment.context import PipelineContext
from src.experiment.data.dimensionality_reducer import DimensionalityReducer
from src.experiment.data.preprocessor import Preprocessor
from src.experiment.evaluation.comparison_report import ComparisonReport
from src.experiment.evaluation.evaluator import ModelEvaluator
from src.experiment.metrics import resolve_minority_class
from src.experiment.models.model_factory import create_model


class Experiment:
    """
    Class to manage the experiment pipeline.
    Orchestrates the experiment lifecycle.

    Attributes:
        context (PipelineContext): Shared runtime state for the experiment.
    """

    def __init__(self, context: PipelineContext) -> None:
        """
        Initialize the experiment with a shared pipeline context.

        Args:
            context (PipelineContext): Shared runtime state.
        """
        self.context: PipelineContext = context

    def run(self) -> None:
        """
        Execute the pipeline stages in canonical order.
        """
        logger = self.context.logger
        logger.info("Starting experiment pipeline.")

        try:
            self._preprocess()
            self._reduce_dimensionality()
            self._train_models()
            self._compare_models()
            logger.info("Experiment pipeline completed successfully.")

        except Exception as exc:
            logger.error(f"Experiment pipeline failed: {exc}", exc_info=True)
            raise

    def _preprocess(self) -> None:
        """
        Fit the configurable preprocessor on the training split and transform
        both splits.

        The fitted preprocessor and the transformed matrices are stored on the
        shared context for reuse.
        """
        logger = self.context.logger
        config = self.context.config
        data_manager = self.context.data_manager

        preprocessing_config = config.get_section("PREPROCESSING")
        metric_config = config.get_section("METRIC")

        logger.info("Fitting preprocessor on the training split.")
        preprocessor = Preprocessor(config=preprocessing_config, logger=logger)
        x_train_processed = preprocessor.fit_transform(
            data_manager.X_train, data_manager.y_train
        )
        x_test_processed = preprocessor.transform(data_manager.X_test)

        minority_class = resolve_minority_class(
            data_manager.y_train, metric_config.get("minority_class", "auto")
        )
        logger.info(
            f"Primary metric: {metric_config.get('primary', 'f1_minority')} "
            f"(minority class = {minority_class})."
        )

        self.context.preprocessor = preprocessor
        self.context.X_train_processed = x_train_processed
        self.context.X_test_processed = x_test_processed
        self.context.minority_class = minority_class

        logger.info(
            "Preprocessing complete. "
            f"Train shape: {x_train_processed.shape}, "
            f"Test shape: {x_test_processed.shape}."
        )

    def _reduce_dimensionality(self) -> None:
        """
        Fit the configurable PCA reducer on the preprocessed training split and
        project both splits.
        """
        logger = self.context.logger
        config = self.context.config

        pca_config = config.get_section("PCA")
        reducer = DimensionalityReducer(config=pca_config, logger=logger)

        if not reducer.enabled:
            logger.info("PCA stage disabled; keeping preprocessed features.")
            self.context.dimensionality_reducer = reducer
            return

        logger.info("Fitting PCA on the preprocessed training split.")
        x_train_reduced = reducer.fit_transform(self.context.X_train_processed)
        x_test_reduced = reducer.transform(self.context.X_test_processed)

        self.context.dimensionality_reducer = reducer
        self.context.X_train_processed = x_train_reduced
        self.context.X_test_processed = x_test_reduced
        self.context.pca_explained_variance = reducer.cumulative_variance_

        logger.info(
            "Dimensionality reduction complete. "
            f"Train shape: {x_train_reduced.shape}, "
            f"Test shape: {x_test_reduced.shape}, "
            f"variance retained: {reducer.cumulative_variance_:.4f}."
        )

    def _train_models(self) -> None:
        """
        Train all configured models on the processed training split.

        Reads the ``[MODEL] models`` list, instantiates each via the factory,
        fits on the processed training data, saves the fitted model into a
        per-model folder under the report directory, and stores fitted
        instances on the shared context for the comparison stage.
        """
        logger = self.context.logger
        config = self.context.config
        data_manager = self.context.data_manager

        model_config = config.get_section("MODEL")
        model_names_raw = model_config.get("models", "")
        model_names = [
            name.strip() for name in model_names_raw.split(",") if name.strip()
        ]

        if not model_names:
            logger.warning("No models configured in [MODEL] models. Skipping training.")
            return

        logger.info(f"Training {len(model_names)} model(s): {', '.join(model_names)}.")

        for model_name in model_names:
            try:
                model_section_name = model_name.upper()
                model_cfg = config.get_section(model_section_name)
                if not model_cfg:
                    logger.warning(
                        f"No [{model_section_name}] config section found. "
                        f"Skipping {model_name}."
                    )
                    continue

                model = create_model(model_name, config=model_cfg, logger=logger)
                model.fit(self.context.X_train_processed, data_manager.y_train)
                model_folder = self.context.experiment_folder / model_name
                model_folder.mkdir(parents=True, exist_ok=True)
                model.save(model_folder / f"{model_name}.joblib")
                self.context.fitted_models[model_name] = model

            except Exception as exc:
                logger.error(f"Failed to train {model_name}: {exc}", exc_info=True)
                raise

        logger.info(
            f"Model training complete. Fitted models: {list(self.context.fitted_models.keys())}."
        )

    def _compare_models(self) -> None:
        """
        Evaluate all fitted models on the test split and generate comparison reports.

        For each model, computes metrics, plots confusion matrices, performs
        error analysis, and saves all outputs into per-model folders. Then
        aggregates metrics into a cross-model comparison table saved to the
        comparison folder.
        """
        logger = self.context.logger
        data_manager = self.context.data_manager

        if not self.context.fitted_models:
            logger.warning("No fitted models to compare. Skipping comparison.")
            return

        logger.info(
            f"Comparing {len(self.context.fitted_models)} model(s) on test set..."
        )

        comparison_folder = self.context.experiment_folder / "comparison"
        comparison_folder.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator(
            minority_class=self.context.minority_class, logger=logger
        )
        report = ComparisonReport(output_folder=comparison_folder, logger=logger)

        for model_name, model in self.context.fitted_models.items():
            logger.info(f"Evaluating {model_name}...")

            # Compute metrics
            metrics = evaluator.evaluate(
                model=model,
                X_test=self.context.X_test_processed,
                y_test=data_manager.y_test,
            )
            report.add_model_metrics(model_name, metrics)

            # Per-model folder
            model_folder = self.context.experiment_folder / model_name
            model_folder.mkdir(parents=True, exist_ok=True)

            # Save metrics JSON
            import json

            with (model_folder / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2, default=str)

            # Save classification report
            with (model_folder / "classification_report.json").open("w") as f:
                json.dump(metrics["classification_report"], f, indent=2)

            # Save confusion matrices
            import numpy as np

            cm = np.array(metrics["confusion_matrix"])
            cm_norm = np.array(metrics["confusion_matrix_normalized"])
            np.savetxt(
                model_folder / "confusion_matrix.csv", cm, delimiter=",", fmt="%d"
            )
            np.savetxt(
                model_folder / "confusion_matrix_normalized.csv",
                cm_norm,
                delimiter=",",
                fmt="%.4f",
            )

            # Plot confusion matrix
            evaluator.plot_confusion_matrix(
                cm=metrics["confusion_matrix"],
                labels=metrics["labels"],
                model_name=model_name,
                output_path=model_folder / "confusion_matrix.png",
            )

            # Error analysis
            error_analysis = evaluator.analyze_errors(
                model=model,
                X_test=self.context.X_test_processed,
                y_test=data_manager.y_test,
            )
            error_analysis["false_positives"].to_csv(
                model_folder / "errors_false_positives.csv", index=False
            )
            error_analysis["false_negatives"].to_csv(
                model_folder / "errors_false_negatives.csv", index=False
            )

        # Aggregate comparison
        report.save()
        best_model, best_score = report.get_best_model(metric="f1_macro")
        logger.info(
            f"Comparison complete. Best model: {best_model} (F1 macro = {best_score:.4f})."
        )

        # Store for later stages
        self.context.best_model_name = best_model
        self.context.comparison_metrics = report.metrics_table
