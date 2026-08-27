"""
Experiment orchestration for the pipeline.
"""

import json
import time

import numpy as np

from src.experiment.context import PipelineContext
from src.experiment.data.dimensionality_reducer import DimensionalityReducer
from src.experiment.data.preprocessor import Preprocessor
from src.experiment.evaluation.comparison_plots import ComparisonPlots
from src.experiment.evaluation.comparison_report import ComparisonReport
from src.experiment.evaluation.evaluator import ModelEvaluator
from src.experiment.metrics import resolve_minority_class
from src.experiment.models.model_factory import create_model, list_available_models


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
            f"Primary metric: {metric_config.get('primary', 'f1_macro')} "
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

        Reads the ``[MODEL] models`` list and ``n_runs`` parameter, validates
        each name against the auto-discovered registry, trains each model
        n_runs times with seeds 1..n_runs, measures training time, saves each
        run to a per-run folder, and stores all runs in the shared context.

        Models without a corresponding ``[MODELNAME]`` config section are skipped.
        """
        logger = self.context.logger
        config = self.context.config
        data_manager = self.context.data_manager

        # Discover available models
        available_models = list_available_models()
        logger.info(f"Discovered models: {', '.join(available_models)}.")

        # Parse configured model list
        model_config = config.get_section("MODEL")
        model_names_raw = model_config.get("models", "")
        model_names = [
            name.strip() for name in model_names_raw.split(",") if name.strip()
        ]

        if not model_names:
            logger.warning("No models configured in [MODEL] models. Skipping training.")
            return

        # Validate each configured name against the discovered registry
        invalid_names = [name for name in model_names if name not in available_models]
        if invalid_names:
            logger.error(
                f"Invalid model names in [MODEL] models: {', '.join(invalid_names)}. "
                f"Available models: {', '.join(available_models)}."
            )
            raise ValueError(
                f"Unknown models: {invalid_names}. Available: {available_models}"
            )

        # Number of runs per model
        n_runs = int(model_config.get("n_runs", 1))
        logger.info(
            f"Training {len(model_names)} model(s) with {n_runs} run(s) each: "
            f"{', '.join(model_names)}."
        )

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

                # Initialize run list for this model
                self.context.model_runs[model_name] = []

                for run in range(1, n_runs + 1):
                    logger.info(
                        f"Training {model_name}: run {run}/{n_runs} (seed={run})..."
                    )

                    # Override random_state with run number
                    run_config = dict(model_cfg)
                    run_config["random_state"] = str(run)

                    # Create and fit model
                    model = create_model(model_name, config=run_config, logger=logger)

                    start_time = time.perf_counter()
                    model.fit(self.context.X_train_processed, data_manager.y_train)
                    train_time = time.perf_counter() - start_time

                    # Save to run-specific folder
                    model_folder = self.context.experiment_folder / model_name
                    run_folder = model_folder / f"run_{run}"
                    run_folder.mkdir(parents=True, exist_ok=True)
                    model.save(run_folder / f"{model_name}.joblib")

                    # Store run info
                    self.context.model_runs[model_name].append(
                        {
                            "run": run,
                            "seed": run,
                            "model": model,
                            "train_time": train_time,
                        }
                    )

                    logger.info(
                        f"{model_name} run {run} completed in {train_time:.2f}s."
                    )

                # For backward compatibility, store first run in fitted_models
                if self.context.model_runs[model_name]:
                    self.context.fitted_models[model_name] = self.context.model_runs[
                        model_name
                    ][0]["model"]

            except Exception as exc:
                logger.error(f"Failed to train {model_name}: {exc}", exc_info=True)
                raise

        total_runs = sum(len(runs) for runs in self.context.model_runs.values())
        logger.info(
            f"Model training complete. Total runs: {total_runs} "
            f"({len(self.context.model_runs)} models × {n_runs} runs)."
        )

    def _compare_models(self) -> None:
        """
        Evaluate all model runs on the test split and generate comparison reports.

        For each model, evaluates every run (measuring prediction time), collects
        per-run metrics, and computes aggregated mean/std statistics. Confusion
        matrices and error analysis are produced from the first run of each model
        as a representative sample. Aggregated metrics, per-run tables, and
        distribution plots (boxplots) are saved to the comparison folder.
        """

        logger = self.context.logger
        data_manager = self.context.data_manager

        if not self.context.model_runs:
            logger.warning("No model runs to compare. Skipping comparison.")
            return

        logger.info(f"Comparing {len(self.context.model_runs)} model(s) on test set...")

        comparison_folder = self.context.experiment_folder / "comparison"
        comparison_folder.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator(
            minority_class=self.context.minority_class, logger=logger
        )
        report = ComparisonReport(output_folder=comparison_folder, logger=logger)

        for model_name, runs in self.context.model_runs.items():
            logger.info(f"Evaluating {model_name} ({len(runs)} run(s))...")

            run_metrics_list: list[dict] = []
            model_folder = self.context.experiment_folder / model_name
            model_folder.mkdir(parents=True, exist_ok=True)

            for run_info in runs:
                run = run_info["run"]
                model = run_info["model"]

                # Compute metrics with prediction timing
                start_time = time.perf_counter()
                metrics = evaluator.evaluate(
                    model=model,
                    X_test=self.context.X_test_processed,
                    y_test=data_manager.y_test,
                )
                predict_time = time.perf_counter() - start_time

                # Attach run-level info
                metrics["run"] = run
                metrics["seed"] = run_info["seed"]
                metrics["train_time"] = run_info["train_time"]
                metrics["predict_time"] = predict_time
                run_metrics_list.append(metrics)

                # Save per-run metrics JSON
                run_folder = model_folder / f"run_{run}"
                run_folder.mkdir(parents=True, exist_ok=True)
                with (run_folder / "metrics.json").open("w") as f:
                    json.dump(metrics, f, indent=2, default=str)

            # Register all runs for aggregation
            report.add_model_runs(model_name, run_metrics_list)

            # Produce representative artifacts from the first run
            first_metrics = run_metrics_list[0]

            # Save classification report
            with (model_folder / "classification_report.json").open("w") as f:
                json.dump(first_metrics["classification_report"], f, indent=2)

            # Save confusion matrices
            cm = np.array(first_metrics["confusion_matrix"])
            cm_norm = np.array(first_metrics["confusion_matrix_normalized"])
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
                cm=first_metrics["confusion_matrix"],
                labels=first_metrics["labels"],
                model_name=model_name,
                output_path=model_folder / "confusion_matrix.png",
            )

            # Error analysis
            first_model = runs[0]["model"]
            error_analysis = evaluator.analyze_errors(
                model=first_model,
                X_test=self.context.X_test_processed,
                y_test=data_manager.y_test,
            )
            error_analysis["false_positives"].to_csv(
                model_folder / "errors_false_positives.csv", index=False
            )
            error_analysis["false_negatives"].to_csv(
                model_folder / "errors_false_negatives.csv", index=False
            )

        # Aggregate comparison (mean/std tables + per-run table)
        report.save()

        # Generate distribution plots
        plots_folder = comparison_folder / "plots"
        plotter = ComparisonPlots(output_folder=plots_folder, logger=logger)
        plotter.generate_all_plots(report.per_run_data)

        best_model, best_score = report.get_best_model(metric="f1_macro_mean")
        logger.info(
            f"Comparison complete. Best model: {best_model} "
            f"(F1 macro mean = {best_score:.4f})."
        )

        # Store for later stages
        self.context.best_model_name = best_model
        self.context.comparison_metrics = report.metrics_table
