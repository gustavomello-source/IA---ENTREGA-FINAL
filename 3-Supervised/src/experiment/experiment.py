"""
Experiment orchestration for the pipeline.
"""

from src.experiment.context import PipelineContext
from src.experiment.data.dimensionality_reducer import DimensionalityReducer
from src.experiment.data.preprocessor import Preprocessor
from src.experiment.metrics import resolve_minority_class


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
