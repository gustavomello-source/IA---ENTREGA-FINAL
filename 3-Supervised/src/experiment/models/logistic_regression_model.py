"""
Logistic Regression classifier implementation for the experiment pipeline.
"""

from typing import Any

from sklearn.linear_model import LogisticRegression

from src.experiment.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier wrapper.

    Reads configuration from the ``[LOGISTICREGRESSION]`` section and builds a
    ``LogisticRegression`` with the specified hyperparameters.
    """

    MODEL_NAME = "LogisticRegression"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the Logistic Regression model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[LOGISTICREGRESSION]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)

    def _build_estimator(self) -> LogisticRegression:
        """
        Construct a LogisticRegression from the configuration.

        Returns:
            LogisticRegression: Configured estimator.
        """
        return LogisticRegression(
            C=float(self.config.get("C", 1.0)),
            max_iter=int(self.config.get("max_iter", 1000)),
            class_weight=str(self.config.get("class_weight", "balanced")),
            solver=str(self.config.get("solver", "lbfgs")),
            random_state=self.random_state,
            n_jobs=int(self.config.get("n_jobs", -1)),
        )
