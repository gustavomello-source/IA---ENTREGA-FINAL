"""
AdaBoost classifier implementation for the experiment pipeline.
"""

from typing import Any

from sklearn.ensemble import AdaBoostClassifier

from src.experiment.models.base_model import BaseModel


class AdaBoostModel(BaseModel):
    """
    AdaBoost classifier wrapper.

    Reads configuration from the ``[ADABOOST]`` section and builds an
    ``AdaBoostClassifier`` with the specified hyperparameters.
    """

    MODEL_NAME = "AdaBoost"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the AdaBoost model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[ADABOOST]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)

    def _build_estimator(self) -> AdaBoostClassifier:
        """
        Construct an AdaBoostClassifier from the configuration.

        Returns:
            AdaBoostClassifier: Configured estimator.
        """
        return AdaBoostClassifier(
            n_estimators=int(self.config.get("n_estimators", 100)),
            learning_rate=float(self.config.get("learning_rate", 1.0)),
            random_state=self.random_state,
        )
