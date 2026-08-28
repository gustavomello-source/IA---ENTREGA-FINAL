"""
Naive Bayes classifier implementation for the experiment pipeline.
"""

from typing import Any

from sklearn.naive_bayes import GaussianNB

from src.experiment.models.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """
    Gaussian Naive Bayes classifier wrapper.

    Reads configuration from the ``[NAIVEBAYES]`` section and builds a
    ``GaussianNB`` with the specified hyperparameters.
    """

    MODEL_NAME = "NaiveBayes"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the Naive Bayes model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[NAIVEBAYES]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)

    def _build_estimator(self) -> GaussianNB:
        """
        Construct a GaussianNB from the configuration.

        Returns:
            GaussianNB: Configured estimator.
        """
        return GaussianNB(
            var_smoothing=float(self.config.get("var_smoothing", 1e-9)),
        )
