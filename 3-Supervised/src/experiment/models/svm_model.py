"""
Support Vector Machine (SVM) classifier implementation for the experiment pipeline.
"""

from typing import Any

from sklearn.svm import SVC

from src.experiment.models.base_model import BaseModel


class SVMModel(BaseModel):
    """
    Support Vector Machine (SVM) classifier wrapper.

    Reads configuration from the ``[SVM]`` section and builds an
    ``SVC`` with the specified hyperparameters.

    Note:
        SVM with RBF kernel scales poorly on large datasets. Training on
        175k samples may take several minutes.
    """

    MODEL_NAME = "SVM"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the SVM model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[SVM]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)

    def _build_estimator(self) -> SVC:
        """
        Construct an SVC from the configuration.

        Returns:
            SVC: Configured estimator.
        """
        return SVC(
            C=float(self.config.get("C", 1.0)),
            kernel=str(self.config.get("kernel", "rbf")),
            gamma=str(self.config.get("gamma", "scale")),
            class_weight=str(self.config.get("class_weight", "balanced")),
            probability=True,  # Enable predict_proba for evaluation
            random_state=self.random_state,
        )
