"""
Random Forest classifier implementation for the experiment pipeline.
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.experiment.models.base_model import BaseModel


def _parse_max_depth(raw: str | int | None) -> int | None:
    """
    Parse the max_depth configuration into an int or None.

    Args:
        raw (str | int | None): Raw configuration value.

    Returns:
        int | None: Parsed max_depth value.
    """
    if raw is None or str(raw).strip().lower() == "none":
        return None
    return int(raw)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier wrapper.

    Reads configuration from the ``[RANDOMFOREST]`` section and builds a
    ``RandomForestClassifier`` with the specified hyperparameters.
    """

    MODEL_NAME = "RandomForest"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the Random Forest model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[RANDOMFOREST]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)

    def _build_estimator(self) -> RandomForestClassifier:
        """
        Construct a RandomForestClassifier from the configuration.

        Returns:
            RandomForestClassifier: Configured estimator.
        """
        return RandomForestClassifier(
            n_estimators=int(self.config.get("n_estimators", 100)),
            max_depth=_parse_max_depth(self.config.get("max_depth", None)),
            min_samples_split=int(self.config.get("min_samples_split", 2)),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 1)),
            max_features=str(self.config.get("max_features", "sqrt")),
            class_weight=str(self.config.get("class_weight", "balanced")),
            random_state=self.random_state,
            n_jobs=int(self.config.get("n_jobs", -1)),
        )
