"""
CatBoost classifier implementation for the experiment pipeline.
"""

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from src.experiment.models.base_model import BaseModel


def _parse_bool(raw: str | bool | None, default: bool = False) -> bool:
    """
    Parse a boolean configuration value.

    Args:
        raw (str | bool | None): Raw configuration value.
        default (bool): Default value if raw is None or empty.

    Returns:
        bool: Parsed boolean.
    """
    if raw is None or str(raw).strip() == "":
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes"}


class CatBoostModel(BaseModel):
    """
    CatBoost classifier wrapper.

    Reads configuration from the ``[CATBOOST]`` section and builds a
    ``CatBoostClassifier`` with the specified hyperparameters. Supports
    ``scale_pos_weight = auto`` to auto-compute class imbalance weighting
    from the training target.
    """

    MODEL_NAME = "CatBoost"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the CatBoost model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[CATBOOST]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)
        self._scale_pos_weight: float | str = self.config.get(
            "scale_pos_weight", "auto"
        )

    def _build_estimator(self) -> CatBoostClassifier:
        """
        Construct a CatBoostClassifier from the configuration.

        Returns:
            CatBoostClassifier: Configured estimator.
        """
        return CatBoostClassifier(
            iterations=int(self.config.get("iterations", 100)),
            learning_rate=float(self.config.get("learning_rate", 0.1)),
            depth=int(self.config.get("depth", 6)),
            scale_pos_weight=1.0,  # Placeholder; overridden in fit
            random_state=self.random_state,
            thread_count=int(self.config.get("thread_count", -1)),
            verbose=_parse_bool(self.config.get("verbose", False)),
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        """
        Fit the CatBoost model on the training data.

        When ``scale_pos_weight = auto``, the ratio is computed from the
        training target as ``n_majority / n_minority`` and set on the
        estimator before fitting.

        Args:
            x (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            CatBoostModel: The fitted instance.
        Raises:
            Exception: If fitting fails.
        """
        try:
            self._log(f"Fitting {self.name} on {x.shape[0]} samples...")
            self.fitted_estimator_ = self._build_estimator()

            # Auto-compute scale_pos_weight if configured
            if str(self._scale_pos_weight).strip().lower() == "auto":
                counts = y.value_counts()
                n_positive = int(counts.get(1, 0))
                n_negative = int(y.shape[0] - n_positive)
                if n_positive == 0:
                    scale = 1.0
                else:
                    scale = float(n_negative / n_positive)
                self.fitted_estimator_.set_params(scale_pos_weight=scale)
                self._log(
                    f"{self.name} scale_pos_weight auto-computed: {scale:.4f} "
                    f"(negative={n_negative}, positive={n_positive})."
                )
            else:
                scale = float(self._scale_pos_weight)
                self.fitted_estimator_.set_params(scale_pos_weight=scale)

            self.fitted_estimator_.fit(x, y)
            self._log(f"{self.name} fitted successfully.")
        except Exception as e:
            print(f"Error fitting {self.name}: {e}")
            raise
        return self
