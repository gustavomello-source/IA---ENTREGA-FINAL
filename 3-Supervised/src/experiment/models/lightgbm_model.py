"""
LightGBM classifier implementation for the experiment pipeline.
"""

from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from src.experiment.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM classifier wrapper.

    Reads configuration from the ``[LIGHTGBM]`` section and builds an
    ``LGBMClassifier`` with the specified hyperparameters. Supports
    ``scale_pos_weight = auto`` to auto-compute class imbalance weighting
    from the training target.
    """

    MODEL_NAME = "LightGBM"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the LightGBM model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[LIGHTGBM]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)
        self._scale_pos_weight: float | str = self.config.get(
            "scale_pos_weight", "auto"
        )

    def _build_estimator(self) -> LGBMClassifier:
        """
        Construct an LGBMClassifier from the configuration.

        Returns:
            LGBMClassifier: Configured estimator.
        """
        return LGBMClassifier(
            n_estimators=int(self.config.get("n_estimators", 100)),
            learning_rate=float(self.config.get("learning_rate", 0.1)),
            num_leaves=int(self.config.get("num_leaves", 31)),
            max_depth=int(self.config.get("max_depth", -1)),
            scale_pos_weight=1.0,  # Placeholder; overridden in fit
            random_state=self.random_state,
            n_jobs=int(self.config.get("n_jobs", -1)),
            verbose=int(self.config.get("verbose", -1)),
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """
        Fit the LightGBM model on the training data.

        When ``scale_pos_weight = auto``, the ratio is computed from the
        training target as ``n_majority / n_minority`` and set on the
        estimator before fitting.

        Args:
            x (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            LightGBMModel: The fitted instance.
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
