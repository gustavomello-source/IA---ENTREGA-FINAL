"""
Abstract base class for supervised learning models in the experiment pipeline.

The :class:`BaseModel` enforces a consistent interface (fit, predict, save/load)
across all concrete model implementations, allowing the comparison stage to
treat models uniformly. Subclasses implement only the estimator-specific logic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base for all model implementations.

    Concrete subclasses implement :meth:`_build_estimator` to construct the
    underlying scikit-learn (or compatible) estimator. The base class handles
    fit/predict delegation, logging, and persistence.

    Attributes:
        MODEL_NAME (str | None): Class-level registration name for the model.
            Subclasses must override this to participate in auto-discovery.
            Leave as ``None`` for abstract intermediate classes.
        name (str): Human-readable model name (e.g., ``"RandomForest"``).
        config (dict[str, Any]): Configuration mapping for the model.
        logger (Any): Optional logger for progress messages.
        random_state (int): Random seed for reproducibility.
        fitted_estimator_ (Any): The underlying fitted estimator, set after
            :meth:`fit`.
    """

    MODEL_NAME: str | None = None  # Subclasses override for auto-discovery

    def __init__(
        self, name: str | None = None, config: dict[str, Any] = None, logger: Any = None
    ) -> None:
        """
        Initialize the base model.

        Args:
            name (str | None): Model name. If ``None``, uses the class-level
                ``MODEL_NAME`` attribute.
            config (dict[str, Any]): Configuration mapping, typically a section
                from ``config.ini`` (e.g., ``[RANDOMFOREST]``).
            logger (Any): Optional logger for progress messages.
        """
        self.name: str = name if name is not None else (self.MODEL_NAME or "UnnamedModel")
        self.config: dict[str, Any] = config or {}
        self.logger: Any = logger
        self.random_state: int = int(self.config.get("random_state", 1))
        self.fitted_estimator_: Any = None

    @abstractmethod
    def _build_estimator(self) -> Any:
        """
        Construct the underlying estimator from the configuration.

        Subclasses implement this to return a scikit-learn (or compatible)
        estimator instance configured from ``self.config``.

        Returns:
            Any: Unfitted estimator instance.
        """

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Fit the model on the training data.

        Args:
            x (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            BaseModel: The fitted instance.
        Raises:
            Exception: If fitting fails.
        """
        try:
            self._log(f"Fitting {self.name} on {x.shape[0]} samples...")
            self.fitted_estimator_ = self._build_estimator()
            self.fitted_estimator_.fit(x, y)
            self._log(f"{self.name} fitted successfully.")
        except Exception as e:
            print(f"Error fitting {self.name}: {e}")
            raise
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Generate class predictions for the input features.

        Args:
            x (pd.DataFrame): Feature frame.

        Returns:
            np.ndarray: Predicted class labels.
        Raises:
            RuntimeError: If called before :meth:`fit`.
            Exception: If prediction fails.
        """
        if self.fitted_estimator_ is None:
            raise RuntimeError(f"{self.name} must be fitted before calling predict.")
        try:
            return self.fitted_estimator_.predict(x)
        except Exception as e:
            print(f"Error predicting with {self.name}: {e}")
            raise

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """
        Generate class probability estimates for the input features.

        Args:
            x (pd.DataFrame): Feature frame.

        Returns:
            np.ndarray: Predicted class probabilities, shape
            ``(n_samples, n_classes)``.
        Raises:
            RuntimeError: If called before :meth:`fit`.
            Exception: If prediction fails.
        """
        if self.fitted_estimator_ is None:
            raise RuntimeError(
                f"{self.name} must be fitted before calling predict_proba."
            )
        try:
            return self.fitted_estimator_.predict_proba(x)
        except Exception as e:
            print(f"Error predicting probabilities with {self.name}: {e}")
            raise

    def save(self, path: Path) -> None:
        """
        Persist the fitted model to disk using joblib.

        Args:
            path (Path): Output file path (e.g., ``models/RandomForest.joblib``).
        Raises:
            RuntimeError: If called before :meth:`fit`.
            Exception: If saving fails.
        """
        if self.fitted_estimator_ is None:
            raise RuntimeError(f"{self.name} must be fitted before saving.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.fitted_estimator_, path)
            self._log(f"{self.name} saved to {path}.")
        except Exception as e:
            print(f"Error saving {self.name}: {e}")
            raise

    def load(self, path: Path) -> "BaseModel":
        """
        Load a fitted model from disk.

        Args:
            path (Path): Path to the saved model file.

        Returns:
            BaseModel: The instance with the loaded estimator.
        Raises:
            Exception: If loading fails.
        """
        try:
            self.fitted_estimator_ = joblib.load(path)
            self._log(f"{self.name} loaded from {path}.")
        except Exception as e:
            print(f"Error loading {self.name}: {e}")
            raise
        return self

    def get_params(self) -> dict[str, Any]:
        """
        Return the fitted estimator's parameters.

        Returns:
            dict[str, Any]: Parameter dictionary.
        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.fitted_estimator_ is None:
            raise RuntimeError(
                f"{self.name} must be fitted before calling get_params."
            )
        return self.fitted_estimator_.get_params()
