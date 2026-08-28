"""
Configurable PCA dimensionality reduction for the experiment pipeline.

The :class:`DimensionalityReducer` follows the same fit-on-train /
apply-to-test contract as :class:`~src.experiment.data_handling.preprocessor.Preprocessor`,
so it plugs in as a toggleable stage after preprocessing and before model
fitting. Behaviour is driven by the ``[PCA]`` section of ``config.ini`` so the
comparison stage can evaluate pipelines with and without PCA.
"""

from typing import Any

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _to_bool(value: str | bool, default: bool = False) -> bool:
    """
    Parse a configuration flag into a boolean.

    Args:
        value (str | bool): Raw configuration value.
        default (bool): Value returned when parsing is not possible.

    Returns:
        bool: Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_n_components(raw: str | int | float) -> int | float:
    """
    Parse the ``n_components`` setting into an int or float.

    scikit-learn's ``PCA`` treats a float in ``(0, 1)`` as a target
    explained-variance ratio and an integer as an exact component count. This
    helper preserves that distinction: values containing a decimal point are
    returned as ``float`` and whole numbers as ``int``.

    Args:
        raw (str | int | float): Raw configuration value.

    Returns:
        int | float: Parsed component specification.
    """
    if isinstance(raw, (int, float)):
        return raw
    text = str(raw).strip()
    value = float(text)
    if value.is_integer() and "." not in text:
        return int(value)
    return value


class DimensionalityReducer:
    """
    Configurable PCA dimensionality reduction.

    The transformation is learned in :meth:`fit` (training data only) and
    replayed in :meth:`transform`, keeping the train/test evaluation free of
    leakage. When ``standardize_before`` is enabled, a ``StandardScaler`` is
    fitted ahead of PCA so every input feature (including one-hot columns)
    shares a common scale.

    Attributes:
        enabled (bool): Master toggle. When ``False``, ``transform`` returns
            its input unchanged.
        n_components (int | float): Component count (int) or target
            explained-variance ratio (float in ``(0, 1)``).
        standardize_before (bool): Whether to standardize features before PCA.
        svd_solver (str): SVD solver passed to ``PCA``.
        whiten (bool): Whether to whiten the components.
        random_state (int): Random seed for reproducibility.
        logger (Any): Optional logger used for progress messages.
        pipeline_ (Pipeline | None): Fitted scaler + PCA pipeline.
        n_components_ (int): Number of components retained after fitting.
        explained_variance_ratio_ (list[float]): Per-component explained
            variance ratios.
        cumulative_variance_ (float): Total variance retained by the
            components.
        feature_names_ (list[str]): Output component names (``pc1``..``pcK``).
    """

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Build a :class:`DimensionalityReducer` from a configuration mapping.

        Args:
            config (dict[str, Any]): PCA configuration, typically the ``[PCA]``
                section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        self.logger = logger

        self.enabled: bool = _to_bool(config.get("enabled", True), default=True)
        self.n_components: int | float = _parse_n_components(
            config.get("n_components", 0.95)
        )
        self.standardize_before: bool = _to_bool(
            config.get("standardize_before", True), default=True
        )
        self.svd_solver: str = str(config.get("svd_solver", "auto")).strip().lower()
        self.whiten: bool = _to_bool(config.get("whiten", False), default=False)
        self.random_state: int = int(config.get("random_state", 1))

        # Learned state.
        self.pipeline_: Pipeline | None = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: list[float] = []
        self.cumulative_variance_: float = 0.0
        self.feature_names_: list[str] = []

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def _build_pipeline(self) -> Pipeline:
        """
        Assemble the optional scaler + PCA pipeline.

        Returns:
            Pipeline: Unfitted dimensionality-reduction pipeline.
        """
        steps: list[tuple[str, Any]] = []
        if self.standardize_before:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "pca",
                PCA(
                    n_components=self.n_components,
                    svd_solver=self.svd_solver,
                    whiten=self.whiten,
                    random_state=self.random_state,
                ),
            )
        )
        return Pipeline(steps=steps)

    def fit(
        self, x: pd.DataFrame, y: pd.Series | None = None
    ) -> "DimensionalityReducer":
        """
        Learn the PCA projection from the training data.

        Args:
            x (pd.DataFrame): Preprocessed training features.
            y (pd.Series | None): Training target (unused, kept for API symmetry).

        Returns:
            DimensionalityReducer: The fitted instance.
        Raises:
            Exception: If fitting fails.
        """
        if not self.enabled:
            self._log("PCA disabled; dimensionality reduction skipped.")
            return self
        try:
            self.pipeline_ = self._build_pipeline()
            self.pipeline_.fit(x, y)

            pca: PCA = self.pipeline_.named_steps["pca"]
            self.explained_variance_ratio_ = pca.explained_variance_ratio_.tolist()
            self.n_components_ = int(pca.n_components_)
            self.cumulative_variance_ = float(sum(self.explained_variance_ratio_))
            self.feature_names_ = [f"pc{i + 1}" for i in range(self.n_components_)]
            self._log(
                f"PCA fitted: {x.shape[1]} -> {self.n_components_} components "
                f"({self.cumulative_variance_:.4f} variance retained)."
            )
        except Exception as e:
            print(f"Error fitting dimensionality reducer: {e}")
            raise
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Project a feature frame onto the learned principal components.

        Args:
            x (pd.DataFrame): Preprocessed features to project.

        Returns:
            pd.DataFrame: Component scores (``pc1``..``pcK``) with the input
            index. When PCA is disabled, the input frame is returned unchanged.
        Raises:
            RuntimeError: If called before :meth:`fit` while enabled.
            Exception: If transformation fails.
        """
        if not self.enabled:
            return x
        if self.pipeline_ is None:
            raise RuntimeError(
                "DimensionalityReducer must be fitted before calling transform."
            )
        try:
            projected = self.pipeline_.transform(x)
            return pd.DataFrame(
                projected,
                columns=self.feature_names_,
                index=x.index,
            )
        except Exception as e:
            print(f"Error reducing dimensionality: {e}")
            raise

    def fit_transform(
        self, x: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Fit on the training data and return the projected training features.

        Args:
            x (pd.DataFrame): Preprocessed training features.
            y (pd.Series | None): Training target (unused).

        Returns:
            pd.DataFrame: Projected training features (or the input unchanged
            when PCA is disabled).
        """
        return self.fit(x, y).transform(x)
