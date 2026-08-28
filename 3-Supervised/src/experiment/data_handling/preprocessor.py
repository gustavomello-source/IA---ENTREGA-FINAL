"""
Configurable preprocessing for the supervised-learning experiment.

The :class:`Preprocessor` learns every transformation on the training data
only and applies it to any other split, which keeps the train/test evaluation
free of leakage. All behaviour is driven by the ``[PREPROCESSING]`` section of
``config.ini`` so different pipelines can be compared experimentally without
code changes.
"""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


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


class Preprocessor:
    """
    Configurable, leakage-safe preprocessing pipeline.

    The transformations are learned in :meth:`fit` (training data only) and
    replayed in :meth:`transform`. The concrete behaviour is controlled by a
    configuration mapping, typically the ``[PREPROCESSING]`` section of
    ``config.ini``.

    Attributes:
        id_column (str): Identifier column removed before preprocessing.
        sentinel_values (list[float]): Placeholder codes replaced with NaN.
        drop_near_constant (bool): Whether to drop near-constant columns.
        near_constant_threshold (float): Single-value fraction that marks a
            column as near-constant.
        drop_high_missing (bool): Whether to drop columns with too many
            missing values.
        high_missing_threshold (float): Missing fraction that marks a column
            for removal.
        imputation_strategy (str): Strategy passed to ``SimpleImputer``.
        add_missing_indicator (bool): Whether to append missing-indicator
            columns for numeric features.
        one_hot_encode (bool): Whether low-cardinality columns are one-hot
            encoded. When ``False`` they are treated as numeric.
        low_cardinality_max (int): Maximum unique training values for a column
            to be treated as low-cardinality (categorical).
        scaler (str): Scaler applied to numeric features
            (``standard`` | ``robust`` | ``none``).
        logger (Any): Optional logger used for progress messages.
        kept_columns_ (list[str]): Columns retained after the drop steps.
        low_cardinality_columns_ (list[str]): Columns routed to one-hot
            encoding.
        numeric_columns_ (list[str]): Columns routed to imputation + scaling.
        pipeline_ (Pipeline | None): Fitted scikit-learn pipeline.
        feature_names_ (list[str]): Output feature names after transformation.
    """

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Build a :class:`Preprocessor` from a configuration mapping.

        Args:
            config (dict[str, Any]): Preprocessing configuration, typically the
                ``[PREPROCESSING]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        self.logger = logger

        self.id_column: str = str(config.get("id_column", "ID")).strip()

        self.sentinel_values: list[float] = self._parse_sentinels(
            config.get("sentinel_values", "")
        )

        self.drop_near_constant: bool = _to_bool(
            config.get("drop_near_constant", True), default=True
        )
        self.near_constant_threshold: float = float(
            config.get("near_constant_threshold", 0.99)
        )

        self.drop_high_missing: bool = _to_bool(
            config.get("drop_high_missing", True), default=True
        )
        self.high_missing_threshold: float = float(
            config.get("high_missing_threshold", 0.95)
        )

        self.imputation_strategy: str = str(
            config.get("imputation_strategy", "median")
        ).strip()
        self.add_missing_indicator: bool = _to_bool(
            config.get("add_missing_indicator", False), default=False
        )

        self.one_hot_encode: bool = _to_bool(
            config.get("one_hot_encode", True), default=True
        )
        self.low_cardinality_max: int = int(config.get("low_cardinality_max", 10))

        self.scaler: str = str(config.get("scaler", "standard")).strip().lower()

        # Learned state.
        self.kept_columns_: list[str] = []
        self.low_cardinality_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.pipeline_: Pipeline | None = None
        self.feature_names_: list[str] = []

    @staticmethod
    def _parse_sentinels(raw: str | list[float]) -> list[float]:
        """
        Parse the configured sentinel values into a list of floats.

        Args:
            raw (str | list[float]): Comma-separated string or list of codes.

        Returns:
            list[float]: Parsed sentinel values (empty when none configured).
        """
        if isinstance(raw, list):
            return [float(v) for v in raw]
        if not raw:
            return []
        return [float(part.strip()) for part in str(raw).split(",") if part.strip()]

    def _log(self, message: str) -> None:
        """
        Emit an info-level message when a logger is available.

        Args:
            message (str): Message to log.
        """
        if self.logger is not None:
            self.logger.info(message)

    def _make_scaler(self) -> Any:
        """
        Build the scaler instance for numeric features.

        Returns:
            Any: A scikit-learn scaler, or ``"passthrough"`` when disabled.
        Raises:
            ValueError: If an unknown scaler name is configured.
        """
        if self.scaler in {"none", ""}:
            return "passthrough"
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "robust":
            return RobustScaler()
        raise ValueError(
            f"Unknown scaler '{self.scaler}'. Use 'standard', 'robust' or 'none'."
        )

    def _replace_sentinels(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Replace configured sentinel codes with NaN.

        Args:
            frame (pd.DataFrame): Feature frame.

        Returns:
            pd.DataFrame: Frame with sentinels replaced by NaN.
        """
        if not self.sentinel_values:
            return frame
        return frame.mask(frame.isin(self.sentinel_values))

    def _prepare_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the identifier column and replace sentinel codes.

        Args:
            x (pd.DataFrame): Raw feature frame.

        Returns:
            pd.DataFrame: Cleaned feature frame.
        """
        frame = x.copy()
        if self.id_column and self.id_column in frame.columns:
            frame = frame.drop(columns=[self.id_column])
        return self._replace_sentinels(frame)

    def _select_columns(self, frame: pd.DataFrame) -> None:
        """
        Decide which columns to keep and how to route them.

        The decisions (near-constant removal, high-missing removal, and
        low-cardinality vs numeric routing) are made on the training data only
        and stored for reuse in :meth:`transform`.

        Args:
            frame (pd.DataFrame): Cleaned training feature frame.
        """
        columns = list(frame.columns)
        dropped_near_constant: list[str] = []
        dropped_high_missing: list[str] = []

        if self.drop_high_missing:
            missing_fraction = frame.isna().mean()
            dropped_high_missing = missing_fraction[
                missing_fraction >= self.high_missing_threshold
            ].index.tolist()

        if self.drop_near_constant:
            for column in columns:
                if column in dropped_high_missing:
                    continue
                top_fraction = (
                    frame[column].value_counts(normalize=True, dropna=False).iloc[0]
                )
                if top_fraction >= self.near_constant_threshold:
                    dropped_near_constant.append(column)

        to_drop = set(dropped_high_missing) | set(dropped_near_constant)
        self.kept_columns_ = [c for c in columns if c not in to_drop]

        low_cardinality: list[str] = []
        numeric: list[str] = []
        for column in self.kept_columns_:
            unique_count = frame[column].nunique(dropna=True)
            if self.one_hot_encode and unique_count <= self.low_cardinality_max:
                low_cardinality.append(column)
            else:
                numeric.append(column)

        self.low_cardinality_columns_ = low_cardinality
        self.numeric_columns_ = numeric

        self._log(
            "Preprocessing column selection: "
            f"{len(columns)} input, "
            f"{len(dropped_near_constant)} near-constant dropped, "
            f"{len(dropped_high_missing)} high-missing dropped, "
            f"{len(self.kept_columns_)} kept "
            f"({len(low_cardinality)} one-hot, {len(numeric)} numeric)."
        )

    def _build_pipeline(self) -> Pipeline:
        """
        Assemble the ColumnTransformer pipeline from the learned routing.

        Returns:
            Pipeline: Unfitted preprocessing pipeline.
        """
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.imputation_strategy,
                        add_indicator=self.add_missing_indicator,
                    ),
                ),
                ("scaler", self._make_scaler()),
            ]
        )

        transformers: list[tuple[str, Any, list[str]]] = []
        if self.numeric_columns_:
            transformers.append(("numeric", numeric_pipeline, self.numeric_columns_))

        if self.low_cardinality_columns_:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_pipeline, self.low_cardinality_columns_)
            )

        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        return Pipeline(steps=[("columns", column_transformer)])

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "Preprocessor":
        """
        Learn the preprocessing steps from the training data.

        Args:
            x (pd.DataFrame): Training feature frame (may include the ID column).
            y (pd.Series | None): Training target (unused, kept for API symmetry).

        Returns:
            Preprocessor: The fitted instance.
        Raises:
            Exception: If fitting fails.
        """
        try:
            frame = self._prepare_features(x)
            self._select_columns(frame)
            self.pipeline_ = self._build_pipeline()
            self.pipeline_.fit(frame[self.kept_columns_], y)
            self.feature_names_ = list(
                self.pipeline_.named_steps["columns"].get_feature_names_out()
            )
            self._log(
                f"Preprocessor fitted. Output width: {len(self.feature_names_)}."
            )
        except Exception as e:
            print(f"Error fitting preprocessor: {e}")
            raise
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned preprocessing to a feature frame.

        Args:
            x (pd.DataFrame): Feature frame to transform.

        Returns:
            pd.DataFrame: Transformed features with learned column names.
        Raises:
            RuntimeError: If called before :meth:`fit`.
            Exception: If transformation fails.
        """
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        try:
            frame = self._prepare_features(x)
            transformed = self.pipeline_.transform(frame[self.kept_columns_])
            return pd.DataFrame(
                transformed,
                columns=self.feature_names_,
                index=x.index,
            )
        except Exception as e:
            print(f"Error transforming data: {e}")
            raise

    def fit_transform(
        self, x: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Fit on the training data and return the transformed training features.

        Args:
            x (pd.DataFrame): Training feature frame.
            y (pd.Series | None): Training target (unused).

        Returns:
            pd.DataFrame: Transformed training features.
        """
        return self.fit(x, y).transform(x)
