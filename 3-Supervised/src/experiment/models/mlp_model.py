"""
Multi-Layer Perceptron (MLP) classifier implementation for the experiment pipeline.
"""

from typing import Any

from sklearn.neural_network import MLPClassifier

from src.experiment.models.base_model import BaseModel


def _parse_hidden_layer_sizes(raw: str | None) -> tuple[int, ...]:
    """
    Parse the hidden_layer_sizes configuration into a tuple of integers.

    Args:
        raw (str | None): Raw configuration value (e.g., "100, 50").

    Returns:
        tuple[int, ...]: Parsed hidden layer sizes.
    """
    if raw is None or str(raw).strip() == "":
        return (100,)
    return tuple(int(x.strip()) for x in str(raw).split(","))


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


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron (MLP) classifier wrapper.

    Reads configuration from the ``[MLP]`` section and builds an
    ``MLPClassifier`` with the specified hyperparameters. This is a
    feed-forward neural network suitable for tabular classification tasks.
    """

    MODEL_NAME = "MLP"

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        """
        Initialize the MLP model.

        Args:
            config (dict[str, Any]): Configuration mapping from the
                ``[MLP]`` section of ``config.ini``.
            logger (Any): Optional logger for progress messages.
        """
        super().__init__(config=config, logger=logger)

    def _build_estimator(self) -> MLPClassifier:
        """
        Construct an MLPClassifier from the configuration.

        Returns:
            MLPClassifier: Configured estimator.
        """
        return MLPClassifier(
            hidden_layer_sizes=_parse_hidden_layer_sizes(
                self.config.get("hidden_layer_sizes", "100, 50")
            ),
            activation=str(self.config.get("activation", "relu")),
            alpha=float(self.config.get("alpha", 0.0001)),
            learning_rate_init=float(self.config.get("learning_rate_init", 0.001)),
            max_iter=int(self.config.get("max_iter", 200)),
            early_stopping=_parse_bool(self.config.get("early_stopping", True)),
            random_state=self.random_state,
        )
