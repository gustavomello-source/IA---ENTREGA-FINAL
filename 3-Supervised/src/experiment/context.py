"""
Defines :class:`PipelineContext`, that stores shared runtime state for an experiment,
including configuration, logger, experiment folder, selected stages, and artifacts
exchanged between stages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiment.config.config_reader import ConfigReader
from src.experiment.data.data_manager import DataManager


@dataclass
class PipelineContext:
    """
    Shared runtime state for an experiment pipeline.

    Attributes:
        config (ConfigReader): Configuration reader for the experiment.
        logger (Any): Logger instance for logging messages.
        experiment_folder (Path): Path to the experiment folder.
        data_manager (DataManager): Data manager for handling the dataset.
    """

    config: ConfigReader
    logger: Any
    data_manager: DataManager
    experiment_folder: Path
