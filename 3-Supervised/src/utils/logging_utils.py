"""
Logging utility methods for the experiment pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def create_experiment_folder(report_path: str = "./report/") -> tuple[Path, str]:
    """
    Create a timestamped experiment folder under ``report_path``.

    Args:
        report_path (str): Base directory in which to create the run folder.

    Returns:
        tuple[Path, str]: The created experiment folder path and the timestamp
        string (``YYYYmmdd_HHMMSS``) used to name it.
    Raises:
        Exception: If the folder cannot be created.
    """
    try:
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder: Path = Path(report_path) / f"experiment_{timestamp}"
        experiment_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating experiment folder: {e}")
        raise
    return experiment_folder, timestamp


def setup_logging(report_path: str = "./report/") -> tuple[logging.Logger, Path]:
    """
    Configure logging for training pipeline information.

    Create a unique experiment folder with a timestamp and set up logging
    for both console output and a log file within that folder.

    Args:
        report_path (str): Base path for reports and logs.

    Returns:
        tuple[logging.Logger, Path]: Configured logger instance and
        the experiment folder path as a Path object.
    Raises:
        Exception: If logging setup fails.
    """
    try:
        experiment_folder, _ = create_experiment_folder(report_path)
        logs_folder = experiment_folder / "logs"
        logs_folder.mkdir(parents=True, exist_ok=True)

        log_filename = logs_folder / "experiment.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )

        logger = logging.getLogger(__name__)
        logger.info("Starting experiment.")
        logger.info(f"Experiment folder: {experiment_folder}")
        logger.info(f"Log saved to: {log_filename}")

    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise

    return logger, experiment_folder
