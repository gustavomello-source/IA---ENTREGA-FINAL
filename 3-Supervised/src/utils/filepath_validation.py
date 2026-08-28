"""Reusable path validation helpers for the pipeline."""

import logging
import os
from pathlib import Path


def validate_file_path(
    path: str,
    *,
    suffix: str | None = None,
    label: str = "Path",
) -> Path:
    """
    Validate that a path matches the expected file constraints.

    Args:
        path (str): Path to validate.
        suffix (str | None): Optional required suffix.
        label (str): Human-readable name used in error messages.

    Returns:
        Path: Validated path object.
    """
    path_obj = Path(path)

    if suffix and not path.endswith(suffix):
        logging.error(
            f"Invalid {label.lower()} path: {path}. Must end with '{suffix}'."
        )
        raise ValueError(
            f"Invalid {label.lower()} path: {path}. Must end with '{suffix}'."
        )

    if not path_obj.exists():
        logging.error(f"{label} not found at {path}.")
        raise FileNotFoundError(f"{label} not found at {path}")

    if not path_obj.is_file():
        logging.error(f"{label} at {path} is not a file.")
        raise ValueError(f"{label} at {path} is not a file.")

    return path_obj


def validate_directory_path(path: str, *, must_be_writable: bool = False) -> Path:
    """
    Validate that a path matches the expected directory constraints.

    Args:
        path (str): Path to validate.
        must_be_writable (bool): Whether the directory must be writable.

    Returns:
        Path: Validated directory object.
    Raises:
        Exception: If the directory cannot be validated.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        logging.warning(f"Directory not found at {path}. Creating it.")
        path_obj.mkdir(parents=True, exist_ok=True)

    if not path_obj.is_dir():
        logging.error(f"{path} is not a directory.")
        raise ValueError(f"{path} is not a directory.")

    if must_be_writable and not os.access(path, os.W_OK):
        logging.error(f"Directory {path} is not writable.")
        raise PermissionError(f"Directory {path} is not writable.")

    return path_obj
