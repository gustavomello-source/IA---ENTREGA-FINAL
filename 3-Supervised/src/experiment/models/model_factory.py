"""
Model factory with automatic discovery for the experiment pipeline.

Concrete models are discovered automatically by scanning this package for
:class:`BaseModel` subclasses that declare a ``MODEL_NAME`` class attribute.
Dropping a new model module into this package is enough to register it; no
manual edits to this factory are required.
"""

import importlib
import pkgutil
from functools import cache
from typing import Any

from src.experiment.models.base_model import BaseModel


def _import_all_model_modules(logger: Any = None) -> None:
    """
    Import every module in this package so model subclasses are defined.

    Args:
        logger (Any): Optional logger for warnings.
    """
    package = importlib.import_module("src.experiment.models")
    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = module_info.name
        # Skip this factory module and the base class module.
        if module_name in {"model_factory", "base_model"}:
            continue
        full_name = f"src.experiment.models.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:
            message = f"Skipping model module '{full_name}': {exc}"
            if logger is not None:
                logger.warning(message)
            else:
                print(f"Warning: {message}")


def _collect_subclasses(
    cls: type[BaseModel], registry: dict[str, type[BaseModel]]
) -> None:
    """
    Recursively collect ``BaseModel`` subclasses with a ``MODEL_NAME``.

    Args:
        cls (type[BaseModel]): Class whose subclasses are inspected.
        registry (dict[str, type[BaseModel]]): Registry to populate in place.
    """
    for subclass in cls.__subclasses__():
        model_name = getattr(subclass, "MODEL_NAME", None)
        if model_name:
            registry[model_name] = subclass
        _collect_subclasses(subclass, registry)


@cache
def get_model_registry() -> dict[str, type[BaseModel]]:
    """
    Build (and cache) the registry of all discovered models.

    Returns:
        dict[str, type[BaseModel]]: Mapping of model name to model class.
    """
    _import_all_model_modules()
    registry: dict[str, type[BaseModel]] = {}
    _collect_subclasses(BaseModel, registry)
    return registry


def list_available_models() -> list[str]:
    """
    Return the names of all discovered models.

    Returns:
        list[str]: Sorted list of registered model names.
    """
    return sorted(get_model_registry().keys())


def create_model(name: str, config: dict[str, Any], logger: Any = None) -> BaseModel:
    """
    Create a model instance from a discovered name.

    Args:
        name (str): Model name (e.g., ``"RandomForest"``).
        config (dict[str, Any]): Configuration mapping for the model, typically
            a section from ``config.ini`` (e.g., ``[RANDOMFOREST]``).
        logger (Any): Optional logger for progress messages.

    Returns:
        BaseModel: Instantiated model.
    Raises:
        ValueError: If the model name is not registered.
    """
    registry = get_model_registry()
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {sorted(registry.keys())}"
        )
    return registry[name](config=config, logger=logger)
