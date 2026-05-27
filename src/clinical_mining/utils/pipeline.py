"""Core pipeline execution logic."""

import importlib

from typing import Any, Callable

from omegaconf import DictConfig, ListConfig
import polars as pl


def _params_reference_key(params: dict[str, Any], key: str) -> bool:
    """Return True if a step's parameters reference a given data_store key."""
    for _name, value in params.items():
        if isinstance(value, str) and value == f"${key}":
            return True
        if isinstance(value, (list, ListConfig)):
            for v in value:
                if isinstance(v, str) and v == f"${key}":
                    return True
    return False


def _ensure_spark_session(data_store: dict[str, Any]) -> None:
    """Ensure data_store['spark_session'] is initialized lazily."""
    if data_store.get("spark_session") is not None:
        return

    from clinical_mining.utils.spark_helpers import spark_session

    data_store["spark_session"] = spark_session()


def _get_callable(function_path: str) -> Callable[..., Any]:
    """Imports a function or static method from a string path."""
    try:
        try:
            module_path, function_name = function_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ImportError, AttributeError):
            parts = function_path.rsplit(".", 2)
            if len(parts) == 3:
                module_path, class_name, method_name = parts
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                return getattr(cls, method_name)
            raise
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import function '{function_path}': {e}")


def _resolve_params(params: dict[str, Any], data_store: dict[str, Any]) -> dict[str, Any]:
    """Resolves parameter values from the data_store."""
    resolved_params = {}
    for name, value in params.items():
        if isinstance(value, str) and value.startswith("$"):
            resolved_params[name] = data_store[value[1:]]
        elif isinstance(value, (list, ListConfig)):
            resolved_params[name] = [
                data_store[v[1:]] if isinstance(v, str) and v.startswith("$") else v
                for v in value
            ]
        else:
            resolved_params[name] = value
    return resolved_params


def normalise_steps(section: dict | list) -> list[tuple[str, dict]]:
    """Normalize a section (dict or list) into a list of (name, step_dict) tuples.

    Supports both formats:
        # List format (legacy)
        - name: step_name
          function: ...
          parameters: ...

        # Dict format (preferred)
        step_name:
          function: ...
          parameters: ...
    """
    if isinstance(section, list):
        return [(step["name"], step) for step in section]
    elif isinstance(section, (dict, DictConfig)):
        return [(name, {"name": name, **dict(step)}) for name, step in section.items()]
    return []


def execute_step(
    step: dict[str, Any],
    data_store: dict[str, Any],
) -> Any:
    """Executes a single pipeline step and updates data_store."""
    func = _get_callable(step["function"])
    step_params = step.get("parameters", {})
    if _params_reference_key(step_params, "spark_session"):
        _ensure_spark_session(data_store)

    params = _resolve_params(step_params, data_store)

    result = func(**params)
    if not isinstance(result, pl.DataFrame) and hasattr(result, "df"):
        result = result.df
    data_store[step["name"]] = result
    return result
