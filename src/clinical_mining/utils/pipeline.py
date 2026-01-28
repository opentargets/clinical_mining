"""Core pipeline execution logic."""

import importlib

from omegaconf import ListConfig
import polars as pl


def _get_callable(function_path: str):
    """Imports a function or static method from a string path.

    Supports both module-level functions and static/class methods within classes.
    Examples:
        - Module function: 'clinical_mining.data_sources.aact.aact.extract_clinical_record'
        - Static method: 'clinical_mining.dataset.clinical_indication.ClinicalIndication.assign_approval_status'

    Args:
        function_path (str): The path to the function to import.
    Returns:
        function: The imported function.
    Raises:
        ImportError: If the function or static method cannot be imported.
    """
    try:
        # Try to import as a module-level function first
        try:
            module_path, function_name = function_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ImportError, AttributeError):
            # If that fails, try to import as a class method
            parts = function_path.rsplit(".", 2)
            if len(parts) == 3:
                module_path, class_name, method_name = parts
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                return getattr(cls, method_name)
            raise
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import function '{function_path}': {e}")


def _resolve_params(params: dict, data_store: dict) -> dict:
    """Resolves parameter values from the data_store.

    Args:
        params (dict): The parameters to resolve.
        data_store (dict): The data store to resolve the parameters from.
    Returns:
        dict: The resolved parameters.
    """
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


def execute_step(
    step: dict[str, any],
    data_store: dict[str, any],
) -> any:
    """Executes a single pipeline step and updates data_store to include the result.

    Args:
        step (dict[str, any]): The step to execute.
        data_store (dict[str, any]): The data store with all dependencies
    Returns:
        any: The result of the step execution.
    """
    func = _get_callable(step.function)
    params = _resolve_params(step.get("parameters", {}), data_store)

    result = func(**params)
    if not isinstance(result, pl.DataFrame) and hasattr(result, "df"):
        result = result.df
    data_store[step.name] = result
    return result
