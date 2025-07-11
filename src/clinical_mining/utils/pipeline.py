"""Core pipeline execution logic."""

import importlib
import inspect

from omegaconf import ListConfig

from clinical_mining.utils.spark import SparkSession


def _get_callable(function_path: str):
    """Imports a function from a string path. E.g. clinical_mining.data_sources.aact.aact.extract_clinical_trials.

    Args:
        function_path (str): The path to the function to import.
    Returns:
        function: The imported function.
    Raises:
        ImportError: If the function cannot be imported.
    """
    try:
        module_path, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
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
    step: dict[str, any], data_store: dict[str, any], spark: SparkSession
) -> any:
    """Executes a single pipeline step with automatic dependency injection for Spark and updates data_store to include the result.

    Args:
        step (dict[str, any]): The step to execute.
        data_store (dict[str, any]): The data store with all dependencies
        spark (SparkSession): The Spark session to use.
    Returns:
        any: The result of the step execution.
    """
    func = _get_callable(step.function)
    params = _resolve_params(step.get("parameters", {}), data_store)

    # Inspect function signature to inject spark when needed
    if "spark_session" in inspect.signature(func).parameters:
        params["spark_session"] = spark

    result = func(**params)
    data_store[step.name] = result
    return result
