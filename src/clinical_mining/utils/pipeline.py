"""Core pipeline execution logic."""

import importlib

from omegaconf import ListConfig
import polars as pl


def _params_reference_key(params: dict, key: str) -> bool:
    """Return True if a step's parameters reference a given data_store key.

    This inspects values like `$spark_session` and also lists containing such
    references. It is used to detect whether a step requires that dependency to
    exist in `data_store` before parameters are resolved.
    """
    for _name, value in params.items():
        if isinstance(value, str) and value == f"${key}":
            return True
        if isinstance(value, (list, ListConfig)):
            for v in value:
                if isinstance(v, str) and v == f"${key}":
                    return True
    return False


def _ensure_spark_session(data_store: dict[str, any]) -> None:
    """Ensure `data_store['spark_session']` is initialized.

    Spark is created lazily, only when a pipeline step explicitly references
    `$spark_session`.

    This function is idempotent: if the session already exists in `data_store`,
    it will be reused and no new Spark session will be created.
    """
    if data_store.get("spark_session") is not None:
        return

    from clinical_mining.utils.spark_helpers import spark_session

    data_store["spark_session"] = spark_session()


def _get_callable(function_path: str):
    """Imports a function or static method from a string path.

    Supports both module-level functions and static/class methods within classes.
    Examples:
        - Module function: 'clinical_mining.data_sources.aact.aact.extract_clinical_report'
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
    step_params = step.get("parameters", {})
    if _params_reference_key(step_params, "spark_session"):
        _ensure_spark_session(data_store)

    params = _resolve_params(step_params, data_store)

    result = func(**params)
    if not isinstance(result, pl.DataFrame) and hasattr(result, "df"):
        result = result.df
    data_store[step.name] = result
    return result
