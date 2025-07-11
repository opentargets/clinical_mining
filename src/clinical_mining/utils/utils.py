"""Utils for the clinical mining pipeline."""

import inspect
from typing import Callable

from pyspark.sql import DataFrame
import pyspark.sql.functions as f


def assign_approval_status(
    indications: DataFrame,
) -> DataFrame:
    """Assign approval status to the indications DataFrame.

    Args:
        indications: DataFrame with drug/indication relationships, some of which come from a regulatory agency
    """
    SOURCES_FOR_APPROVAL = ["FDA", "EMA", "DailyMed"]
    approved_indications = (
        indications.filter(f.col("source").isin(SOURCES_FOR_APPROVAL))
        .withColumn(
            "approval",
            f.struct(f.col("source").alias("source"),
            f.lit(None).cast("date").alias("date")
            ),
        )
        .groupBy("drug_id", "disease_id")
        .agg(f.collect_set("approval").alias("approval"))
    )
    return indications.join(
        approved_indications, on=["drug_id", "disease_id"], how="left"
    )

def call_with_dependencies(func: Callable, data_sources: dict[str, DataFrame]) -> any:
    """
    Inspects a function's signature and calls it by injecting dependencies from a dictionary.

    Args:
        func (Callable): The function to call.
        data_sources (dict[str, DataFrame]): A dictionary of available data sources (DataFrames).

    Returns:
        any: The result of the function call.
    """
    required_params = inspect.signature(func).parameters
    kwargs = {name: data_sources[name] for name in required_params if name in data_sources}
    return func(**kwargs)