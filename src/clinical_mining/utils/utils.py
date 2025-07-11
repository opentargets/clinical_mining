"""Utils for the clinical mining pipeline."""

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
            f.struct(
                f.col("source").alias("source"),
                f.lit(None).cast("date").alias("date"),
            ),
        )
        .groupBy("drug_id", "disease_id")
        .agg(f.collect_set("approval").alias("approval"))
    )
    return indications.join(
        approved_indications, on=["drug_id", "disease_id"], how="left"
    )
