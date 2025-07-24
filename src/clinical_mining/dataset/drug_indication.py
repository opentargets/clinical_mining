from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from clinical_mining.schemas import (
    validate_schema,
    DrugIndicationEvidence,
    DrugIndication,
)


class DrugIndicationEvidenceDataset:
    """A dataset for drug-indication evidence, wrapping a Spark DataFrame."""

    def __init__(self, df: DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, DrugIndicationEvidence)

    @staticmethod
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


class DrugIndicationDataset:
    """A dataset for drug-indication relationships, wrapping a Spark DataFrame."""

    AGGREGATION_FIELDS = {
        "drug_id",
        "disease_id",
        "drug_name",
        "disease_name",
        "approval",
    }

    def __init__(self, df: DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, DrugIndication)

    @classmethod
    def _get_study_metadata_columns(cls, df: DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(set(df.columns) - cls.AGGREGATION_FIELDS)

    @classmethod
    def from_evidence(cls, evidence: DataFrame) -> "DrugIndicationDataset":
        """Aggregate drug/indication evidence into a DrugIndicationDataset."""
        # Assert validity of evidence
        validate_schema(evidence, DrugIndicationEvidence)

        agg_df = (
            evidence.withColumn(
                "study_info",
                f.struct(
                    *[
                            f.col(c).alias(c)
                            for c in cls._get_study_metadata_columns(evidence)
                    ],
                ),
            )
            .groupBy(*cls.AGGREGATION_FIELDS)
            .agg(
                f.collect_set("study_info").alias("sources"),
                # f.collect_set("approval").alias("approval"), TODO: remove from agg keys
            )
        )
        return cls(df=agg_df)
