import polars as pl

from clinical_mining.schemas import (
    validate_schema,
    DrugIndicationEvidence,
    DrugIndication,
)


class DrugIndicationEvidenceDataset:
    """A dataset for drug-indication evidence, wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, DrugIndicationEvidence)

    @staticmethod
    def assign_approval_status(
        indications: pl.DataFrame,
    ) -> pl.DataFrame:
        """Assign approval status to the indications DataFrame.

        Args:
            indications: DataFrame with drug/indication relationships, some of which come from a regulatory agency.
        """
        SOURCES_FOR_APPROVAL = ["FDA", "EMA", "DailyMed"]
        approved_indications = (
            indications.filter(pl.col("source").is_in(SOURCES_FOR_APPROVAL))
            .with_columns(
                approval=pl.struct([
                    pl.col("source").alias("source"),
                    pl.lit(None, dtype=pl.Date).alias("date"),
                ])
            )
            .group_by("drug_id", "disease_id")
            .agg(pl.col("approval").unique().alias("approval"))
        )
        return indications.join(
            approved_indications, on=["drug_id", "disease_id"], how="left"
        )


class DrugIndicationDataset:
    """A dataset for drug-indication relationships, wrapping a Polars DataFrame."""

    AGGREGATION_FIELDS = {
        "drug_id",
        "disease_id",
        "drug_name",
        "disease_name",
        "approval",
    }

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, DrugIndication)

    @classmethod
    def _get_study_metadata_columns(cls, df: pl.DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(list(set(df.columns) - cls.AGGREGATION_FIELDS))

    @classmethod
    def from_evidence(cls, evidence: pl.DataFrame) -> "DrugIndicationDataset":
        """Aggregate drug/indication evidence into a DrugIndicationDataset."""
        # Assert validity of evidence
        validate_schema(evidence, DrugIndicationEvidence)

        study_metadata_cols = cls._get_study_metadata_columns(evidence)
        
        agg_df = (
            evidence.with_columns(
                study_info=pl.struct([pl.col(c) for c in study_metadata_cols])
            )
            .group_by(list(cls.AGGREGATION_FIELDS))
            .agg(
                pl.col("study_info").unique().alias("sources"),
            )
        )
        return cls(df=agg_df)
