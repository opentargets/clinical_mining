import polars as pl
import polars_hash as plh

from clinical_mining.schemas import (
    validate_schema,
    ClinicalIndicationSchema,
    ClinicalReportSchema,
    ClinicalStageCategory,
)

# Category ranking for Maximum Clinical Development Status
CATEGORY_RANKS = {
    ClinicalStageCategory.APPROVAL: 1,
    ClinicalStageCategory.PREAPPROVAL: 2,
    ClinicalStageCategory.PHASE_3: 3,
    ClinicalStageCategory.PHASE_2_3: 4,
    ClinicalStageCategory.PHASE_2: 5,
    ClinicalStageCategory.PHASE_1_2: 6,
    ClinicalStageCategory.PHASE_1: 7,
    ClinicalStageCategory.EARLY_PHASE_1: 8,
    ClinicalStageCategory.IND: 9,
    ClinicalStageCategory.PRECLINICAL: 10,
    ClinicalStageCategory.UNKNOWN: 11,
}

CATEGORY_RANKS_STR = {k.value: v for k, v in CATEGORY_RANKS.items()}
RANK_TO_CATEGORY_STR = {v: k.value for k, v in CATEGORY_RANKS.items()}


class ClinicalIndication:
    """A dataset for drug-indication relationships, wrapping a Polars DataFrame."""

    AGGREGATION_FIELDS = {
        "id",
        "drugName",
        "diseaseName",
        "drugId",
        "diseaseId",
    }

    def __init__(self, df: pl.DataFrame):
        """Initialises the dataset, validating and aligning the DataFrame.

        Also assigns mapping status and maximum clinical status.
        """

        df = df.with_columns(
            mappingStatus=self._assign_mapping_status(
                pl.col("drugId"), pl.col("diseaseId")
            ),
        )
        self.df = validate_schema(df, ClinicalIndicationSchema)

    @classmethod
    def _get_study_metadata_columns(cls, df: pl.DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(list(set(df.columns) - cls.AGGREGATION_FIELDS))

    @classmethod
    def from_report(cls, report: pl.DataFrame) -> "ClinicalIndication":
        """Aggregate drug/disease information collected in clinical report into a ClinicalIndication."""
        # Assert validity of reports
        report = validate_schema(report, ClinicalReportSchema)

        return cls(
            df=(
                # Explode clinical reports to get lists of study/disease/drug
                report.explode("drugs")
                .explode("diseases")
                .unnest(["drugs", "diseases"])
                .rename({"id": "clinicalReportId"})
                .with_columns(
                    clinicalStageForMax=pl.when(
                        pl.col("clinicalStage").is_in(
                            [
                                ClinicalStageCategory.WITHDRAWN.value,
                                ClinicalStageCategory.PHASE_4.value,
                                ClinicalStageCategory.APPROVAL.value,
                            ]
                        )
                    )
                    .then(pl.lit(ClinicalStageCategory.APPROVAL.value))
                    .otherwise(pl.col("clinicalStage")),
                )
                .with_columns(
                    # Create hash to use as primary key (no studyId in this case)
                    id=plh.concat_str(
                        pl.coalesce(pl.col("drugId"), pl.col("drugFromSource")),
                        pl.coalesce(pl.col("diseaseId"), pl.col("diseaseFromSource")),
                    ).chash.sha2_256(),
                    drugName=pl.coalesce(pl.col("drugId"), pl.col("drugFromSource")),
                    diseaseName=pl.coalesce(
                        pl.col("diseaseId"), pl.col("diseaseFromSource")
                    ),
                    # Map clinicalStage to rank for sorting
                    clinicalStageRank=pl.col("clinicalStageForMax").replace_strict(
                        CATEGORY_RANKS_STR,
                        default=CATEGORY_RANKS[ClinicalStageCategory.UNKNOWN],
                    ),
                )
                .sort("clinicalStageRank")
                .group_by(list(cls.AGGREGATION_FIELDS), maintain_order=True)
                .agg(
                    pl.col("clinicalReportId").unique().alias("clinicalReportIds"),
                    # Get the maximum clinical stage (minimum rank = highest priority)
                    pl.col("clinicalStageForMax").first().alias("maxClinicalStage"),
                )
            ),
        )

    @staticmethod
    def _assign_mapping_status(
        drug_id_column: pl.Expr, disease_id_column: pl.Expr
    ) -> pl.Expr:
        return (
            pl.when((drug_id_column.is_not_null()) & (disease_id_column.is_not_null()))
            .then(pl.lit("FULLY_MAPPED"))
            .when((drug_id_column.is_not_null()) & (disease_id_column.is_null()))
            .then(pl.lit("DRUG_MAPPED"))
            .when((drug_id_column.is_null()) & (disease_id_column.is_not_null()))
            .then(pl.lit("DISEASE_MAPPED"))
            .otherwise(pl.lit("UNMAPPED"))
        )

    def filter_by_report_id(self, report_id: str) -> "ClinicalIndication":
        """Get associations supported by a given report ID."""

        return ClinicalIndication(
            df=self.df.filter(pl.col("clinicalReportIds").list.contains(report_id))
        )
