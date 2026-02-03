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
    ClinicalStageCategory.APPROVED: 1,
    ClinicalStageCategory.POST_APPROVAL_WITHDRAWN: 2,
    ClinicalStageCategory.REGULATORY_REVIEW: 3,
    ClinicalStageCategory.PHASE_4: 4,
    ClinicalStageCategory.PHASE_3: 5,
    ClinicalStageCategory.PHASE_2: 6,
    ClinicalStageCategory.PHASE_1: 7,
    ClinicalStageCategory.PRECLINICAL: 8,
    ClinicalStageCategory.NO_DEVELOPMENT_REPORTED: 9,
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

    def __init__(self, df: pl.DataFrame, filter_by_mapping_status: bool = False):
        """Initialises the dataset, validating and aligning the DataFrame.

        Also assigns mapping status and maximum clinical status.
        """

        df = df.with_columns(
            mappingStatus=self._assign_mapping_status(
                pl.col("drugId"), pl.col("diseaseId")
            ),
        )

        if filter_by_mapping_status:
            df = df.filter(pl.col("mappingStatus") == "FULLY_MAPPED").drop(
                "drugName", "diseaseName", "mappingStatus"
            )
        self.df = validate_schema(df, ClinicalIndicationSchema)

    @classmethod
    def _get_study_metadata_columns(cls, df: pl.DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(list(set(df.columns) - cls.AGGREGATION_FIELDS))

    @classmethod
    def from_report(
        cls, report: pl.DataFrame, filter_by_mapping_status: bool = False
    ) -> "ClinicalIndication":
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
                    drugName=pl.coalesce(pl.col("drugId"), pl.col("drugFromSource")),
                    diseaseName=pl.coalesce(
                        pl.col("diseaseId"), pl.col("diseaseFromSource")
                    ),
                    # Create hash to use as primary key (no studyId in this case)
                    id=plh.concat_str(
                        pl.coalesce(pl.col("drugId"), pl.col("drugFromSource")),
                        pl.coalesce(pl.col("diseaseId"), pl.col("diseaseFromSource")),
                    ).chash.sha2_256(),
                    # Map clinicalStage to rank for sorting
                    clinicalStageRank=pl.col("clinicalStage").replace_strict(
                        CATEGORY_RANKS_STR,
                        default=CATEGORY_RANKS[
                            ClinicalStageCategory.NO_DEVELOPMENT_REPORTED
                        ],
                    ),
                )
                .sort("clinicalStageRank")
                .group_by(list(cls.AGGREGATION_FIELDS), maintain_order=True)
                .agg(
                    pl.col("clinicalReportId").unique().alias("clinicalReportIds"),
                    pl.col("hasExpertReview").any().alias("hasExpertReview"),
                    # Get the maximum clinical stage (minimum rank = highest priority)
                    pl.col("clinicalStage").first().alias("maxClinicalStage"),
                )
            ),
            filter_by_mapping_status=filter_by_mapping_status,
        )

    @staticmethod
    def _assign_mapping_status(
        drug_id_column: pl.Expr, disease_id_column: pl.Expr
    ) -> pl.Expr:
        return (
            pl.when((drug_id_column.is_not_null()) & (disease_id_column.is_not_null()))
            .then(pl.lit("FULLY_MAPPED"))
            .when((pl.col("drugId").is_not_null()) & (pl.col("diseaseId").is_null()))
            .then(pl.lit("DRUG_MAPPED"))
            .when((pl.col("drugId").is_null()) & (pl.col("diseaseId").is_not_null()))
            .then(pl.lit("DISEASE_MAPPED"))
            .otherwise(pl.lit("UNMAPPED"))
        )

    def filter_by_report_id(self, report_id: str) -> "ClinicalIndication":
        """Get associations supported by a given report ID."""

        return ClinicalIndication(
            df=self.df.filter(pl.col("clinicalReportIds").list.contains(report_id))
        )
