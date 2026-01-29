import polars as pl
import polars_hash as plh

from clinical_mining.schemas import (
    validate_schema,
    ClinicalIndicationSchema,
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

    def __init__(self, df: pl.DataFrame):
        """Initialises the dataset, validating and aligning the DataFrame.

        Also assigns mapping status and maximum clinical status.
        """

        self.df = df.with_columns(
            mappingStatus=self._assign_mapping_status(
                pl.col("drugId"), pl.col("diseaseId")
            ),
        )
        self.assign_max_clinical_status()
        self.df = validate_schema(self.df, ClinicalIndicationSchema)

    @classmethod
    def _get_study_metadata_columns(cls, df: pl.DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(list(set(df.columns) - cls.AGGREGATION_FIELDS))

    @classmethod
    def from_evidence(cls, evidence: pl.DataFrame) -> "ClinicalIndication":
        """Aggregate drug/indication evidence into a ClinicalIndication."""
        # Assert validity of evidence

        study_metadata_cols = cls._get_study_metadata_columns(evidence)

        agg_df = (
            evidence.with_columns(
                # Create hash to use as primary key (no studyId in this case)
                id=plh.concat_str("drugName", "diseaseName").chash.sha2_256(),
                study_info=pl.struct([pl.col(c) for c in study_metadata_cols]),
            )
            .group_by(list(cls.AGGREGATION_FIELDS))
            .agg(
                pl.col("study_info").unique().alias("sources"),
            )
        )
        return cls(df=agg_df)

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

    def filter_by_studyid(self, studyId: str) -> "ClinicalIndication":
        """Get associations supported by a given study; for example, a clinical trial ID."""

        return ClinicalIndication(
            df=self.df.with_columns(
                pl.col("sources")
                .list.eval(pl.element().struct["studyId"])
                .alias("studyIds"),
            )
            .filter(pl.col("studyIds").list.contains(studyId))
            .drop("studyIds")
        )

    def assign_max_clinical_status(self) -> "ClinicalIndication":
        """Assign harmonised clinical status using Maximum Clinical Development Status (MCDS) logic.

        For each drug-indication pair, determines the highest-ranked clinical status
        across all supporting sources based on the harmonisation categories.

        Returns:
            ClinicalIndication with clinical_status field added containing the MCDS category
        """

        def get_max_clinical_status(sources_list) -> str:
            """Get the maximum clinical stage status category from a list of sources."""
            # Convert Polars list to Python list if needed
            if hasattr(sources_list, "to_list"):
                sources_list = sources_list.to_list()

            # Extract all clinical statuses from sources and find the best rank
            best_rank = CATEGORY_RANKS[ClinicalStageCategory.NO_DEVELOPMENT_REPORTED]
            best_category = ClinicalStageCategory.NO_DEVELOPMENT_REPORTED

            for source in sources_list:
                status_str = source.get("clinicalStage")
                if status_str:
                    rank = CATEGORY_RANKS_STR.get(
                        status_str,
                        CATEGORY_RANKS[ClinicalStageCategory.NO_DEVELOPMENT_REPORTED],
                    )
                    category = ClinicalStageCategory(status_str)
                else:
                    category = ClinicalStageCategory.NO_DEVELOPMENT_REPORTED
                    rank = CATEGORY_RANKS[category]

                # Lower rank number = higher priority
                if rank < best_rank:
                    best_rank = rank
                    best_category = category

            return best_category.value

        # Apply MCDS logic to each drug-indication pair
        self.df = self.df.with_columns(
            maxClinicalStage=pl.col("sources").map_elements(
                get_max_clinical_status, return_dtype=pl.String
            )
        )
        return self
