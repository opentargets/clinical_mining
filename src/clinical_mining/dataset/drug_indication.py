import polars as pl
import polars_hash as plh

from clinical_mining.schemas import (
    validate_schema,
    DrugIndicationEvidence,
    DrugIndication,
    MaxClinicalStatusCategory,
)

# Clinical status harmonization constants
PHASE_TO_CATEGORY_MAP = {
    # APPROVED (Rank 1)
    "approved": MaxClinicalStatusCategory.APPROVED,
    "authorised": MaxClinicalStatusCategory.APPROVED,
    "approved (orphan drug)": MaxClinicalStatusCategory.APPROVED,
    "approved in china": MaxClinicalStatusCategory.APPROVED,
    "approved in eu": MaxClinicalStatusCategory.APPROVED,
    "opinion": MaxClinicalStatusCategory.APPROVED,
    # POST_APPROVAL_WITHDRAWN (Rank 2)
    "withdrawn": MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "withdrawn from market": MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "revoked": MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "expired": MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "lapsed": MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "suspended": MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    # REGULATORY_REVIEW (Rank 3)
    "application submitted": MaxClinicalStatusCategory.REGULATORY_REVIEW,
    "approval submitted": MaxClinicalStatusCategory.REGULATORY_REVIEW,
    "nda filed": MaxClinicalStatusCategory.REGULATORY_REVIEW,
    "bla submitted": MaxClinicalStatusCategory.REGULATORY_REVIEW,
    "ind submitted": MaxClinicalStatusCategory.REGULATORY_REVIEW,
    "opinion under re-examination": MaxClinicalStatusCategory.REGULATORY_REVIEW,
    # PHASE_4 (Rank 4)
    "phase4": MaxClinicalStatusCategory.PHASE_4,
    "phase 4": MaxClinicalStatusCategory.PHASE_4,
    "discontinued in phase 4": MaxClinicalStatusCategory.PHASE_4,
    # PHASE_3 (Rank 5)
    "phase3": MaxClinicalStatusCategory.PHASE_3,
    "phase 3": MaxClinicalStatusCategory.PHASE_3,
    "phase2/phase3": MaxClinicalStatusCategory.PHASE_3,
    "phase 2/3": MaxClinicalStatusCategory.PHASE_3,
    "discontinued in phase 3": MaxClinicalStatusCategory.PHASE_3,
    "discontinued in phase 2/3": MaxClinicalStatusCategory.PHASE_3,
    # PHASE_2 (Rank 6)
    "phase2": MaxClinicalStatusCategory.PHASE_2,
    "phase 2": MaxClinicalStatusCategory.PHASE_2,
    "phase 2a": MaxClinicalStatusCategory.PHASE_2,
    "phase 2b": MaxClinicalStatusCategory.PHASE_2,
    "phase1/phase2": MaxClinicalStatusCategory.PHASE_2,
    "phase 1/2": MaxClinicalStatusCategory.PHASE_2,
    "phase 1b/2a": MaxClinicalStatusCategory.PHASE_2,
    "phase 1/2a": MaxClinicalStatusCategory.PHASE_2,
    "discontinued in phase 2": MaxClinicalStatusCategory.PHASE_2,
    "discontinued in phase 1/2": MaxClinicalStatusCategory.PHASE_2,
    "discontinued in phase 2a": MaxClinicalStatusCategory.PHASE_2,
    "discontinued in phase 2b": MaxClinicalStatusCategory.PHASE_2,
    # PHASE_1 (Rank 7)
    "phase1": MaxClinicalStatusCategory.PHASE_1,
    "phase 1": MaxClinicalStatusCategory.PHASE_1,
    "phase 1b": MaxClinicalStatusCategory.PHASE_1,
    "early_phase1": MaxClinicalStatusCategory.PHASE_1,
    "phase 0": MaxClinicalStatusCategory.PHASE_1,
    "discontinued in phase 1": MaxClinicalStatusCategory.PHASE_1,
    # PRECLINICAL (Rank 8)
    "preclinical": MaxClinicalStatusCategory.PRECLINICAL,
    "patented": MaxClinicalStatusCategory.PRECLINICAL,
    # NO_DEVELOPMENT_REPORTED (Rank 9)
    "investigative": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "clinical trial": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "registered": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "preregistration": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "terminated": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "discontinued in preregistration": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "application withdrawn": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "refused": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "withdrawn from rolling review": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "NA": MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
}

# Category ranking for Maximum Clinical Development Status
CATEGORY_RANKS = {
    MaxClinicalStatusCategory.APPROVED: 1,
    MaxClinicalStatusCategory.POST_APPROVAL_WITHDRAWN: 2,
    MaxClinicalStatusCategory.REGULATORY_REVIEW: 3,
    MaxClinicalStatusCategory.PHASE_4: 4,
    MaxClinicalStatusCategory.PHASE_3: 5,
    MaxClinicalStatusCategory.PHASE_2: 6,
    MaxClinicalStatusCategory.PHASE_1: 7,
    MaxClinicalStatusCategory.PRECLINICAL: 8,
    MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED: 9,
}

# Sources that indicate approved status when phase is null
APPROVED_SOURCES = {"ATC", "EMA", "FDA", "DailyMed", "PMDA"}


def map_phase_to_category(phase: str | None, source: str) -> MaxClinicalStatusCategory:
    """Map original phase value to standardised category.

    Args:
        phase: Original phase value (can be null)
        source: Data source name

    Returns:
        Standardised clinical status category
    """
    if phase is None:
        # Handle null values based on source
        if source in APPROVED_SOURCES:
            return MaxClinicalStatusCategory.APPROVED
        else:
            return MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED

    # Handle case-insensitive mapping
    phase_lower = phase.lower() if isinstance(phase, str) else str(phase).lower()

    return PHASE_TO_CATEGORY_MAP.get(
        phase_lower, MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED
    )


class DrugIndicationEvidenceDataset:
    """A dataset for drug-indication evidence, wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, DrugIndicationEvidence)


class DrugIndicationDataset:
    """A dataset for drug-indication relationships, wrapping a Polars DataFrame."""

    AGGREGATION_FIELDS = {
        "id",
        "drug_id",
        "disease_id",
        "primary_drug_name",
        "primary_disease_name",
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
                primary_drug_name=pl.coalesce("drug_id", "drug_name"),
                primary_disease_name=pl.coalesce("disease_id", "disease_name"),
                study_info=pl.struct([pl.col(c) for c in study_metadata_cols]),
            )
            .with_columns(
                # Create hash to use as primary key
                id=plh.concat_str(
                    "primary_drug_name", "primary_disease_name"
                ).chash.sha2_256()
            )
            .group_by(list(cls.AGGREGATION_FIELDS))
            .agg(
                pl.col("study_info").unique().alias("sources"),
            )
            .with_columns(
                mapping_status=pl.when(
                    (pl.col("drug_id").is_not_null())
                    & (pl.col("disease_id").is_not_null())
                )
                .then(pl.lit("FULLY_MAPPED"))
                .when(
                    (pl.col("drug_id").is_not_null())
                    & (pl.col("disease_id").is_null())
                )
                .then(pl.lit("DRUG_MAPPED"))
                .when(
                    (pl.col("drug_id").is_null())
                    & (pl.col("disease_id").is_not_null())
                )
                .then(pl.lit("DISEASE_MAPPED"))
                .otherwise(pl.lit("UNMAPPED"))
            )
        )
        return cls(df=agg_df)

    def filter_by_studyid(self, studyId: str) -> "DrugIndicationDataset":
        """Get associations supported by a given study; for example, a clinical trial ID."""

        return DrugIndicationDataset(
            df=self.df.with_columns(
                pl.col("sources")
                .list.eval(pl.element().struct["studyId"])
                .alias("studyIds"),
            )
            .filter(pl.col("studyIds").list.contains(studyId))
            .drop("studyIds")
        )

    @staticmethod
    def assign_max_clinical_status(df: pl.DataFrame) -> "DrugIndicationDataset":
        """Assign harmonized clinical status using Maximum Clinical Development Status (MCDS) logic.

        For each drug-indication pair, determines the highest-ranked clinical status
        across all supporting sources based on the harmonization categories.

        Returns:
            DrugIndicationDataset with clinical_status field added containing the MCDS category
        """

        def get_max_clinical_status(sources_list) -> str:
            """Get the maximum clinical development status category from a list of sources."""
            # Convert Polars list to Python list if needed
            if hasattr(sources_list, "to_list"):
                sources_list = sources_list.to_list()

            # Extract all clinical statuses from sources and find the best rank
            best_rank = CATEGORY_RANKS[MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED]
            best_category = MaxClinicalStatusCategory.NO_DEVELOPMENT_REPORTED

            for source in sources_list:
                phase = source.get("clinical_phase")
                source_name = source.get("source")

                if source_name:
                    category = map_phase_to_category(phase, source_name)
                    rank = CATEGORY_RANKS[category]

                    # Lower rank number = higher priority
                    if rank < best_rank:
                        best_rank = rank
                        best_category = category

            return best_category.value

        # Apply MCDS logic to each drug-indication pair
        df_with_status = df.with_columns(
            [
                pl.col("sources")
                .map_elements(get_max_clinical_status, return_dtype=pl.String)
                .alias("max_clinical_status")
            ]
        )

        return DrugIndicationDataset(df=df_with_status)
