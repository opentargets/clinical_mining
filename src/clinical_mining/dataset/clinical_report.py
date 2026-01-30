import polars as pl

from clinical_mining.schemas import validate_schema, ClinicalReportSchema, ClinicalStageCategory, snake_to_camel
from clinical_mining.dataset.clinical_indication import CATEGORY_RANKS_STR

# Clinical status harmonization constants
PHASE_TO_CATEGORY_MAP = {
    # APPROVED (Rank 1)
    "approved": ClinicalStageCategory.APPROVED,
    "authorised": ClinicalStageCategory.APPROVED,
    "approved (orphan drug)": ClinicalStageCategory.APPROVED,
    "approved in china": ClinicalStageCategory.APPROVED,
    "approved in eu": ClinicalStageCategory.APPROVED,
    "opinion": ClinicalStageCategory.APPROVED,
    # POST_APPROVAL_WITHDRAWN (Rank 2)
    "withdrawn": ClinicalStageCategory.POST_APPROVAL_WITHDRAWN,
    "withdrawn from market": ClinicalStageCategory.POST_APPROVAL_WITHDRAWN,
    "revoked": ClinicalStageCategory.POST_APPROVAL_WITHDRAWN,
    "expired": ClinicalStageCategory.POST_APPROVAL_WITHDRAWN,
    "lapsed": ClinicalStageCategory.POST_APPROVAL_WITHDRAWN,
    "suspended": ClinicalStageCategory.POST_APPROVAL_WITHDRAWN,
    # REGULATORY_REVIEW (Rank 3)
    "application submitted": ClinicalStageCategory.REGULATORY_REVIEW,
    "approval submitted": ClinicalStageCategory.REGULATORY_REVIEW,
    "nda filed": ClinicalStageCategory.REGULATORY_REVIEW,
    "bla submitted": ClinicalStageCategory.REGULATORY_REVIEW,
    "ind submitted": ClinicalStageCategory.REGULATORY_REVIEW,
    "opinion under re-examination": ClinicalStageCategory.REGULATORY_REVIEW,
    # PHASE_4 (Rank 4)
    "phase4": ClinicalStageCategory.PHASE_4,
    "phase 4": ClinicalStageCategory.PHASE_4,
    "discontinued in phase 4": ClinicalStageCategory.PHASE_4,
    # PHASE_3 (Rank 5)
    "phase3": ClinicalStageCategory.PHASE_3,
    "phase 3": ClinicalStageCategory.PHASE_3,
    "phase2/phase3": ClinicalStageCategory.PHASE_3,
    "phase 2/3": ClinicalStageCategory.PHASE_3,
    "discontinued in phase 3": ClinicalStageCategory.PHASE_3,
    "discontinued in phase 2/3": ClinicalStageCategory.PHASE_3,
    # PHASE_2 (Rank 6)
    "phase2": ClinicalStageCategory.PHASE_2,
    "phase 2": ClinicalStageCategory.PHASE_2,
    "phase 2a": ClinicalStageCategory.PHASE_2,
    "phase 2b": ClinicalStageCategory.PHASE_2,
    "phase1/phase2": ClinicalStageCategory.PHASE_2,
    "phase 1/2": ClinicalStageCategory.PHASE_2,
    "phase 1b/2a": ClinicalStageCategory.PHASE_2,
    "phase 1/2a": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 2": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 1/2": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 2a": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 2b": ClinicalStageCategory.PHASE_2,
    # PHASE_1 (Rank 7)
    "phase1": ClinicalStageCategory.PHASE_1,
    "phase 1": ClinicalStageCategory.PHASE_1,
    "phase 1b": ClinicalStageCategory.PHASE_1,
    "early_phase1": ClinicalStageCategory.PHASE_1,
    "phase 0": ClinicalStageCategory.PHASE_1,
    "discontinued in phase 1": ClinicalStageCategory.PHASE_1,
    # PRECLINICAL (Rank 8)
    "preclinical": ClinicalStageCategory.PRECLINICAL,
    "patented": ClinicalStageCategory.PRECLINICAL,
    # NO_DEVELOPMENT_REPORTED (Rank 9)
    "investigative": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "clinical trial": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "registered": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "preregistration": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "terminated": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "discontinued in preregistration": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "application withdrawn": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "refused": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "withdrawn from rolling review": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
    "NA": ClinicalStageCategory.NO_DEVELOPMENT_REPORTED,
}

# Sources that indicate approved status when phase is null
APPROVED_SOURCES = {"ATC", "EMA", "FDA", "DailyMed", "PMDA"}


def map_phase_to_category(phase: str | None, source: str) -> ClinicalStageCategory:
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
            return ClinicalStageCategory.APPROVED
        else:
            return ClinicalStageCategory.NO_DEVELOPMENT_REPORTED

    # Handle case-insensitive mapping
    phase_lower = phase.lower() if isinstance(phase, str) else str(phase).lower()

    return PHASE_TO_CATEGORY_MAP.get(
        phase_lower, ClinicalStageCategory.NO_DEVELOPMENT_REPORTED
    )


class ClinicalReport:
    """A dataset for clinical reports (e.g. clinical trial, an USAN reference), wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initialises the dataset, validating and aligning the DataFrame."""
        # Harmonise column names from snake to camel case
        df = df.rename(
            {col: snake_to_camel(col) for col in df.columns}
        )

        # Assign clinical stage
        df = df.with_columns(
            clinicalStage=pl.struct(["phaseFromSource", "source"]).map_elements(
                lambda row: map_phase_to_category(row["phaseFromSource"], row["source"]),
                return_dtype=pl.String
            )
        )

        # Drop duplicates by id, keeping the row with the best clinical stage
        df = self.drop_duplicates(df)

        self.df = validate_schema(df, ClinicalReportSchema)

    @classmethod
    def drop_duplicates(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Drop duplicate reports based on clinical stage."""
        return (
            df.with_columns(
                clinicalStageRank=pl.col("clinicalStage").replace_strict(
                    CATEGORY_RANKS_STR,
                    default=CATEGORY_RANKS_STR[ClinicalStageCategory.NO_DEVELOPMENT_REPORTED.value],
                )
            )
            .sort(["id", "clinicalStageRank"])
            .unique(subset=["id"], keep="first")
            .drop("clinicalStageRank")
        )
