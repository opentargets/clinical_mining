import polars as pl
import polars_hash as plh

from clinical_mining.schemas import (
    snake_to_camel,
    validate_schema,
    ClinicalEvidenceSchema,
    ClinicalAssociationSchema,
    ClinicalStatusCategory,
)
from clinical_mining.utils.polars_helpers import coalesce_column

# Clinical status harmonization constants
PHASE_TO_CATEGORY_MAP = {
    # APPROVED (Rank 1)
    "approved": ClinicalStatusCategory.APPROVED,
    "authorised": ClinicalStatusCategory.APPROVED,
    "approved (orphan drug)": ClinicalStatusCategory.APPROVED,
    "approved in china": ClinicalStatusCategory.APPROVED,
    "approved in eu": ClinicalStatusCategory.APPROVED,
    "opinion": ClinicalStatusCategory.APPROVED,
    # POST_APPROVAL_WITHDRAWN (Rank 2)
    "withdrawn": ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "withdrawn from market": ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "revoked": ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "expired": ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "lapsed": ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    "suspended": ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN,
    # REGULATORY_REVIEW (Rank 3)
    "application submitted": ClinicalStatusCategory.REGULATORY_REVIEW,
    "approval submitted": ClinicalStatusCategory.REGULATORY_REVIEW,
    "nda filed": ClinicalStatusCategory.REGULATORY_REVIEW,
    "bla submitted": ClinicalStatusCategory.REGULATORY_REVIEW,
    "ind submitted": ClinicalStatusCategory.REGULATORY_REVIEW,
    "opinion under re-examination": ClinicalStatusCategory.REGULATORY_REVIEW,
    # PHASE_4 (Rank 4)
    "phase4": ClinicalStatusCategory.PHASE_4,
    "phase 4": ClinicalStatusCategory.PHASE_4,
    "discontinued in phase 4": ClinicalStatusCategory.PHASE_4,
    # PHASE_3 (Rank 5)
    "phase3": ClinicalStatusCategory.PHASE_3,
    "phase 3": ClinicalStatusCategory.PHASE_3,
    "phase2/phase3": ClinicalStatusCategory.PHASE_3,
    "phase 2/3": ClinicalStatusCategory.PHASE_3,
    "discontinued in phase 3": ClinicalStatusCategory.PHASE_3,
    "discontinued in phase 2/3": ClinicalStatusCategory.PHASE_3,
    # PHASE_2 (Rank 6)
    "phase2": ClinicalStatusCategory.PHASE_2,
    "phase 2": ClinicalStatusCategory.PHASE_2,
    "phase 2a": ClinicalStatusCategory.PHASE_2,
    "phase 2b": ClinicalStatusCategory.PHASE_2,
    "phase1/phase2": ClinicalStatusCategory.PHASE_2,
    "phase 1/2": ClinicalStatusCategory.PHASE_2,
    "phase 1b/2a": ClinicalStatusCategory.PHASE_2,
    "phase 1/2a": ClinicalStatusCategory.PHASE_2,
    "discontinued in phase 2": ClinicalStatusCategory.PHASE_2,
    "discontinued in phase 1/2": ClinicalStatusCategory.PHASE_2,
    "discontinued in phase 2a": ClinicalStatusCategory.PHASE_2,
    "discontinued in phase 2b": ClinicalStatusCategory.PHASE_2,
    # PHASE_1 (Rank 7)
    "phase1": ClinicalStatusCategory.PHASE_1,
    "phase 1": ClinicalStatusCategory.PHASE_1,
    "phase 1b": ClinicalStatusCategory.PHASE_1,
    "early_phase1": ClinicalStatusCategory.PHASE_1,
    "phase 0": ClinicalStatusCategory.PHASE_1,
    "discontinued in phase 1": ClinicalStatusCategory.PHASE_1,
    # PRECLINICAL (Rank 8)
    "preclinical": ClinicalStatusCategory.PRECLINICAL,
    "patented": ClinicalStatusCategory.PRECLINICAL,
    # NO_DEVELOPMENT_REPORTED (Rank 9)
    "investigative": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "clinical trial": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "registered": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "preregistration": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "terminated": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "discontinued in preregistration": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "application withdrawn": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "refused": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "withdrawn from rolling review": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
    "NA": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED,
}

# Category ranking for Maximum Clinical Development Status
CATEGORY_RANKS = {
    ClinicalStatusCategory.APPROVED: 1,
    ClinicalStatusCategory.POST_APPROVAL_WITHDRAWN: 2,
    ClinicalStatusCategory.REGULATORY_REVIEW: 3,
    ClinicalStatusCategory.PHASE_4: 4,
    ClinicalStatusCategory.PHASE_3: 5,
    ClinicalStatusCategory.PHASE_2: 6,
    ClinicalStatusCategory.PHASE_1: 7,
    ClinicalStatusCategory.PRECLINICAL: 8,
    ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED: 9,
}

CATEGORY_RANKS_STR = {k.value: v for k, v in CATEGORY_RANKS.items()}
RANK_TO_CATEGORY_STR = {v: k.value for k, v in CATEGORY_RANKS.items()}

# Sources that indicate approved status when phase is null
APPROVED_SOURCES = {"ATC", "EMA", "FDA", "DailyMed", "PMDA"}


def map_phase_to_category(phase: str | None, source: str) -> ClinicalStatusCategory:
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
            return ClinicalStatusCategory.APPROVED
        else:
            return ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED

    # Handle case-insensitive mapping
    phase_lower = phase.lower() if isinstance(phase, str) else str(phase).lower()

    return PHASE_TO_CATEGORY_MAP.get(
        phase_lower, ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED
    )


class ClinicalEvidence:
    """A dataset for drug-indication evidence, wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""

        # Set ID column
        df = coalesce_column(df, "drugName", ["drugId", "drugFromSource"])
        df = coalesce_column(df, "diseaseName", ["diseaseId", "diseaseFromSource"])
        df = df.with_columns(
            id=plh.concat_str("drugName", "diseaseName", "studyId").chash.sha2_256()
        )

        # Harmonise column names from snake to camel case
        df = df.rename({col: snake_to_camel(col) for col in df.columns})

        # Assign clinical status to evidence (if available)
        if "clinicalPhase" in df.columns and "source" in df.columns:
            df = df.with_columns(
                pl.struct(["clinicalPhase", "source"])
                .map_elements(
                    lambda r: map_phase_to_category(
                        r.get("clinicalPhase"), r.get("source")
                    ).value
                )
                .alias("clinicalStatus")
            )
        else:
            df = df.with_columns(
                clinicalStatus=pl.lit(ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED)
            )

        self.df = validate_schema(df, ClinicalEvidenceSchema)


class ClinicalAssociation:
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
        self.df = validate_schema(self.df, ClinicalAssociationSchema)

    @classmethod
    def _get_study_metadata_columns(cls, df: pl.DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(list(set(df.columns) - cls.AGGREGATION_FIELDS))

    @classmethod
    def from_evidence(cls, evidence: pl.DataFrame) -> "ClinicalAssociation":
        """Aggregate drug/indication evidence into a ClinicalAssociation."""
        # Assert validity of evidence
        validate_schema(evidence, ClinicalEvidenceSchema)

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

    def filter_by_studyid(self, studyId: str) -> "ClinicalAssociation":
        """Get associations supported by a given study; for example, a clinical trial ID."""

        return ClinicalAssociation(
            df=self.df.with_columns(
                pl.col("sources")
                .list.eval(pl.element().struct["studyId"])
                .alias("studyIds"),
            )
            .filter(pl.col("studyIds").list.contains(studyId))
            .drop("studyIds")
        )

    def assign_max_clinical_status(self) -> "ClinicalAssociation":
        """Assign harmonised clinical status using Maximum Clinical Development Status (MCDS) logic.

        For each drug-indication pair, determines the highest-ranked clinical status
        across all supporting sources based on the harmonisation categories.

        Returns:
            ClinicalAssociation with clinical_status field added containing the MCDS category
        """

        def get_max_clinical_status(sources_list) -> str:
            """Get the maximum clinical development status category from a list of sources."""
            # Convert Polars list to Python list if needed
            if hasattr(sources_list, "to_list"):
                sources_list = sources_list.to_list()

            # Extract all clinical statuses from sources and find the best rank
            best_rank = CATEGORY_RANKS[ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED]
            best_category = ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED

            for source in sources_list:
                status_str = source.get("clinicalStatus")
                if status_str:
                    rank = CATEGORY_RANKS_STR.get(
                        status_str,
                        CATEGORY_RANKS[ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED],
                    )
                    category = ClinicalStatusCategory(status_str)
                else:
                    phase = source.get("clinicalPhase")
                    source_name = source.get("source")
                    if source_name:
                        category = map_phase_to_category(phase, source_name)
                        rank = CATEGORY_RANKS[category]
                    else:
                        category = ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED
                        rank = CATEGORY_RANKS[category]

                # Lower rank number = higher priority
                if rank < best_rank:
                    best_rank = rank
                    best_category = category

            return best_category.value

        # Apply MCDS logic to each drug-indication pair
        self.df = self.df.with_columns(
            maxClinicalStatus=pl.col("sources").map_elements(
                get_max_clinical_status, return_dtype=pl.String
            )
        )
        return self
