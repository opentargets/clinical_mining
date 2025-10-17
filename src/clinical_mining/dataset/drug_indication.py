import polars as pl

from clinical_mining.schemas import (
    validate_schema,
    DrugIndicationEvidence,
    DrugIndication,
    ClinicalStatusCategory,
    ClinicalStatus,
)


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

# Sources that indicate approved status when phase is null
APPROVED_SOURCES = {"ATC", "EMA", "FDA", "DailyMed"}


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
    
    return PHASE_TO_CATEGORY_MAP.get(phase_lower, ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED)


class DrugIndicationEvidenceDataset:
    """A dataset for drug-indication evidence, wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, DrugIndicationEvidence)



class DrugIndicationDataset:
    """A dataset for drug-indication relationships, wrapping a Polars DataFrame."""

    AGGREGATION_FIELDS = {
        "drug_id",
        "disease_id",
        "drug_name",
        "disease_name",
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
    def assign_clinical_status(df: pl.DataFrame) -> "DrugIndicationDataset":
        """Assign harmonized clinical status using Maximum Clinical Development Status logic.
        
        For each drug-indication pair, determines the highest-ranked clinical status
        across all supporting sources based on the harmonisation categories.
        
        Returns:
            DrugIndicationDataset with clinical_status field added
        """
        def get_max_clinical_status(sources_list) -> dict:
            """Get the maximum clinical development status from a list of sources."""
            # Convert Polars list to Python list if needed
            if hasattr(sources_list, 'to_list'):
                sources_list = sources_list.to_list()
            
            if not sources_list or len(sources_list) == 0:
                return {
                    "category": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED.value,
                    "phase": None,
                    "source": None
                }
            
            # Extract all clinical statuses from sources
            clinical_statuses = []
            for source in sources_list:
                phase = source.get("clinical_phase") or source.get("phase")
                source_name = source.get("source")
                
                if source_name:
                    category = map_phase_to_category(phase, source_name)
                    clinical_statuses.append({
                        "category": category.value,
                        "phase": phase,
                        "source": source_name,
                        "rank": CATEGORY_RANKS[category]
                    })
            
            if not clinical_statuses:
                return {
                    "category": ClinicalStatusCategory.NO_DEVELOPMENT_REPORTED.value,
                    "phase": None,
                    "source": None
                }
            
            # Sort by rank (lower number = higher priority) and return the best one
            best_status = min(clinical_statuses, key=lambda x: x["rank"])
            return {
                "category": best_status["category"],
                "phase": best_status["phase"],
                "source": best_status["source"]
            }
        
        # Apply MCDS logic to each drug-indication pair
        df_with_status = df.with_columns([
            pl.col("sources")
            .map_elements(
                get_max_clinical_status,
                return_dtype=pl.Struct([
                    pl.Field("category", pl.String),
                    pl.Field("phase", pl.String),
                    pl.Field("source", pl.String)
                ])
            )
            .alias("clinical_status")
        ])
        
        return DrugIndicationDataset(df=df_with_status)
