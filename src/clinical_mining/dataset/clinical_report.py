import polars as pl

from clinical_mining.schemas import (
    validate_schema,
    ClinicalReportSchema,
    ClinicalStageCategory,
    snake_to_camel,
)
from clinical_mining.dataset.clinical_indication import CATEGORY_RANKS_STR
from clinical_mining.utils.mapping import map_entities
from pyspark.sql import DataFrame, SparkSession

# Clinical status harmonization constants
PHASE_TO_CATEGORY_MAP = {
    # WITHDRAWN (Rank 1)
    "withdrawn": ClinicalStageCategory.WITHDRAWN,
    "withdrawn from market": ClinicalStageCategory.WITHDRAWN,
    "revoked": ClinicalStageCategory.WITHDRAWN,
    "expired": ClinicalStageCategory.WITHDRAWN,
    "lapsed": ClinicalStageCategory.WITHDRAWN,
    "suspended": ClinicalStageCategory.WITHDRAWN,
    # PHASE_4 (Rank 2)
    "phase4": ClinicalStageCategory.PHASE_4,
    "phase 4": ClinicalStageCategory.PHASE_4,
    "discontinued in phase 4": ClinicalStageCategory.PHASE_4,
    # APPROVED (Rank 3)
    "approved": ClinicalStageCategory.APPROVED,
    "authorised": ClinicalStageCategory.APPROVED,
    "approved (orphan drug)": ClinicalStageCategory.APPROVED,
    "approved in china": ClinicalStageCategory.APPROVED,
    "approved in eu": ClinicalStageCategory.APPROVED,
    "registered": ClinicalStageCategory.APPROVED,
    # PREAPPROVAL (Rank 4)
    "preregistration": ClinicalStageCategory.PREAPPROVAL,
    "application submitted": ClinicalStageCategory.PREAPPROVAL,
    "approval submitted": ClinicalStageCategory.PREAPPROVAL,
    "nda filed": ClinicalStageCategory.PREAPPROVAL,
    "bla submitted": ClinicalStageCategory.PREAPPROVAL,
    "opinion": ClinicalStageCategory.PREAPPROVAL,
    "opinion under re-examination": ClinicalStageCategory.PREAPPROVAL,
    "discontinued in preregistration": ClinicalStageCategory.PREAPPROVAL,
    # PHASE_3 (Rank 5)
    "phase3": ClinicalStageCategory.PHASE_3,
    "phase 3": ClinicalStageCategory.PHASE_3,
    "discontinued in phase 3": ClinicalStageCategory.PHASE_3,
    # PHASE_2_3 (Rank 6)
    "phase2/phase3": ClinicalStageCategory.PHASE_2_3,
    "phase 2/3": ClinicalStageCategory.PHASE_2_3,
    "discontinued in phase 2/3": ClinicalStageCategory.PHASE_2_3,
    # PHASE_2 (Rank 7)
    "phase2": ClinicalStageCategory.PHASE_2,
    "phase 2": ClinicalStageCategory.PHASE_2,
    "phase 2a": ClinicalStageCategory.PHASE_2,
    "phase 2b": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 2": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 2a": ClinicalStageCategory.PHASE_2,
    "discontinued in phase 2b": ClinicalStageCategory.PHASE_2,
    # PHASE_1_2 (Rank 8)
    "phase1/phase2": ClinicalStageCategory.PHASE_1_2,
    "phase 1/2": ClinicalStageCategory.PHASE_1_2,
    "phase 1b/2a": ClinicalStageCategory.PHASE_1_2,
    "phase 1/2a": ClinicalStageCategory.PHASE_1_2,
    "discontinued in phase 1/2": ClinicalStageCategory.PHASE_1_2,
    # PHASE_1 (Rank 9)
    "phase1": ClinicalStageCategory.PHASE_1,
    "phase 1": ClinicalStageCategory.PHASE_1,
    "phase 1b": ClinicalStageCategory.PHASE_1,
    "discontinued in phase 1": ClinicalStageCategory.PHASE_1,
    # EARLY_PHASE_1 (Rank 10)
    "early_phase1": ClinicalStageCategory.EARLY_PHASE_1,
    "phase 0": ClinicalStageCategory.EARLY_PHASE_1,
    # IND (Rank 11)
    "ind submitted": ClinicalStageCategory.IND,
    "investigative": ClinicalStageCategory.IND,
    # PRECLINICAL (Rank 12)
    "preclinical": ClinicalStageCategory.PRECLINICAL,
    "patented": ClinicalStageCategory.PRECLINICAL,
    # UNKNOWN (Rank 13)
    "clinical trial": ClinicalStageCategory.UNKNOWN,
    "terminated": ClinicalStageCategory.UNKNOWN,
    "application withdrawn": ClinicalStageCategory.UNKNOWN,
    "refused": ClinicalStageCategory.UNKNOWN,
    "withdrawn from rolling review": ClinicalStageCategory.UNKNOWN,
    "NA": ClinicalStageCategory.UNKNOWN,
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
    if source in APPROVED_SOURCES:
        return ClinicalStageCategory.APPROVED

    # Handle case-insensitive mapping
    phase_lower = phase.lower() if isinstance(phase, str) else str(phase).lower()

    return PHASE_TO_CATEGORY_MAP.get(phase_lower, ClinicalStageCategory.UNKNOWN)


class ClinicalReport:
    """A dataset for clinical reports (e.g. clinical trial, an USAN reference), wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initialises the dataset, validating and aligning the DataFrame."""
        # Harmonise column names from snake to camel case
        df = df.rename({col: snake_to_camel(col) for col in df.columns})

        # Assign clinical stage
        df = df.with_columns(
            clinicalStage=pl.struct(["phaseFromSource", "source"]).map_elements(
                lambda row: map_phase_to_category(
                    row["phaseFromSource"], row["source"]
                ),
                return_dtype=pl.String,
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
                    default=CATEGORY_RANKS_STR[ClinicalStageCategory.UNKNOWN.value],
                )
            )
            .sort(["id", "clinicalStageRank"])
            .unique(subset=["id"], keep="first")
            .drop("clinicalStageRank")
        )

    @classmethod
    def map_entities(
        cls,
        spark: SparkSession,
        reports: pl.DataFrame,
        disease_index: DataFrame,
        drug_index: DataFrame,
        chembl_curation: pl.DataFrame,
        drug_column_name: str,
        disease_column_name: str,
        drug_id_column_name: str = "drugId",
        disease_id_column_name: str = "diseaseId",
        ner_extract_drug: bool = True,
        ner_batch_size: int = 256,
        ner_cache_path: str | None = None,
    ) -> "ClinicalReport":
        """Map entities to IDs."""
        # Explode clinical reports to get lists of study/disease/drug
        exploded_reports = (
            reports.explode("drugs").explode("diseases").unnest(["drugs", "diseases"])
        )

        mapped_exploded_reports = map_entities(
            spark,
            exploded_reports,
            disease_index,
            drug_index,
            chembl_curation,
            drug_column_name,
            disease_column_name,
            drug_id_column_name,
            disease_id_column_name,
            ner_extract_drug,
            ner_batch_size,
            ner_cache_path,
        )

        mapped_reports = (
            mapped_exploded_reports.with_columns(
                disease=pl.struct(pl.col("diseaseFromSource"), pl.col("diseaseId")),
                drug=pl.struct(pl.col("drugFromSource"), pl.col("drugId")),
            )
            .drop(["diseaseFromSource", "drugFromSource", "diseaseId", "drugId"])
            .unique()
        )

        return ClinicalReport(
            df=(
                mapped_reports.group_by(
                    [c for c in mapped_reports.columns if c not in ["disease", "drug"]]
                ).agg(
                    pl.col("disease").unique().alias("diseases"),
                    pl.col("drug").unique().alias("drugs"),
                )
            )
        )
