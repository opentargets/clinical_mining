"""Utils to transform AACT database to drug/indication relationships."""

import polars as pl
from clinical_mining.dataset import ClinicalReport
from clinical_mining.schemas import ClinicalReportType


def process_interventions(interventions: pl.DataFrame) -> pl.DataFrame:
    """Extract relevant MeSH terms for interventional studies."""
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    return (
        interventions.with_columns(drugFromSource=pl.col("name").str.to_lowercase())
        .filter(pl.col("intervention_type").is_in(INTERVENTION_WHITELIST))
        .filter(~pl.col("drugFromSource").str.starts_with("placebo"))
        .drop("intervention_type", "name")
        .unique()
    )


def process_conditions(conditions: pl.DataFrame) -> pl.DataFrame:
    """Extract relevant MeSH terms for conditions."""
    return (
        conditions.with_columns(
            diseaseFromSource=pl.col("downcase_name").str.to_lowercase()
        )
        .filter(~pl.col("diseaseFromSource").str.contains("healthy"))
        .drop("downcase_name")
        .unique()
    )


def replace_with_llm_indications(
    studies: pl.DataFrame,
    llm_extraction_df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace diseaseFromSource/drugFromSource with LLM-extracted indications for trials with available results.

    Trials not present in llm_extraction_df get null indications.
    """
    llm_extracted = (
        llm_extraction_df.rename(
            {"id": "nct_id", "diseases": "diseaseFromSource", "drugs": "drugFromSource"}
        )
        .explode("diseaseFromSource")
        .explode("drugFromSource")
    )

    return pl.concat(
        [
            # Trials not covered by LLM extraction get null indications
            studies.join(
                llm_extracted.select("nct_id"),
                left_on=pl.col("nct_id").str.to_uppercase(),
                right_on=pl.col("nct_id").str.to_uppercase(),
                how="anti",
            ).with_columns(
                pl.lit(None, dtype=pl.String).alias("drugFromSource"),
                pl.lit(None, dtype=pl.String).alias("diseaseFromSource"),
            ),
            # Trials covered by LLM extraction get LLM annotations
            studies.drop(["diseaseFromSource", "drugFromSource"])
            .unique()  # to avoid ID duplication due to exploded diseases/drugs
            .join(
                llm_extracted,
                left_on=pl.col("nct_id").str.to_uppercase(),
                right_on=pl.col("nct_id").str.to_uppercase(),
                how="inner",
            )
            .select(studies.columns),  # to keep the same column order
        ]
    )


def extract_clinical_report(
    studies: pl.DataFrame,
    interventions: pl.DataFrame,
    conditions: pl.DataFrame,
    additional_metadata: list[pl.DataFrame] | None = None,
    aggregation_specs: dict[str, dict[str, str]] | None = None,
    detailed_descriptions: pl.DataFrame | None = None,
    llm_extraction_df: pl.DataFrame | None = None,
) -> ClinicalReport:
    """Return clinical trials with desired extra annotations from other tables."""
    STUDY_TYPES = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
    interventions = process_interventions(interventions)
    conditions = process_conditions(conditions)
    studies = studies.join(interventions, on="nct_id", how="left").join(
        conditions, on="nct_id", how="left"
    )
    if llm_extraction_df is not None and not llm_extraction_df.is_empty():
        studies = replace_with_llm_indications(studies, llm_extraction_df)
    if detailed_descriptions is not None:
        studies = studies.join(
            detailed_descriptions.rename({"description": "detailed_description"}),
            on="nct_id",
            how="left",
        )
    if additional_metadata is not None:
        for metadata_df in additional_metadata:
            if aggregation_specs:
                for key, spec in aggregation_specs.items():
                    if key in metadata_df.columns:
                        metadata_df = metadata_df.group_by(spec["group_by"]).agg(
                            pl.col(key).alias(spec["alias"])
                        )
            studies = studies.join(metadata_df, on="nct_id", how="left")

    trial_metadata_cols = [
        c
        for c in studies.columns
        if c not in ["nct_id", "drugFromSource", "diseaseFromSource"]
    ]

    reports = (
        studies.filter(pl.col("study_type").is_in(STUDY_TYPES))
        .filter(pl.col("drugFromSource").is_not_null())
        .rename({col: f"trial_{col}" for col in trial_metadata_cols})
        .rename({"nct_id": "id"})
        .with_columns(
            trial_phase=pl.when(
                (pl.col("trial_phase").is_not_null()) & (pl.col("trial_phase") != "NA")
            )
            .then(pl.col("trial_phase"))
            .otherwise(pl.lit(None))
        )
        .with_columns(
            source=pl.lit("AACT"),
            url=pl.concat_str(
                [pl.lit("https://clinicaltrials.gov/study/"), pl.col("id")],
                separator="",
            ),
            hasExpertReview=pl.lit(False),
            type=pl.lit(ClinicalReportType.CLINICAL_TRIAL),
            phaseFromSource=pl.col("trial_phase"),
            disease=pl.struct(
                pl.lit(None, dtype=pl.String).alias("diseaseId"),
                pl.col("diseaseFromSource"),
            ),
            drug=pl.struct(
                pl.col("drugFromSource"),
                pl.lit(None, dtype=pl.String).alias("drugId"),
            ),
        )
        .drop(["diseaseFromSource", "drugFromSource"])
        .unique()
    )
    return ClinicalReport(
        df=(
            reports.group_by(
                [c for c in reports.columns if c not in ["disease", "drug"]]
            ).agg(
                pl.col("disease").unique().alias("diseases"),
                pl.col("drug").unique().alias("drugs"),
            )
        )
    )
