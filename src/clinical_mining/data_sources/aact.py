"""Utils to transform AACT database to drug/indication relationships."""

import polars as pl
from clinical_mining.dataset import ClinicalEvidence, ClinicalReport


def process_interventions(interventions: pl.DataFrame) -> pl.DataFrame:
    """Extract relevant MeSH terms for interventional studies.

    Args:
        interventions (pl.DataFrame): The interventions or exposures (including drugs, medical devices, procedures, vaccines, and other products) of interest to the study, or associated with study arms/groups.
    Returns:
        pl.DataFrame: Interventions table with MeSH terms.
    """
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    return (
        interventions.with_columns(drugFromSource=pl.col("name").str.to_lowercase())
        .filter(pl.col("intervention_type").is_in(INTERVENTION_WHITELIST))
        .filter(~pl.col("drugFromSource").str.starts_with("placebo"))
        .rename({"nct_id": "studyId"})
        .drop("intervention_type", "name")
        .unique()
    )


def process_conditions(
    conditions: pl.DataFrame,
) -> pl.DataFrame:
    """Extract relevant MeSH terms for conditions.

    Args:
        conditions (pl.DataFrame): Conditions studied in the clinical trial.

    Returns:
        pl.DataFrame: Conditions table with MeSH terms.
    """
    return (
        conditions.with_columns(
            diseaseFromSource=pl.col("downcase_name").str.to_lowercase()
        )
        .filter(~pl.col("diseaseFromSource").str.contains("healthy"))
        .rename({"nct_id": "studyId"})
        .drop("downcase_name")
        .unique()
    )


def extract_clinical_record(
    studies: pl.DataFrame,
    additional_metadata: list[pl.DataFrame] | None = None,
    aggregation_specs: dict[str, dict[str, str]] | None = None,
) -> ClinicalReport:
    """Return clinical trials with desired extra annotations from other tables.

    Args:
        studies (pl.DataFrame): The studies to process.
        additional_metadata (list[pl.DataFrame] | None): Optional list of DataFrames to join on and add additional metadata.
        aggregation_specs (dict[str, dict[str, str]] | None): Optional dictionary of aggregation specifications for the additional metadata DataFrames.

    Returns:
        ClinicalReport: The processed studies.
    """
    STUDY_TYPES = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
    studies = studies.filter(pl.col("study_type").is_in(STUDY_TYPES))
    if additional_metadata is not None:
        for metadata_df in additional_metadata:
            # Check if any of the metadata needs aggregating before joining
            if aggregation_specs:
                for key, spec in aggregation_specs.items():
                    if key in metadata_df.columns:
                        metadata_df = metadata_df.group_by(spec["group_by"]).agg(
                            pl.col(key).alias(spec["alias"])
                        )

            studies = studies.join(metadata_df, on="nct_id", how="left")

    # Add `trial_` prefix to all trial metadata columns
    trial_metadata_cols = [c for c in studies.columns if c not in ["nct_id", "phase"]]
    return ClinicalReport(
        df=(
            studies
            .rename({col: f"trial_{col}" for col in trial_metadata_cols})
            .rename({"nct_id": "studyId"})
        )
    )


def extract_clinical_indication(
    interventions: pl.DataFrame,
    conditions: pl.DataFrame,
) -> ClinicalEvidence:
    """Extract drug/indication relationships from AACT database.

    Args:
        interventions (pl.DataFrame): Interventions table with interventions or exposures of interest to the study, or associated with study arms/groups.
        conditions (pl.DataFrame): Conditions table with name(s) of the condition(s) studied in the clinical study, or the focus of the clinical study.
    Returns:
        ClinicalEvidence: The processed drug/indication relationships.
    """
    processed_interventions = process_interventions(interventions)
    processed_conditions = process_conditions(conditions)
    return ClinicalEvidence(
        df=(
            processed_interventions.join(
                processed_conditions,
                on="studyId",
                how="inner",
            )
            .with_columns(
                source=pl.lit("AACT"),
                url=pl.concat_str(
                    [
                        pl.lit("https://clinicaltrials.gov/search?term="),
                        pl.col("studyId"),
                    ],
                    separator="",
                ),
            )
            .unique()
        )
    )
