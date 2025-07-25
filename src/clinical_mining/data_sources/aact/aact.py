"""Utils to transform AACT database to drug/indication relationships."""

import polars as pl
from clinical_mining.dataset import ClinicalStudyDataset, DrugIndicationEvidenceDataset


def process_interventions(
    interventions: pl.DataFrame, browse_interventions: pl.DataFrame
) -> pl.DataFrame:
    """Extract relevant MeSH terms for interventional studies.

    Args:
        interventions (pl.DataFrame): The interventions or exposures (including drugs, medical devices, procedures, vaccines, and other products) of interest to the study, or associated with study arms/groups.
        browse_interventions (pl.DataFrame): MeSH terms that describe the condition(s) being addressed by the clinical trial
    Returns:
        pl.DataFrame: Interventions table with MeSH terms.
    """
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    interventions = (
        interventions.filter(pl.col("intervention_type").is_in(INTERVENTION_WHITELIST))
        .filter(~pl.col("name").str.to_lowercase().str.starts_with("placebo"))
        .unique()
    )
    browse_interventions = browse_interventions.filter(
        pl.col("mesh_type") == "mesh-list"
    ).select(pl.col("nct_id"), pl.col("downcase_mesh_term").alias("mesh_term"))

    return (
        interventions.join(browse_interventions, on="nct_id", how="left")
        .with_columns(drug_name=pl.coalesce(pl.col("mesh_term"), pl.col("name")))
        .rename({"nct_id": "studyId"})
        .drop("name", "mesh_term")
        .unique()
    )


def process_conditions(
    conditions: pl.DataFrame, browse_conditions: pl.DataFrame
) -> pl.DataFrame:
    """Extract relevant MeSH terms for conditions.

    Args:
        conditions (pl.DataFrame): Conditions studied in the clinical trial.
        browse_conditions (pl.DataFrame): MeSH terms for conditions.

    Returns:
        pl.DataFrame: Conditions table with MeSH terms.
    """
    browse_conditions = (
        browse_conditions.join(conditions.select("nct_id"), on="nct_id", how="left")
        .filter(pl.col("mesh_type") == "mesh-list")
        .select(pl.col("nct_id"), pl.col("downcase_mesh_term").alias("mesh_term"))
        .unique()
    )
    return (
        conditions.select(
            pl.col("nct_id").alias("studyId"), pl.col("downcase_name")
        )
        .join(
            browse_conditions.select(
                pl.col("nct_id").alias("studyId"), pl.col("mesh_term")
            ),
            on="studyId",
            how="left",
        )
        .with_columns(
            disease_name=pl.coalesce(pl.col("mesh_term"), pl.col("downcase_name"))
        )
        .drop("downcase_name", "mesh_term")
        .unique()
    )


def extract_clinical_trials(
    studies: pl.DataFrame, additional_metadata: list[pl.DataFrame] | None = None
) -> ClinicalStudyDataset:
    """Return clinical trials with desired extra annotations from other tables.

    Args:
        studies (pl.DataFrame): The studies to process.
        additional_metadata (list[pl.DataFrame] | None): Optional list of DataFrames to join on and add additional metadata.

    Returns:
        ClinicalStudyDataset: The processed studies.
    """
    STUDY_TYPES = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
    studies = studies.filter(pl.col("study_type").is_in(STUDY_TYPES))
    if additional_metadata is not None:
        for metadata_df in additional_metadata:
            studies = studies.join(metadata_df, on="nct_id", how="left")
    return ClinicalStudyDataset(df=studies.rename({"nct_id": "studyId"}))


def extract_drug_indications(
    interventions: pl.DataFrame,
    conditions: pl.DataFrame,
    browse_conditions: pl.DataFrame,
    browse_interventions: pl.DataFrame,
) -> DrugIndicationEvidenceDataset:
    """Extract drug/indication relationships from AACT database.

    Args:
        interventions (pl.DataFrame): Interventions table with interventions or exposures of interest to the study, or associated with study arms/groups.
        conditions (pl.DataFrame): Conditions table with name(s) of the condition(s) studied in the clinical study, or the focus of the clinical study.
        browse_conditions (pl.DataFrame): Browse_conditions table with MeSH terms that describe the condition(s) being addressed by the clinical trial.
        browse_interventions (pl.DataFrame): Browse_interventions table with MeSH terms that describe the intervention(s) of interest to the study, or associated with study arms/groups.
    Returns:
        DrugIndicationEvidenceDataset: The processed drug/indication relationships.
    """
    processed_interventions = process_interventions(interventions, browse_interventions)
    processed_conditions = process_conditions(conditions, browse_conditions)
    return DrugIndicationEvidenceDataset(
        df=(
            processed_interventions.join(
                processed_conditions,
                on="studyId",
                how="inner",
            )
            .with_columns(
                source=pl.lit("AACT"),
                url=pl.concat_str(
                    [pl.lit("https://clinicaltrials.gov/search?term="), pl.col("studyId")],
                    separator="",
                )
            )
            .unique()
        )
    )
