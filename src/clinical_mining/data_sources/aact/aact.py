"""Utils to transform AACT database to drug/indication relationships."""

import polars as pl
from clinical_mining.dataset import ClinicalStudyDataset, DrugIndicationEvidenceDataset


def process_interventions(
    interventions: pl.DataFrame
) -> pl.DataFrame:
    """Extract relevant MeSH terms for interventional studies.

    Args:
        interventions (pl.DataFrame): The interventions or exposures (including drugs, medical devices, procedures, vaccines, and other products) of interest to the study, or associated with study arms/groups.
    Returns:
        pl.DataFrame: Interventions table with MeSH terms.
    """
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    return (
        interventions.filter(pl.col("intervention_type").is_in(INTERVENTION_WHITELIST))
        .filter(~pl.col("name").str.to_lowercase().str.starts_with("placebo"))
        .rename({"nct_id": "studyId", "name": "drug_name"})
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
    return conditions.rename({"nct_id": "studyId", "downcase_name": "disease_name"}).unique()


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
) -> DrugIndicationEvidenceDataset:
    """Extract drug/indication relationships from AACT database.

    Args:
        interventions (pl.DataFrame): Interventions table with interventions or exposures of interest to the study, or associated with study arms/groups.
        conditions (pl.DataFrame): Conditions table with name(s) of the condition(s) studied in the clinical study, or the focus of the clinical study.
    Returns:
        DrugIndicationEvidenceDataset: The processed drug/indication relationships.
    """
    processed_interventions = process_interventions(interventions)
    processed_conditions = process_conditions(conditions)
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
