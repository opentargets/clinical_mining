"""Utils to transform AACT database to drug/indication relationships."""

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
import pyspark.sql.functions as f

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def process_interventions(
    interventions: DataFrame, browse_interventions: DataFrame
) -> DataFrame:
    """Extract relevant MeSH terms for interventional studies.

    Args:
        interventions (DataFrame): The interventions or exposures (including drugs, medical devices, procedures, vaccines, and other products) of interest to the study, or associated with study arms/groups.
        browse_interventions (DataFrame): MeSH terms that describe the condition(s) being addressed by the clinical trial
    Returns:
        DataFrame: Interventions table with MeSH terms
    """
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    interventions = (
        interventions.filter(f.col("intervention_type").isin(INTERVENTION_WHITELIST))
        # Drop studies where the intervention is placebo
        .filter(~f.lower("name").startswith("placebo"))
        .distinct()
    )
    browse_interventions = browse_interventions.filter(
        f.col("mesh_type") == "mesh-list"
    ).selectExpr("nct_id", "downcase_mesh_term as mesh_term")

    return (
        interventions.join(browse_interventions, "nct_id", "left")
        .withColumn("drug_name", f.coalesce(f.col("mesh_term"), f.col("name")))
        .drop("name", "mesh_term")
        .distinct()
    )


def process_conditions(
    conditions: DataFrame, browse_conditions: DataFrame
) -> DataFrame:
    browse_conditions = (
        browse_conditions.join(conditions.select("nct_id"), "nct_id", "left").filter(f.col("mesh_type") == "mesh-list")
        .selectExpr("nct_id", "downcase_mesh_term as mesh_term")
        .distinct()
    )
    return (
        conditions.selectExpr("nct_id", "downcase_name")
        .join(browse_conditions.selectExpr("nct_id", "mesh_term"), "nct_id", "left")
        .withColumn(
            "disease_name", f.coalesce(f.col("mesh_term"), f.col("downcase_name"))
        )
        .drop("downcase_name", "mesh_term")
        .distinct()
    )


def extract_clinical_trials(
    studies: DataFrame, additional_metadata: list[DataFrame] | None = None
) -> DataFrame:
    """Return clinical trials with desired extra annotations from other tables.

    Args:
        studies (DataFrame): The studies to process
        additional_metadata (list[DataFrame] | None): Optional list of DataFrames to join on and add additional trial metadata
    Returns:
        DataFrame: The processed studies
    """
    STUDY_TYPES = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
    studies = studies.filter(f.col("study_type").isin(STUDY_TYPES))
    if additional_metadata is not None:
        for metadata_df in additional_metadata:
            studies = studies.join(metadata_df, on="nct_id", how="left")
    return studies


def extract_drug_indications(
    studies: DataFrame,
    interventions: DataFrame,
    conditions: DataFrame,
    browse_conditions: DataFrame,
    browse_interventions: DataFrame,
) -> DataFrame:
    processed_interventions = process_interventions(interventions, browse_interventions)
    processed_conditions = process_conditions(conditions, browse_conditions)
    return (
        studies.join(
            processed_interventions,
            on="nct_id",
            how="inner",
        )
        .join(
            processed_conditions,
            on="nct_id",
            how="inner",
        )
        .withColumn("source", f.lit("AACT"))
        .withColumn(
            "url",
            f.concat(f.lit("https://clinicaltrials.gov/search?term="), f.col("nct_id")),
        )
        .distinct()
    )
