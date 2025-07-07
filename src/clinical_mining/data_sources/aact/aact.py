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
    interventions = (
        interventions.filter(f.col("intervention_type") == "DRUG")
        # Drop studies where the intervention is placebo
        .filter(~f.lower("name").contains("placebo"))
        .distinct()
    )
    browse_interventions = (
        browse_interventions
        # filter list of studies
        .join(interventions.select("nct_id"), "nct_id")
        # Get the direct mapping
        .filter(f.col("mesh_type") == "mesh-list")
        .selectExpr("nct_id", "downcase_mesh_term as mesh_term")
        .distinct()
    )
    return (
        interventions.selectExpr("nct_id", "name")
        .join(browse_interventions.selectExpr("nct_id", "mesh_term"), "nct_id", "left")
        .withColumn("drug_name", f.coalesce(f.col("mesh_term"), f.col("name")))
        .drop("name", "mesh_term")
        .distinct()
    )


def process_conditions(
    conditions: DataFrame, browse_conditions: DataFrame
) -> DataFrame:
    browse_conditions = (
        browse_conditions.filter(f.col("mesh_type") == "mesh-list")
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


def extract_aact_indications(
    studies: DataFrame,
    interventions: DataFrame,
    conditions: DataFrame,
    browse_conditions: DataFrame,
    browse_interventions: DataFrame,
    study_references: DataFrame,
    study_design: DataFrame,
) -> DataFrame:
    processed_interventions = process_interventions(interventions, browse_interventions)
    processed_conditions = process_conditions(conditions, browse_conditions)
    return (
        studies.filter(f.col("study_type") == "INTERVENTIONAL")
        .join(
            processed_interventions,
            on="nct_id",
            how="inner",
        )
        .join(
            processed_conditions,
            on="nct_id",
            how="inner",
        )
        .join(
            study_design.selectExpr("nct_id", "primary_purpose as purpose"),
            on="nct_id",
            how="inner",
        )
        .join(
            study_references,
            on="nct_id",
            how="left",
        )
        .withColumn("source", f.lit("AACT"))
        .withColumn(
            "url",
            f.concat(f.lit("https://clinicaltrials.gov/search?term="), f.col("nct_id")),
        )
        .drop("study_type")
        .distinct()
    )
