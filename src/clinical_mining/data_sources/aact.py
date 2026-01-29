"""Utils to transform AACT database to drug/indication relationships."""

import polars as pl
from clinical_mining.dataset import ClinicalReport
from clinical_mining.schemas import ClinicalReportType


def process_interventions(interventions: pl.DataFrame) -> pl.DataFrame:
    """Extract relevant MeSH terms for interventional studies.

    Args:
        interventions (pl.DataFrame): The interventions or exposures (including drugs, medical devices, procedures, vaccines, and other products) of interest to the study, or associated with study arms/groups.
    Returns:
        pl.DataFrame: Interventions table with MeSH terms indexed on nct_id.
    """
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    return (
        interventions.with_columns(drugFromSource=pl.col("name").str.to_lowercase())
        .filter(pl.col("intervention_type").is_in(INTERVENTION_WHITELIST))
        .filter(~pl.col("drugFromSource").str.starts_with("placebo"))
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
        pl.DataFrame: Conditions table with MeSH terms indexed on nct_id.
    """
    return (
        conditions.with_columns(
            diseaseFromSource=pl.col("downcase_name").str.to_lowercase()
        )
        .filter(~pl.col("diseaseFromSource").str.contains("healthy"))
        .drop("downcase_name")
        .unique()
    )


def extract_clinical_report(
    studies: pl.DataFrame,
    interventions: pl.DataFrame,
    conditions: pl.DataFrame,
    additional_metadata: list[pl.DataFrame] | None = None,
    aggregation_specs: dict[str, dict[str, str]] | None = None,
) -> ClinicalReport:
    """Return clinical trials with desired extra annotations from other tables.

    Args:
        studies (pl.DataFrame): The studies to process
        interventions (pl.DataFrame): The interventions to process
        conditions (pl.DataFrame): The conditions to process
        additional_metadata (list[pl.DataFrame] | None): Optional list of DataFrames to join on and add additional metadata.
        aggregation_specs (dict[str, dict[str, str]] | None): Optional dictionary of aggregation specifications for the additional metadata DataFrames.

    Returns:
        ClinicalReport: The processed studies.
    """
    STUDY_TYPES = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
    interventions = process_interventions(interventions)
    conditions = process_conditions(conditions)
    studies = studies.join(interventions, on="nct_id", how="left").join(
        conditions, on="nct_id", how="left"
    )
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
                [
                    pl.lit("https://clinicaltrials.gov/study/"),
                    pl.col("id"),
                ],
                separator="",
            ),
            hasExpertReview=pl.lit(False),
            type=pl.lit(ClinicalReportType.CLINICAL_TRIAL),
            phaseFromSource=pl.col("trial_phase"),
        )
        .unique()
    )
    mapped_reports = (
        # TODO: call mapping function
        reports.with_columns(
            disease=pl.struct(
                pl.col("diseaseFromSource"), pl.lit("CHEMBL_TO_DO").alias("diseaseId")
            ),
            drug=pl.struct(
                pl.col("drugFromSource"), pl.lit("EFO_TO_DO").alias("drugId")
            ),
        )
        .drop(["diseaseFromSource", "drugFromSource"])
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
