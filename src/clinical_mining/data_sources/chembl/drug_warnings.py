"""Extraction of withdrawal clinical reports from ChEMBL drug drug warning dataset."""

import polars as pl

from clinical_mining.schemas import ClinicalReportType
from clinical_mining.dataset import ClinicalReport


def extract_clinical_report(
    drug_warning: pl.DataFrame,
    molecule_dictionary: pl.DataFrame,
    warning_refs: pl.DataFrame,
) -> ClinicalReport:
    """Extract clinical reports from ChEMBL drug warning dataset.

    Args:
        drug_warning: Drug warning dataset
        molecule_dictionary: Molecule dictionary dataset
        warning_refs: Warning references dataset

    Returns:
        ClinicalReport object with extracted reports
    """

    reports = (
        drug_warning.join(molecule_dictionary, "molregno")
        .join(warning_refs, "warning_id")
        .select(
            pl.concat_str(
                # One reference can report multiple withdrawals
                pl.col("ref_id"),
                pl.col("chembl_id"),
            ).chash.sha2_256().alias("id"),
            pl.col("warning_type").str.to_lowercase().alias("phaseFromSource"),
            pl.lit(ClinicalReportType.CURATED_RESOURCE).alias("type"),
            pl.struct(
                pl.col("efo_id").str.replace(":", "_").alias("diseaseId"),
                pl.col("efo_term").alias("diseaseFromSource"),
            ).alias("sideEffect"),
            pl.struct(
                pl.lit(None).alias("drugFromSource"),
                pl.col("chembl_id").alias("drugId"),
            ).alias("drug"),
            pl.col("warning_year").alias("year"),
            pl.col("warning_country").str.split(";").alias("countries"),
            pl.col("ref_type").alias("source"),
            pl.lit(True).alias("hasExpertReview"),
            pl.col("ref_url").alias("url"),
        )
        .unique()
    )

    return ClinicalReport(
        df=(
            reports.group_by(
                [c for c in reports.columns if c not in ["sideEffect", "drug"]]
            ).agg(
                pl.col("sideEffect").unique().alias("sideEffects"),
                pl.col("drug").unique().alias("drugs"),
            )
        )
    )
