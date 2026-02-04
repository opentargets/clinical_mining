"""Extraction of drug/indication relationships from ChEMBL Indications dataset."""

import polars as pl

from clinical_mining.schemas import ClinicalReportType
from clinical_mining.dataset import ClinicalReport


def extract_clinical_report(
    indications: pl.DataFrame,
    molecule: pl.DataFrame,
    indications_refs: pl.DataFrame,
) -> ClinicalReport:
    """
    Extract clinical reports from the curation ChEMBL does for drugs and clinical candidates.
    Sources include: FDA, EMA, WHO ATC, ClinicalTrials.gov, INN, USAN.

    Args:
        indications: `drug_indications` table from ChEMBL
        molecule: `molecule_dictionary` table from ChEMBL
        indications_refs: `drug_indication_refs` table from ChEMBL

    Returns:
        ClinicalReport: Dataset with drug/indication relationships
    """

    reports = (
        indications.join(molecule, "molregno")
        .join(indications_refs, "drugind_id")
        .filter(pl.col("efo_id").is_not_null())
        # Some EMA references that are duplicated
        .filter(~pl.col("ref_url").str.starts_with("www"))
        .select(
            pl.col("ref_id").str.split(",").alias("id"),
            pl.col("max_phase_for_ind")
            .cast(pl.Float16)
            .cast(pl.String)
            .alias("phaseFromSource"),
            pl.when(pl.col("ref_type") == "ClinicalTrials")
            .then(pl.lit(ClinicalReportType.CLINICAL_TRIAL))
            .when(pl.col("ref_type") == "DailyMed")
            .then(pl.lit(ClinicalReportType.DRUG_LABEL))
            .otherwise(pl.lit(ClinicalReportType.CURATED_RESOURCE))
            .alias("type"),
            pl.when(pl.col("ref_type") == "ClinicalTrials")
            .then(
                pl.concat_str(
                    pl.lit("https://clinicaltrials.gov/study/"), pl.col("ref_id")
                )
            )
            .otherwise(pl.col("ref_url"))
            .alias("url"),
            pl.col("ref_type").alias("source"),
            pl.struct(
                pl.col("efo_id").str.replace(":", "_").alias("diseaseId"),
                pl.col("efo_term").str.to_lowercase().alias("diseaseFromSource"),
            ).alias("disease"),
            pl.struct(
                pl.lit(None).alias("drugFromSource"),
                pl.col("chembl_id").alias("drugId"),
            ).alias("drug"),
        )
        .explode("id")
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
