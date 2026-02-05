"""Extraction of clinical reports from ChEMBL drug/indications dataset."""

import polars as pl

from clinical_mining.schemas import ClinicalReportType
from clinical_mining.dataset import ClinicalReport


def extract_clinical_report(
    drug_indication: pl.DataFrame,
    molecule_dictionary: pl.DataFrame,
    indication_refs: pl.DataFrame,
) -> ClinicalReport:
    """
    Extract clinical reports from the curation ChEMBL does for drugs and clinical candidates.
    Sources include: FDA, EMA, WHO ATC, ClinicalTrials.gov, INN, USAN.

    Args:
        drug_indication: `drug_indication` table from ChEMBL
        molecule_dictionary: `molecule_dictionary` table from ChEMBL
        indication_refs: `indication_refs` table from ChEMBL

    Returns:
        ClinicalReport: Dataset with drug/indication relationships
    """

    reports = (
        drug_indication.join(molecule_dictionary, "molregno")
        .join(indication_refs, "drugind_id")
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
            pl.lit(True).alias("hasExpertReview"),
            pl.col("ref_type").alias("source"),
            pl.struct(
                pl.col("efo_id").str.replace(":", "_").alias("diseaseId"),
                pl.col("efo_term").str.to_lowercase().alias("diseaseFromSource"),
            ).alias("disease"),
            pl.struct(
                pl.lit(None, dtype=pl.String).alias("drugFromSource"),
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
