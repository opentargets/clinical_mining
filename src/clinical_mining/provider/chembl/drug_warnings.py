"""Extraction of withdrawal clinical reports from ChEMBL drug drug warning dataset."""

import polars as pl

from clinical_mining.dataset import ClinicalReport
from clinical_mining.schemas import (
    ClinicalReportOrigin,
    ClinicalReportType,
    ClinicalSource,
)


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
            id=pl.concat_str(
                # One reference can report multiple withdrawals
                pl.col("ref_id"),
                pl.col("chembl_id"),
            ).chash.sha2_256(),
            phaseFromSource=pl.col("warning_type").str.to_lowercase(),
            type=pl.lit(ClinicalReportType.SAFETY.value),
            origin=pl.when(pl.col("ref_type") == ClinicalSource.DailyMed.value)
            .then(pl.lit(ClinicalReportOrigin.DRUG_LABEL.value))
            .when(
                pl.col("ref_type").is_in(
                    [ClinicalSource.FDA.value, ClinicalSource.EMA.value]
                )
            )
            .then(pl.lit(ClinicalReportOrigin.REGULATORY.value))
            .otherwise(pl.lit(ClinicalReportOrigin.CURATED_RESOURCE.value)),
            # Avoid null objects in sideEffects
            sideEffect=pl.when(
                pl.any_horizontal(
                    pl.coalesce("efo_id", "efo_id_for_warning_class").is_not_null(),
                    pl.coalesce("efo_term", "warning_class").is_not_null(),
                )
            ).then(
                pl.struct(
                    pl.coalesce(
                        pl.col("efo_id"),
                        pl.col("efo_id_for_warning_class"),
                    )
                    .str.replace(":", "_")
                    .alias("diseaseId"),
                    pl.coalesce(
                        pl.col("efo_term"),
                        pl.col("warning_class"),
                    ).alias("diseaseFromSource"),
                )
            ),
            drug=pl.struct(
                pl.col("pref_name").alias("drugFromSource"),
                pl.col("chembl_id").alias("drugId"),
            ),
            year=pl.col("warning_year"),
            countries=pl.col("warning_country").str.split(";"),
            source=pl.col("ref_type"),
            provider=pl.lit("ChEMBL"),
            url=pl.col("ref_url"),
        )
        .unique()
    )

    return ClinicalReport(
        df=(
            reports.group_by(
                [c for c in reports.columns if c not in ["sideEffect", "drug"]]
            )
            .agg(
                pl.col("sideEffect").drop_nulls().unique().alias("sideEffects"),
                pl.col("drug").unique().alias("drugs"),
            )
            .with_columns(
                pl.when(pl.col("sideEffects").list.len() > 0).then(
                    pl.col("sideEffects")
                )
            )
        )
    )
