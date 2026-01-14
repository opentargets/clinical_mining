"""Extraction of drug/indication relationships from ChEMBL Indications dataset."""

import polars as pl

from clinical_mining.dataset import DrugIndicationEvidenceDataset


def extract_chembl_indications(
    raw_indications: pl.DataFrame, exclude_trials: bool = False
) -> DrugIndicationEvidenceDataset:
    """
    Extract drug/indication relationships from ChEMBL Indications JSON file storing indications for drugs, and clinical candidate drugs, from a variety of sources (e.g., FDA, EMA, WHO ATC, ClinicalTrials.gov, INN, USAN).

    Args:
        raw_indications: ChEMBL Indications dataset in JSON form
    Returns:
        DrugIndicationEvidenceDataset: Dataset with drug/indication relationships
    """

    indications = (
        raw_indications.select(
            pl.col("_metadata")
            .struct.field("all_molecule_chembl_ids")
            .alias("drug_ids"),
            pl.col("efo_id").str.replace(":", "_").alias("disease_id"),
            "indication_refs",
            "max_phase_for_ind",
        )
        .explode("indication_refs")
        .with_columns(
            studyId=pl.col("indication_refs").struct.field("ref_id").str.split(","),
            source=pl.col("indication_refs").struct.field("ref_type"),
            url=pl.col("indication_refs").struct.field("ref_url"),
        )
        .explode("studyId")
        .explode("drug_ids")  # Explode drug_ids after all other explodes
        .rename({"drug_ids": "drug_id"})
        .with_columns(
            url=pl.when(pl.col("source") == "ClinicalTrials")
            .then(
                pl.concat_str(
                    [
                        pl.lit("https://clinicaltrials.gov/search?term="),
                        pl.col("studyId"),
                    ]
                )
            )
            .otherwise(pl.col("url")),
            clinical_phase=pl.when(pl.col("source") == "ClinicalTrials")
            .then(pl.col("max_phase_for_ind"))
            .when(pl.col("source").is_in(["EMA", "FDA", "DailyMed", "ATC"]))
            .then(pl.lit("approved"))
            .otherwise(pl.lit(None)),
        )
        .drop("indication_refs", "max_phase_for_ind")
        # Not all evidence are mapped. We drop those since we lack drug/disease labels
        .filter(
            (pl.col("disease_id").is_not_null()) & (pl.col("drug_id").is_not_null())
        )
        .unique()
    )

    if exclude_trials:
        indications = indications.filter(pl.col("source") != "ClinicalTrials")

    return DrugIndicationEvidenceDataset(df=indications)
