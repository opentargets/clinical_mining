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
            pl.col("_metadata").struct.field("all_molecule_chembl_ids").alias("drug_ids"),
            pl.col("efo_id").str.replace(":", "_").alias("disease_id"),
            "indication_refs",
        )
        .explode("indication_refs")
        .with_columns(
            pl.col("indication_refs").struct.field("ref_id").str.split(",").alias("studyId"),
            pl.col("indication_refs").struct.field("ref_type").alias("source"),
            pl.col("indication_refs").struct.field("ref_url").alias("url"),
        )
        .explode("studyId")
        .explode("drug_ids")  # Explode drug_ids after all other explodes
        .rename({"drug_ids": "drug_id"})
        .with_columns(
            url=pl.when(pl.col("source") == "ClinicalTrials")
            .then(pl.concat_str([pl.lit("https://clinicaltrials.gov/search?term="), pl.col("studyId")]))
            .otherwise(pl.col("url"))
        )
        .drop("indication_refs")
        .unique()
    )

    if exclude_trials:
        indications = indications.filter(pl.col("source") != "ClinicalTrials")

    return DrugIndicationEvidenceDataset(df=indications)