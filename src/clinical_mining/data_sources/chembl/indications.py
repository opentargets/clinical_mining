"""Extraction of drug/indication relationships from ChEMBL Indications dataset."""

from pyspark.sql import DataFrame
import pyspark.sql.functions as f

from clinical_mining.dataset import DrugIndicationEvidenceDataset


def extract_chembl_indications(
    raw_indications: DataFrame, exclude_trials: bool = False
) -> DrugIndicationEvidenceDataset:
    """
    Extract drug/indication relationships from ChEMBL Indications dataset.

    Args:
        raw_indications: ChEMBL Indications dataset in JSON form
    Returns:
        DrugIndicationEvidenceDataset: Dataset with drug/indication relationships
    """

    indications = (
        raw_indications.select(
            f.explode("_metadata.all_molecule_chembl_ids").alias("drug_id"),
            f.translate("efo_id", ":", "_").alias("disease_id"),
            "indication_refs",
        )
        .withColumn("indication_ref", f.explode("indication_refs"))
        # A drug/disease pair can be supported by multiple trials in a single record (e.g. CHEMBL108/EFO_0004263)
        .withColumn("studyId", f.explode(f.split(f.col("indication_ref.ref_id"), ",")))
        .withColumn("source", f.col("indication_ref.ref_type"))
        .withColumn(
            "url",
            f.when(
                f.col("source") == "ClinicalTrials",
                f.concat(f.lit("https://clinicaltrials.gov/search?term="), f.col("studyId")),
            ).otherwise(f.col("indication_ref.ref_url")),
        )
        .drop("indication_refs", "indication_ref")
        .distinct()
    )
    if exclude_trials:
        return indications.filter(f.col("source") != "ClinicalTrials")
    return DrugIndicationEvidenceDataset(df=indications)
