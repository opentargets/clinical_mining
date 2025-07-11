"""Extraction of drug/indication relationships from ChEMBL Indications dataset."""

from pyspark.sql import DataFrame
import pyspark.sql.functions as f


def extract_chembl_indications(raw_indications: DataFrame) -> DataFrame:
    """
    Extract drug/indication relationships from ChEMBL Indications dataset.

    Args:
        raw_indications: ChEMBL Indications dataset in JSON form
    Returns:
        DataFrame with drug/indication relationships
    """
    return (
        raw_indications.select(
            f.explode("_metadata.all_molecule_chembl_ids").alias("drug_id"),
            f.translate("efo_id", ":", "_").alias("disease_id"),
            "indication_refs",
        )
        .withColumn("indication_ref", f.explode("indication_refs"))
        # A drug/disease pair can be supported by multiple trials in a single record (e.g. CHEMBL108/EFO_0004263)
        .withColumn("id", f.explode(f.split(f.col("indication_ref.ref_id"), ",")))
        .withColumn(
            "nct_id",
            f.when(
                f.col("indication_ref.ref_type") == "ClinicalTrials",
                f.col("id"),
            ),
        )
        .withColumn("source", f.col("indication_ref.ref_type"))
        .withColumn(
            "url",
            f.when(
                f.col("source") == "ClinicalTrials",
                f.concat(
                    f.lit("https://clinicaltrials.gov/search?term="), f.col("nct_id")
                ),
            ).otherwise(f.col("indication_ref.ref_url")),
        )
        .drop("indication_refs", "indication_ref")
        .distinct()
    )
