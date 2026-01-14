import polars as pl

from clinical_mining.dataset import DrugIndicationEvidenceDataset


def extract_ema_indications(
    indications_path: str,
) -> DrugIndicationEvidenceDataset:
    """Extract drug/indication relationships from the EMA list of human drugs."""
    raw = pl.read_excel(
        indications_path,
        sheet_name="Medicine",
    )
    raw.columns = raw.iter_rows().__next__()  # Assign columns names from first row
    return DrugIndicationEvidenceDataset(
        df=(
            raw.slice(1)  # drop header
            .filter(pl.col("Category") == "Human")
            .select(
                drug_name=pl.coalesce(
                    "International non-proprietary name (INN) / common name",
                    "Active substance",
                    "Name of medicine",
                )
                .str.to_lowercase()
                .str.split(";"),
                disease_name=pl.col("Therapeutic area (MeSH)")
                .str.to_lowercase()
                .str.split(";"),
                phase=pl.col("Medicine status").str.to_lowercase(),
                studyId=pl.col("EMA product number").str.to_lowercase(),
                source=pl.lit("EMA Human Drugs"),
            )
            .explode("drug_name")
            .explode("disease_name")
            # 66 rows do not report a MeSH term
            # TODO: Use NER to extract indications from the `Therapeutic indication` column 
            .filter(pl.col("disease_name").is_not_null())
        )
    )
