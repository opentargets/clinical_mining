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
            raw.slice(1)  # omit header
            .select(
                drug_name=pl.col("Active substance").str.to_lowercase().str.split(";"),
                disease_name=pl.col("Therapeutic area (MeSH)")
                .str.to_lowercase()
                .str.split(";"),
                overall_status=pl.col("Medicine status").str.to_lowercase(),
                studyId=pl.col("EMA product number").str.to_lowercase(),
                source=pl.lit("EMA Human Drugs"),
            )
            .explode("drug_name")
            .explode("disease_name")
        )
    )
