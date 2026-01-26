import polars as pl

from ontoma.ner.disease import extract_disease_entities

from clinical_mining.dataset import ClinicalEvidence
from clinical_mining.utils.polars_helpers import convert_polars_to_spark
from clinical_mining.utils.spark_helpers import spark_session


def extract_clinical_indication(
    indications_path: str,
    spark: spark_session,
) -> ClinicalEvidence:
    """Extract drug/indication relationships from the EMA list of human drugs."""
    raw = pl.read_excel(
        indications_path,
        sheet_name="Medicine",
    )
    raw.columns = raw.iter_rows().__next__()  # Assign columns names from first row

    human_indications = (
        raw.slice(1)
        .filter(  # drop header
            pl.col("Category") == "Human"
        )
    )

    # Drop columns with all nulls to convert to spark
    non_empty_cols = [
        series.name for series in human_indications.iter_columns() if series.null_count() < human_indications.height
    ]
    ner_extracted_indication = (
        pl.from_pandas(
            extract_disease_entities(
                spark,
                df=convert_polars_to_spark(
                    polars_df=human_indications.select(non_empty_cols).with_columns(
                        therapeutic_indication=pl.col("Therapeutic indication")
                        .fill_null("")
                        .str.strip_chars()
                    ),
                    spark=spark,
                ),
                input_col="therapeutic_indication",
                output_col="extracted_diseases",
            ).toPandas()
        )
        # Explode the extracted diseases
        .explode("extracted_diseases")
        .rename({"extracted_diseases": "extracted_disease"})
    )

    return ClinicalEvidence(
        df=ner_extracted_indication.select(
            drugFromSource=pl.coalesce(
                "International non-proprietary name (INN) / common name",
                "Active substance",
                "Name of medicine",
            )
            .str.to_lowercase()
            .str.split(";"),
            diseaseFromSource=pl.coalesce(
                # Prioritise MeSH terms over automatically extracted diseases
                "Therapeutic area (MeSH)",
                "extracted_disease",
            )
            .str.to_lowercase()
            .str.split(";"),
            phase=pl.col("Medicine status").str.to_lowercase(),
            studyId=pl.col("EMA product number").str.to_lowercase(),
            source=pl.lit("EMA Human Drugs"),
        )
        .explode("drugFromSource")
        .explode("diseaseFromSource")
        # After extracting diseases, some rows may have null values (25 currently)
        .filter(pl.col("drugFromSource").is_not_null() & pl.col("diseaseFromSource").is_not_null())
    )
