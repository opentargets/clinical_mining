import polars as pl

from ontoma.ner.disease import extract_disease_entities

from clinical_mining.utils.polars_helpers import convert_polars_to_spark
from clinical_mining.utils.spark_helpers import spark_session
from clinical_mining.schemas import ClinicalReportType
from clinical_mining.dataset import ClinicalReport
from loguru import logger


def extract_clinical_report(
    indications_path: str,
    spark: spark_session,
) -> ClinicalReport:
    """Extract clinical reports from the EMA list of human drugs."""
    raw = pl.read_excel(
        indications_path,
        sheet_name="Medicine",
    )
    raw.columns = raw.iter_rows().__next__()  # Assign columns names from first row

    human_indications = raw.slice(1).filter(  # drop header
        pl.col("Category") == "Human"
    )

    # Drop columns with all nulls to convert to spark
    non_empty_cols = [
        series.name
        for series in human_indications.iter_columns()
        if series.null_count() < human_indications.height
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

    reports = (
        ner_extracted_indication.select(
            id=pl.col("EMA product number").str.to_lowercase(),
            phaseFromSource=pl.col("Medicine status").str.to_lowercase(),
            type=pl.lit(ClinicalReportType.REGULATORY),
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
            source=pl.lit("EMA Human Drugs"),
            url=pl.col("Medicine URL"),
            hasExpertReview=pl.lit(False),
            # TODO: Marketing date
        )
        .explode("drugFromSource")
        .explode("diseaseFromSource")
        # After extracting diseases, some rows may have null values (25 currently)
        .filter(
            pl.col("drugFromSource").is_not_null()
            & pl.col("diseaseFromSource").is_not_null()
        )
        .unique()
    )

    mapped_reports = (
        # TODO: call mapping function
        reports.with_columns(
            disease=pl.struct(
                pl.col("diseaseFromSource"), pl.lit("CHEMBL_TO_DO").alias("diseaseId")
            ),
            drug=pl.struct(
                pl.col("drugFromSource"), pl.lit("EFO_TO_DO").alias("drugId")
            ),
        )
        .drop(["diseaseFromSource", "drugFromSource"])
        .unique()
    )

    return ClinicalReport(
        df=(
            mapped_reports.group_by(
                [c for c in mapped_reports.columns if c not in ["disease", "drug"]]
            ).agg(
                pl.col("disease").unique().alias("diseases"),
                pl.col("drug").unique().alias("drugs"),
            )
        )
    )
