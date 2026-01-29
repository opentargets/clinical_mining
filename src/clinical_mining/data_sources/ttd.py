import polars as pl

from clinical_mining.dataset.clinical_report import ClinicalReport
from clinical_mining.schemas import ClinicalReportType


def read_input_file(input_file_path: str) -> pl.DataFrame:
    """Read TTD Indications dataset into a Polars DataFrame."""
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Initialize variables
    data = []
    current_drug = {"TTDDRUID": None, "DRUGNAME": None}

    # Parse the lines
    for line in lines:
        line = line.strip()
        if line.startswith("TTDDRUID"):
            current_drug["TTDDRUID"] = line.split("\t")[1]
        elif line.startswith("DRUGNAME"):
            current_drug["DRUGNAME"] = line.split("\t")[1]
        elif line.startswith("INDICATI"):
            parts = line.split("\t")
            indication = parts[1]
            icd = parts[2].replace("ICD-11: ", "").strip()
            clinical_stage = parts[3]
            # Append a new row to the data list
            data.append(
                [
                    current_drug["TTDDRUID"],
                    current_drug["DRUGNAME"],
                    indication,
                    icd,
                    clinical_stage,
                ]
            )

    return pl.DataFrame(
        data,
        schema=[
            "ttd_id",
            "drugFromSource",
            "diseaseFromSource",
            "icd11_id",
            "clinical_stage",
        ],
        orient="row",
    ).slice(1)  # remove first line that includes column names


def extract_clinical_report(
    indications_path: str,
) -> ClinicalReport:
    """Extract clinical reports from TTD drug/disease dataset."""

    indications = read_input_file(indications_path)

    reports = indications.select(
        id=pl.concat_str(
            [pl.col("ttd_id"), pl.lit("/"), pl.col("diseaseFromSource")]
        ).str.to_lowercase(),
        type=pl.lit(ClinicalReportType.CURATED_RESOURCE),
        url=pl.concat_str(
            [pl.lit("https://ttd.idrblab.cn/data/drug/details/"), pl.col("ttd_id")]
        ),
        hasExpertReview=pl.lit(True),
        phaseFromSource=pl.col("clinical_stage").str.to_lowercase(),
        drugFromSource=pl.col("drugFromSource").str.to_lowercase(),
        diseaseFromSource=pl.col("diseaseFromSource").str.to_lowercase(),
        source=pl.lit("TTD"),
    ).unique()

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
