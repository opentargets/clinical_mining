import polars as pl

from clinical_mining.dataset import DrugIndicationEvidenceDataset

def extract_ttd_indications(
    indications_path: str,
) -> DrugIndicationEvidenceDataset:
    """Extract drug/indication relationships from TTD Indications dataset."""
    # Read the file into a list of lines
    with open(indications_path, "r", encoding="utf-8") as file:
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
            clinical_status = parts[3]
            # Append a new row to the data list
            data.append(
                [
                    current_drug["TTDDRUID"],
                    current_drug["DRUGNAME"],
                    indication,
                    icd,
                    clinical_status,
                ]
            )

    # Create a DataFrame
    return DrugIndicationEvidenceDataset(
        df=(
            pl.DataFrame(
                data,
                schema=[
                    "ttd_id",
                    "drug_name",
                    "disease_name",
                    "icd11_id",
                    "clinical_status",
                ],
                orient="row"
            )
            # remove first line that includes column names
            .slice(1)
            .select(
                drug_name=pl.col("drug_name").str.to_lowercase(),
                disease_name=pl.col("disease_name").str.to_lowercase(),
                phase=pl.col("clinical_status").str.to_lowercase(),
                studyId=pl.concat_str(
                    [pl.col("ttd_id"), pl.lit("/"), pl.col("disease_name")]
                ).str.to_lowercase(),
                source=pl.lit("TTD"),
            )
        ).unique()
    )