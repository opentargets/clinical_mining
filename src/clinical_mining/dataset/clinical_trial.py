import polars as pl

from clinical_mining.schemas import validate_schema, ClinicalRecordSchema, snake_to_camel


class ClinicalRecord:
    """A dataset for clinical studies (e.g. clinical trial, an USAN reference), wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        # Harmonise column names from snake to camel case
        df = df.rename(
            {col: snake_to_camel(col) for col in df.columns}
        )
        self.df = validate_schema(df, ClinicalRecordSchema)
