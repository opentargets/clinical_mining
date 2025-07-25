import polars as pl

from clinical_mining.schemas import validate_schema, ClinicalStudy


class ClinicalStudyDataset:
    """A dataset for clinical studies (e.g. clinical trial, an USAN reference), wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, ClinicalStudy)
