from pyspark.sql import DataFrame

from clinical_mining.schemas import validate_schema, ClinicalStudy


class ClinicalStudyDataset:
    """A dataset for clinical studies (e.g. clinical trial, an USAN reference), wrapping a Spark DataFrame."""

    def __init__(self, df: DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, ClinicalStudy)
