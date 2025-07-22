from pyspark.sql import DataFrame

from clinical_mining.schemas import validate_schema, ClinicalTrial

class ClinicalTrialDataset:
    """A dataset for clinical trials, wrapping a Spark DataFrame."""

    def __init__(self, df: DataFrame):
        """Initializes the dataset, validating and aligning the DataFrame."""
        self.df = validate_schema(df, ClinicalTrial)
