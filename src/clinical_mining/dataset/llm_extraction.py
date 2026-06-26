import polars as pl

from clinical_mining.schemas import ClinicalReportExtractionSchema, validate_schema


class ClinicalReportExtraction:
    """A validated collection of LLM-extracted clinical trial data.

    The wrapped DataFrame conforms to the ``ClinicalReportExtractionSchema``.
    """

    def __init__(self, df: pl.DataFrame):
        if not df.is_empty():
            df = validate_schema(df, ClinicalReportExtractionSchema)
        self.df = df
