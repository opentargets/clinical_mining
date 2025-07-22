from __future__ import annotations

from pyspark.sql import DataFrame
from pydantic import BaseModel, ConfigDict, Field


def validate_schema(df: DataFrame, model: type[BaseModel]) -> DataFrame:
    """Validates that all mandatory schema fields are present. Resulting DataFrame is reordered to show core fields first."""
    mandatory_fields = [
        field_name
        for field_name, field in model.model_fields.items()
        if field.is_required()
    ]
    if not all(field in df.columns for field in mandatory_fields):
        raise ValueError(
            f"Missing mandatory fields: {set(mandatory_fields) - set(df.columns)}"
        )
    extra_fields = set(df.columns) - set(mandatory_fields)
    return df.select(mandatory_fields + list(extra_fields))


class DrugIndicationEvidence(BaseModel):
    """Represents a single piece of evidence linking a drug to an indication from a source."""

    model_config = ConfigDict(extra="allow")

    studyId: str = Field(..., description="The study identifier that supports the drug/indication relationship.")
    drug_name: str = Field(..., description="The name of the drug.")
    disease_name: str = Field(..., description="The name of the disease.")
    source: str = Field(..., description="The data source of the evidence.")
    drug_id: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    disease_id: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )

class ClinicalTrial(BaseModel):
    """Represents a clinical trial and its metadata."""

    model_config = ConfigDict(extra="allow")

    studyId: str = Field(..., description="The study identifier in ClinicalTrials.gov.")
    
    