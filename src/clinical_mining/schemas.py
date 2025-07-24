from pyspark.sql import DataFrame
from pydantic import BaseModel, ConfigDict, Field

from enum import Enum


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


class DrugIndicationSource(str, Enum):
    """The data source of the evidence."""

    AACT = "AACT"
    USAN = "USAN"
    EMA = "EMA"
    ATC = "ATC"
    INN = "INN"
    DailyMed = "DailyMed"
    FDA = "FDA"


class ApprovalEvidence(BaseModel):
    """Approval status from different sources."""

    class ApprovalSource(str, Enum):
        FDA = "FDA"
        EMA = "EMA"
        DailyMed = "DailyMed"

    source: ApprovalSource
    date: str | None = Field(default=None, description="The approval date.")


class ClinicalStudy(BaseModel):
    """Represents a clinical trial and its metadata."""

    model_config = ConfigDict(extra="allow")

    studyId: str = Field(..., description="The study identifier, e.g. NCT04012606.")
    url: str = Field(
        default=None, description="The URL of the study, e.g. in Dailymed."
    )
    source: DrugIndicationSource | None = Field(
        default=None, description="The data source of the study."
    )


class DrugIndicationEvidence(BaseModel):
    """Represents a single piece of evidence linking a drug to an indication from a source."""

    model_config = ConfigDict(extra="allow")

    studyId: str = Field(
        ...,
        description="The study identifier that supports the drug/indication relationship.",
    )
    drug_name: str = Field(..., description="The name of the drug.")
    disease_name: str = Field(..., description="The name of the disease.")
    source: DrugIndicationSource = Field(
        ..., description="The data source of the evidence."
    )
    drug_id: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    disease_id: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )
    approval: list[ApprovalEvidence] | None = Field(
        default=None,
        description="The approval status of the drug/indication relationship.",
    )


class DrugIndication(BaseModel):
    """Represents a single piece of evidence linking a drug to an indication from a source."""

    model_config = ConfigDict(extra="allow")

    drug_id: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    disease_id: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )
    drug_name: str = Field(..., description="The name of the drug.")
    disease_name: str = Field(..., description="The name of the disease.")
    sources: list[ClinicalStudy] = Field(
        ...,
        description="List of studies and their metadata that supports the association.",
    )
    approval: list[ApprovalEvidence] | None = Field(
        default=None,
        description="The approval status of the drug/indication relationship.",
    )
