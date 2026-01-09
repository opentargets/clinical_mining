import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from enum import Enum


def validate_schema(df: pl.DataFrame, model: type[BaseModel]) -> pl.DataFrame:
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
    extra_fields = list(set(df.columns) - set(mandatory_fields))
    return df.select(mandatory_fields + extra_fields)


class DrugIndicationSource(str, Enum):
    """The data source of the evidence."""

    AACT = "AACT"
    USAN = "USAN"
    EMA = "EMA"
    ATC = "ATC"
    INN = "INN"
    DailyMed = "DailyMed"
    FDA = "FDA"
    EMA_Human_Drugs = "EMA Human Drugs"
    TTD = "TTD"


class MaxClinicalStatusCategory(str, Enum):
    """Standardized clinical development status categories, ranked by development stage."""
    
    APPROVED = "APPROVED"
    POST_APPROVAL_WITHDRAWN = "POST_APPROVAL_WITHDRAWN"
    REGULATORY_REVIEW = "REGULATORY_REVIEW"
    PHASE_4 = "PHASE_4"
    PHASE_3 = "PHASE_3"
    PHASE_2 = "PHASE_2"
    PHASE_1 = "PHASE_1"
    PRECLINICAL = "PRECLINICAL"
    NO_DEVELOPMENT_REPORTED = "NO_DEVELOPMENT_REPORTED"

class MappingStatus(str, Enum):
    """The mapping status of the drug/indication relationship."""
    
    FULLY_MAPPED = "FULLY_MAPPED"
    DRUG_MAPPED = "DRUG_MAPPED"
    DISEASE_MAPPED = "DISEASE_MAPPED"
    UNMAPPED = "UNMAPPED"

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
    source: DrugIndicationSource = Field(
        ..., description="The data source of the evidence."
    )
    drug_name: str | None = Field(
        default=None, description="The name of the drug.")
    disease_name: str | None = Field(
        default=None, description="The name of the disease.")
    drug_id: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    disease_id: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )


class DrugIndication(BaseModel):
    """Aggregated drug-indication relationship with multiple supporting sources."""

    model_config = ConfigDict(extra="allow")

    # Primary identifiers (derived from IDs or most frequent)
    id: str = Field(
        description="Hashed identifier based on canonical drug and disease names"
    )
    primary_drug_name: str = Field(
        description="Preferred drug name from ChEMBL or most frequent label"
    )
    primary_disease_name: str = Field(
        description="Preferred disease name from EFO or most frequent label"
    )

    drug_id: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    disease_id: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )

    sources: list[ClinicalStudy] = Field(
        ...,
        description="List of studies and their metadata that supports the association.",
    )
    max_clinical_status: MaxClinicalStatusCategory | None = Field(
        default=None,
        description="The maximum clinical development status (MCDS) of the drug/indication relationship.",
    )
    mapping_status: MappingStatus = Field(
        description="The mapping status of the drug/indication relationship.",
    )


