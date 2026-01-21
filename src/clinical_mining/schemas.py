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


def snake_to_camel(snake_str: str) -> str:
    """Convert a snake_case string to camelCase.

    Examples:
        >>> snake_to_camel('clinical_phase')
        'clinicalPhase'
        >>> snake_to_camel('studyId')
        'studyId'
    """
    # Split by underscore
    components = snake_str.split("_")
    # Keep first component lowercase, capitalize the rest
    return components[0] + "".join(word.capitalize() for word in components[1:])


class ClinicalSource(str, Enum):
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
    PMDA = "PMDA"


class ClinicalStatusCategory(str, Enum):
    """Standardised clinical development status categories, ranked by development stage."""

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


class ClinicalTrialSchema(BaseModel):
    """Represents a clinical trial and its metadata."""

    model_config = ConfigDict(extra="allow")

    studyId: str = Field(..., description="The study identifier, e.g. NCT04012606.")
    url: str = Field(
        default=None, description="The URL of the study, e.g. in Dailymed."
    )
    source: ClinicalSource | None = Field(
        default=None, description="The data source of the study."
    )


class ClinicalEvidenceSchema(BaseModel):
    """Represents a single piece of evidence linking a drug to an indication from a source."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        description="Hashed identifier based on drug and disease names and studyId"
    )
    drugName: str = Field(
        description="Drug name (ChEMBL ID if mapping is available, otherwise label from clinical source)"
    )
    diseaseName: str = Field(
        description="Disease name (EFO ID if mapping is available, otherwise label from clinical source)"
    )
    studyId: str = Field(
        description="The study identifier that supports the drug/indication relationship.",
    )

    source: ClinicalSource = Field(..., description="The data source of the evidence.")
    drugFromSource: str | None = Field(
        default=None, description="The name of the drug."
    )
    diseaseFromSource: str | None = Field(
        default=None, description="The name of the disease."
    )
    drugId: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    diseaseId: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )
    clinicalStatus: ClinicalStatusCategory = Field(
        description="The clinical development status of the drug/indication relationship.",
    )


class ClinicalAssociationSchema(BaseModel):
    """Aggregated drug-indication relationship with multiple supporting sources."""

    model_config = ConfigDict(extra="allow")

    # Primary identifiers (derived from IDs)
    id: str = Field(description="Hashed identifier based on drug and disease names")
    drugName: str = Field(
        description="Drug name (ChEMBL ID if mapping is available, otherwise label from clinical source)"
    )
    diseaseName: str = Field(
        description="Disease name (EFO ID if mapping is available, otherwise label from clinical source)"
    )

    drugId: str | None = Field(
        default=None,
        description="The ChEMBL ID corresponding to the drug.",
    )
    diseaseId: str | None = Field(
        default=None, description="The EFO ID corresponding to the disease."
    )
    sources: list[ClinicalEvidenceSchema] = Field(
        ...,
        description="List of studies and their metadata that supports the association.",
    )
    maxClinicalStatus: ClinicalStatusCategory = Field(
        description="The maximum clinical development status (MCDS) of the drug/indication relationship.",
    )
    mappingStatus: MappingStatus = Field(
        description="The mapping status of the drug/indication relationship.",
    )
