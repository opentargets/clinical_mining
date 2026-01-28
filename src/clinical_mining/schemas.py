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


class ClinicalStageCategory(str, Enum):
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

class ClinicalReportType(str, Enum):
    """The type of the clinical record."""

    CLINICAL_TRIAL = "CLINICAL_TRIAL"
    DRUG_LABEL = "DRUG_LABEL"
    REGULATORY = "REGULATORY_AGENCY"
    CURATED_RESOURCE = "CURATED_RESOURCE"

class AssociatedDrug(BaseModel):

    drugFromSource: str = Field(..., description="The drug label used at the source.")
    drugId: str | None = Field(..., description="The assigned drug ID.")

class AssociatedDisease(BaseModel):

    diseaseFromSource: str = Field(..., description="The disease label used at the source.")
    diseaseId: str | None = Field(..., description="The assigned disease ID.")

class ClinicalReportSchema(BaseModel):
    """Represents a clinical record and its metadata."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="The identifier for the clinical reference, e.g. NCT04012606.")
    clinicalStage: ClinicalStageCategory = Field(
        description="The clinical development status of the clinical reference after harmonisation .",
    )
    phaseFromSource: str | None = Field(
        default=None, description="The phase of the report at the source."
    )
    type: ClinicalReportType = Field(
        default=None, description="The type of the report."
    )
    url: str | None = Field(
        default=None, description="The URL of the report, e.g. in Dailymed."
    )
    source: ClinicalSource = Field(
        default=None, description="The data source of the report."
    )
    diseases: list[AssociatedDisease] = Field(
        default=None, description="The diseases associated with the report."
    )
    drugs: list[AssociatedDrug] = Field(
        default=None, description="The drugs associated with the study."
    )
    hasExpertReview: bool = Field(
        default=False,
        description="Whether the report has been reviewed by an expert.",
    )

   # + optional trial metadata fields with the `trial` prefix. E.g. trialDescription

class ClinicalIndicationSchema(BaseModel):
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
    maxClinicalStage: ClinicalStageCategory = Field(
        description="The maximum clinical development status (MCDS) of the drug/indication relationship.",
    )
    mappingStatus: MappingStatus = Field(
        description="The mapping status of the drug/indication relationship.",
    )
    clinicalReportIds: list[str] = Field(
        ...,
        description="List of clinical report IDs that support the association.",
    )
    hasExpertReview: bool = Field(
        default=False,
        description="True if any of the supporting clinical reports has been reviewed by an expert.",
    )
