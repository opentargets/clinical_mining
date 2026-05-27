from typing import Literal

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

    WITHDRAWAL = "WITHDRAWAL"
    APPROVAL = "APPROVAL"
    PHASE_4 = "PHASE_4"
    PREAPPROVAL = "PREAPPROVAL"
    PHASE_3 = "PHASE_3"
    PHASE_2_3 = "PHASE_2_3"
    PHASE_2 = "PHASE_2"
    PHASE_1_2 = "PHASE_1_2"
    PHASE_1 = "PHASE_1"
    EARLY_PHASE_1 = "EARLY_PHASE_1"
    IND = "IND"
    PRECLINICAL = "PRECLINICAL"
    UNKNOWN = "UNKNOWN"


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
    drugFromSource: str | None = Field(
        default=None, description="The drug label used at the source."
    )
    drugId: str | None = Field(default=None, description="The assigned drug ID.")


class AssociatedDisease(BaseModel):
    diseaseFromSource: str | None = Field(
        description="The disease label used at the source."
    )
    diseaseId: str | None = Field(description="The assigned disease ID.")


class ClinicalReportSchema(BaseModel):
    """Represents a clinical record and its metadata."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        ..., description="The identifier for the clinical reference, e.g. NCT04012606."
    )
    clinicalStage: ClinicalStageCategory = Field(
        description="The clinical development status of the clinical reference after harmonisation .",
    )
    phaseFromSource: str | None = Field(
        default=None, description="The phase of the report at the source."
    )
    type: ClinicalReportType = Field(description="The type of the report.")
    year: int | None = Field(default=None, description="The year of the report.")
    countries: list[str] | None = Field(
        default=None, description="The countries where the report was conducted."
    )
    url: str | None = Field(
        default=None, description="The URL of the report, e.g. in Dailymed."
    )
    source: ClinicalSource = Field(description="The data source of the report.")
    diseases: list[AssociatedDisease] | None = Field(
        default=None, description="The diseases associated with the report."
    )
    drugs: list[AssociatedDrug] = Field(
        description="The drugs associated with the study."
    )
    sideEffects: list[AssociatedDisease] | None = Field(
        default=None, description="The side effects associated with the report."
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


class ExtractedDrug(BaseModel):
    """A drug or compound in a clinical trial, with its role and textual evidence."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)

    drug: str = Field(
        ...,
        description=(
            "Generic or international nonproprietary name of the drug or compound. "
            "Exclude placebos, vehicles, excipients (e.g. sesame oil, saline, DMSO, "
            "water for injection), and formulation components. "
            "Exclude special characters such as trademark or registered symbols. "
            "Exclude concentrations, volumes, or dosage details from this field. "
            "Exclude routes of administration (e.g. 'IV', 'oral', 'inhaled') and "
            "dosage forms (e.g. 'tablet', 'capsule', 'injection') — those go in "
            "route and formulation respectively."
        ),
    )
    route: str | None = Field(
        default=None,
        description=(
            "Route of administration when explicitly stated in the trial text. "
            "Examples: 'oral', 'IV' (intravenous), 'subcutaneous', 'intramuscular', "
            "'inhaled', 'topical', 'intrathecal', 'intranasal', 'transdermal'. "
            "Example: 'inhaled budesonide' → drug='budesonide', route='inhaled'. "
            "Omit if no route is explicitly stated."
        ),
    )
    formulation: str | None = Field(
        default=None,
        description=(
            "Physical dosage form or formulation when explicitly stated in the trial text. "
            "Examples: 'tablet', 'capsule', 'solution', 'suspension', 'powder', "
            "'cream', 'injection', 'infusion', 'patch', 'inhaler'. "
            "Example: 'oral metformin tablet' → drug='metformin', route='oral', formulation='tablet'. "
            "Omit if not explicitly stated."
        ),
    )
    synonyms: list[str] | None = Field(
        default=None,
        description=(
            "Other names explicitly used in the trial text to refer to the same molecule, "
            "such as brand names, abbreviations, or alternative spellings. "
            "Only include names explicitly mentioned in the input — do not infer or look up synonyms. "
            "Omit if none are present."
        ),
    )
    dosages: list[str] | None = Field(
        default=None,
        description=(
            "Dosage regimens explicitly stated in the trial text for this drug, "
            "each as a free-text string (e.g. '100 mg once daily', '2.5 mg/kg twice daily'). "
            "Use a separate list entry for each distinct regimen if multiple are mentioned. "
            "Only include dosages explicitly mentioned in the input. Omit if unspecified."
        ),
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "An exact verbatim span copied from the input text that directly supports the "
            "inclusion of this drug in this category. Do not paraphrase or summarise — "
            "copy the text exactly as it appears in the input."
        ),
    )


class ExtractedDisease(BaseModel):
    """A disease or condition extracted from a clinical trial."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)

    name: str = Field(
        ...,
        description=(
            "The CORE disease or condition name, stripped of all modifier descriptors. "
            "Severity, stage, onset, and etiology each go in their own dedicated field — they must "
            "NOT appear in name. "
            "Example: 'severe chronic oxaliplatin-induced peripheral neurotoxicity' → "
            "name='peripheral neurotoxicity' (with severity='severe', onset='chronic', "
            "etiology='oxaliplatin-induced'). "
            "Use the full disease name rather than acronyms when both appear. "
            "NEVER populate this field with 'healthy volunteers', 'healthy subjects', "
            "'healthy individuals', 'placebo', or any descriptor indicating healthy participants."
        ),
    )
    severity: str | None = Field(
        default=None,
        description=(
            "Explicit severity modifier stated in the trial text "
            "(e.g. 'mild', 'moderate', 'severe'). Omit if not stated."
        ),
    )
    stage: str | None = Field(
        default=None,
        description=(
            "Explicit disease stage or treatment history stated in the trial text "
            "(e.g. 'stage III', 'stage IV', 'relapsed', 'refractory', 'early-stage'). "
            "Omit if not stated."
        ),
    )
    onset: str | None = Field(
        default=None,
        description=(
            "Explicit onset or chronicity modifier stated in the trial text "
            "(e.g. 'acute', 'chronic', 'early-onset', 'late-onset'). Omit if not stated."
        ),
    )
    etiology: str | None = Field(
        default=None,
        description=(
            "Explicit cause or origin of the disease when distinct from the disease name itself. "
            "Common patterns: drug-induced (e.g. 'oxaliplatin-induced', 'chemotherapy-induced'), "
            "radiation-induced, post-surgical, post-infectious, virally-induced. "
            "Example: 'oxaliplatin-induced neurotoxicity' → name='neurotoxicity', "
            "etiology='oxaliplatin-induced'. "
            "Omit when no explicit cause is stated, or when the cause is inseparable from the "
            "disease name (e.g. 'lung cancer' — no etiology)."
        ),
    )
    evidence_quote: str | None = Field(
        default=None,
        description=(
            "An exact verbatim span copied from the input text that directly supports the "
            "identification of this disease. Do not paraphrase — copy the text exactly. "
            "Required for primary_indications. Best-effort for background_conditions, where "
            "the eligibility text may not contain a standalone quote."
        ),
    )


class ClinicalReportExtraction(BaseModel):
    """LLM-extracted structured information from a clinical trial report."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)

    id: str = Field(
        ...,
        description="The identifier for the clinical reference, e.g. NCT04012606.",
    )
    drug_intent: Literal[
        "therapeutic", "diagnostic", "prevention", "supportive_care", "other"
    ] = Field(
        ...,
        description=(
            "What the investigated_drugs are intended to do. This determines how to interpret "
            "primary_indications:\n"
            "- 'therapeutic': drugs are evaluated as TREATMENT for the primary_indications.\n"
            "- 'diagnostic': drugs are imaging probes / radiotracers / contrast agents / biomarker "
            "  assays used to DETECT, LOCALIZE, or DIAGNOSE the primary_indications.\n"
            "- 'prevention': drugs are evaluated to PREVENT the primary_indications "
            "  (which are events or outcomes, not the patient's chronic background condition).\n"
            "- 'supportive_care': drugs RELIEVE symptoms or side effects of the primary_indications.\n"
            "- 'other': none of the above (e.g. basic science, device feasibility, healthy-volunteer PK).\n"
            "This is DISTINCT from the 'Primary Purpose' field shown in the input — that field is a "
            "hint from ClinicalTrials.gov but is sometimes mislabelled. Classify based on what the "
            "drugs actually do. Example: an antifungal trial labelled SUPPORTIVE_CARE in CT.gov is "
            "still 'therapeutic' if the drugs are tested as antifungal therapy."
        ),
    )
    drug_intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence in the drug_intent classification, between 0.0 and 1.0. "
            "Use 0.95+ when the trial's purpose is unambiguous (e.g. clear treatment of a single "
            "disease). Use 0.7-0.9 when there is some ambiguity (e.g. CT.gov label conflicts with "
            "what the drugs actually do, or the trial spans multiple intents). Use below 0.7 when "
            "the description is sparse or the role of the drug is genuinely unclear. "
            "Be honest — low confidence is a useful signal for downstream review."
        ),
    )
    primary_indications: list[ExtractedDisease] = Field(
        ...,
        description=(
            "Diseases or conditions the trial primarily investigates. The relationship to "
            "investigated_drugs depends on drug_intent: "
            "treated (therapeutic), detected (diagnostic), prevented (prevention), or "
            "alleviated (supportive_care). "
            "List each distinct condition separately when a trial studies multiple in parallel "
            "(e.g. 'non-Hodgkin lymphoma, ALL, and CLL' → three separate entries; "
            "'fever and neutropenia' → two entries). "
            "Do NOT collapse multiple specific diseases into a parent category. "
            "COUPLING RULE: if a single entry's evidence_quote contains more than one specific "
            "disease name, that is wrong — split into multiple entries, each with a quote covering "
            "only its own disease. "
            "For prevention trials, this is the event being prevented (e.g. 'cardiovascular events'); "
            "the patient's underlying chronic disease becomes a background_condition. "
            "Return an empty list only when no indication can be identified at all."
        ),
    )
    background_conditions: list[ExtractedDisease] | None = Field(
        default=None,
        description=(
            "Diseases or conditions required for participant eligibility but NOT the primary "
            "therapeutic target. For example, if a trial studies allergic rhinitis in patients "
            "with asthma, 'asthma' is a background condition. Omit if none are present."
        ),
    )
    investigated_drugs: list[ExtractedDrug] = Field(
        ...,
        description=(
            "Drugs or compounds being evaluated by the trial. The role they play is given by "
            "study_kind: therapeutic agent, diagnostic/imaging probe, preventive agent, or "
            "supportive-care agent. "
            "Do NOT include: placebos, vehicles, excipients (e.g. sesame oil, saline, DMSO, "
            "water for injection), formulation components, active comparators (which go in "
            "comparator_drugs), or drugs given only for symptom management in a trial whose "
            "primary purpose is something else (which go in supportive_drugs)."
        ),
    )
    comparator_drugs: list[ExtractedDrug] | None = Field(
        default=None,
        description=(
            "Already-approved drugs used as an active comparator (standard of care) against which "
            "the investigated_drugs are benchmarked. Omit if no active comparator is present."
        ),
    )
    supportive_drugs: list[ExtractedDrug] | None = Field(
        default=None,
        description=(
            "Drugs given for symptomatic relief or supportive care that are NOT intended to treat "
            "the primary_indication (e.g. morphine for breakthrough pain in an oncology trial, "
            "antiemetics, antipyretics). Omit if none are present."
        ),
    )
    conclusion: str | None = Field(
        default=None,
        description=(
            "A single sentence describing the outcome or result of the clinical trial "
            "(e.g. whether the intervention was effective or safe), if explicitly stated "
            "in the trial data. Do not describe the study design, purpose, or objectives."
        ),
    )
