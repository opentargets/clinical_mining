from enum import Enum
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


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


class ClinicalReportExtractionSchema(BaseModel):
    """LLM-extracted structured information from a clinical trial report."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)

    id: str = Field(
        ...,
        description="The identifier for the clinical reference, e.g. NCT04012606.",
    )
    drug_intent: Literal[
        "therapeutic", "diagnostic", "prevention", "supportive_care", "other"
    ] | None = Field(
        default=None,
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
    drug_intent_confidence: float | None = Field(
        default=None,
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

 
OutcomeCategory = Literal[
    "efficacy",
    "lack_of_efficacy",
    "effectiveness",
    "ineffectiveness",
    "safety_failure",
    "safety_tolerability",
    "mixed",
    "inconclusive",
    "no_results",
]
_OUTCOME_DESCRIPTION = (
    "The outcome category for this therapy/condition key. Determine it by the "
    "CONCLUSION CLASSIFICATION decision procedure in the system prompt (the ordered "
    "fact-tree) — classify strictly from the governing endpoint's result quoted in "
    "evidence_quote. This is NOT a flat menu to pick from and the verdict is NOT re-derived "
    "here; the lines below only disambiguate the enum tokens.\n"
    "- 'efficacy': governing endpoint MET vs an INACTIVE reference (placebo / none / "
    "historical / strategy-intensity-target).\n"
    "- 'lack_of_efficacy': governing endpoint FAILED vs an INACTIVE reference.\n"
    "- 'effectiveness': governing endpoint MET vs a named, distinct ACTIVE drug comparator "
    "(superior, or non-inferiority margin met).\n"
    "- 'ineffectiveness': NOT superior to a named ACTIVE drug comparator (inferior, "
    "equivalent, no significant difference, or non-inferiority fail).\n"
    "- 'safety_failure': trial/arm actually HALTED for toxicity/AEs/safety. Overrides every "
    "other category, even a met primary — but only on an actual halt.\n"
    "- 'safety_tolerability': only a safety / tolerability / MTD / RP2D / PK-PD readout and "
    "efficacy was never assessed. Polarity-neutral; no comparator determination applies.\n"
    "- 'mixed': a genuine category conflict for this key — two co-primary endpoints in "
    "OPPOSITE directions, or abstracts disagreeing on category (after the MULTIPLE RESULTS "
    "not-a-conflict exceptions).\n"
    "- 'inconclusive': efficacy WAS assessed but is genuinely unresolvable; ALSO the single "
    "trial-level row (empty therapies/conditions) when a provided trial description's "
    "abstracts are all irrelevant (evidence_quote from the description, source_pmid null, "
    "primary_endpoint null).\n"
    "- 'no_results': no readout of any kind (protocol / ongoing / terminated pre-readout). "
    "ANY data readout, even safety-only, is 'safety_tolerability' instead.\n"
    "Comparator rules, override precedence, and all tie-breaks live in the system prompt; do "
    "not duplicate or second-guess them here."
)
_CONFIDENCE_DESCRIPTION = (
    "Confidence in the conclusion, between 0.0 and 1.0:\n"
    "- 0.95+: outcome explicitly stated (e.g. 'met the primary endpoint', 'terminated due to "
    "toxicity', 'the maximum tolerated dose was X').\n"
    "- 0.7-0.9: outcome strongly implied but not explicitly stated, or the comparator type had "
    "to be inferred.\n"
    "- Below 0.7: ambiguous or conflicting signals. Be honest — low values are a useful signal "
    "for human review, so do not inflate them.\n"
    "For the single trial-level 'no relevant abstracts' row, this instead reflects confidence "
    "that the abstracts are indeed unrelated to the trial, not confidence in any drug verdict."
)


class Therapy(BaseModel):
    """A single therapy (drug/intervention) referenced by an outcome."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)
    name: str = Field(
        ...,
        description=(
            "Core INN name of the drug only — no route, formulation, or dose (e.g. "
            "'metformin', not 'oral metformin tablet'). A non-drug component of the "
            "regimen (e.g. radiotherapy, a device) may appear here, named plainly, "
            "when it is evaluated as part of the regimen alongside a drug — but it "
            "can never be the SOLE therapy, since every outcome's therapy set must "
            "contain at least one real, named drug. Never a placebo, vehicle, or "
            "excipient — those are comparators, not therapies."
        ),
    )
    synonyms: list[str] | None = Field(
        default=None,
        description=(
            "Other names explicitly used in the trial text to refer to the same therapy, "
            "such as brand names, abbreviations, code names, or alternative spellings. "
            "Only include names explicitly mentioned in the input — do not infer or look up synonyms. "
            "Omit if none are present."
        ),
    )


class Condition(BaseModel):
    """A single condition referenced by an outcome."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)
    name: str = Field(
        ...,
        description=(
            "Most specific disease entity explicitly stated in the abstracts; prefer "
            "the specific term given in a parenthetical. Keep a modifier ONLY when it "
            "names a distinct disease entity (e.g. 'triple-negative breast cancer'; "
            "split 'tonsil and base-of-tongue cancer' into two separate conditions). "
            "Drop etiology/comorbidity/severity/symptom modifiers and stage (e.g. "
            "'breast cancer patients with loss of appetite' -> 'breast cancer'). Use "
            "ONE name per disease across the whole trial (re-check for hidden "
            "duplicates after renaming). E.g. 'non-Hodgkin lymphoma', not 'lymphoma'. "
            "A subgroup-level name earns its own condition only when it is a distinct "
            "disease entity (molecular subtype, distinct anatomical site) reported "
            "with per-subgroup results; use the umbrella term for a single pooled "
            "result across a mixed population. Clinical/demographic/treatment-pattern "
            "subgroups are NOT conditions — they belong in the outcome's 'population'."
        ),
    )
    synonyms: list[str] | None = Field(
        default=None,
        description=(
            "Other names explicitly used in the trial text to refer to the same condition, "
            "such as abbreviations or alternative spellings. "
            "Only include names explicitly mentioned in the input — do not infer or look up synonyms. "
            "Omit if none are present."
        ),
    )


class KeyReconciliation(BaseModel):
    """Forced Pass C + Pass D scratch for ONE outcome key, produced BEFORE any
    'outcomes' row exists.

    This model is the JSON home for the reasoning the system prompt calls the
    forced two-step. Because the API runs under strict structured output with no
    room for free text, the collapse cannot happen on a scratchpad — it must
    happen HERE, in fields the model is required to fill before it writes
    'outcomes'. One entry per DISTINCT (therapies, conditions) key.

    It is working scratch, not a deliverable: downstream consumers should ignore
    it. Its only job is to force the model to inventory every endpoint and then
    name a single governing measure per key, so the verdict in 'outcomes' is
    resolved on one chosen endpoint rather than a blur of all of them.

    NOTE: this model deliberately does NOT carry a category/verdict field. The
    verdict is committed once, later, in TrialOutcome.conclusion — AFTER the
    governing endpoint's own result has been quoted in evidence_quote. Choosing a
    category here (before that quote exists) produced unstable, ungrounded
    verdicts, so classification is intentionally deferred out of the scratch."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)

    therapies: list[str] = Field(
        ...,
        description=(
            "Normalised, alphabetically-ordered core drug name(s) for this key "
            "(same normalisation as the outcome's therapies). Empty ONLY for the "
            "trial-level irrelevance row."
        ),
    )
    conditions: list[str] = Field(
        ...,
        description=(
            "Normalised, alphabetically-ordered disease name(s) for this key "
            "(same normalisation as the outcome's conditions). Empty ONLY for the "
            "trial-level irrelevance row."
        ),
    )
    endpoints_considered: list[str] = Field(
        ...,
        description=(
            "Pass D Step 1 — the INVENTORY. List EVERY endpoint/result reported "
            "for this key across the pooled abstracts: primary, secondary, "
            "exploratory, surrogate, and safety endpoints; every row of a GRADE / "
            "'summary of findings' table; every forest-plot estimate or pooled "
            "comparison. These are NOT rows — listing an endpoint here NEVER "
            "creates an outcome; the list exists only to be collapsed to one "
            "governing measure. A GRADE table with N endpoint-rows becomes N "
            "entries HERE, never N entries in 'outcomes'. Empty only for a "
            "no_results key or the trial-level irrelevance row."
        ),
    )
    governing_endpoint: str | None = Field(
        ...,
        description=(
            "Pass D Step 2 — the SINGLE measure chosen from endpoints_considered "
            "to govern this key's verdict, named plainly (e.g. 'all-cause "
            "mortality', 'objective response rate'). This choice is fixed: the "
            "outcome's conclusion is resolved from THIS endpoint's result alone, "
            "and every other endpoint in the inventory is discarded. MUST equal "
            "the resulting outcome's primary_endpoint. Null only for a no_results "
            "key or the trial-level irrelevance row."
        ),
    )
    governing_basis: Literal[
        "prespecified_primary",
        "authors_conclusion",
        "hardest_endpoint",
        "single_endpoint",
        "safety_only",
        "no_readout",
        "trial_irrelevant",
    ] = Field(
        ...,
        description=(
            "Why governing_endpoint governs, chosen in this priority order: "
            "'prespecified_primary' = a stated primary efficacy endpoint sets the "
            "verdict; else 'authors_conclusion' = the endpoint the authors "
            "foreground in their own bottom line — for a review or meta-analysis "
            "this is the 'Authors' conclusions' statement, NOT the summary-of-"
            "findings / GRADE table; else 'hardest_endpoint' = fallback to the "
            "hardest clinical endpoint reported (e.g. all-cause mortality over a "
            "surrogate or a lone adverse-event count) when the authors state no "
            "clear bottom line. Use 'single_endpoint' when only one endpoint was "
            "reported; 'safety_only' when no efficacy was assessed (verdict is "
            "safety_tolerability); 'no_readout' when no result of any kind exists "
            "yet (no_results); 'trial_irrelevant' for the trial-level irrelevance "
            "row."
        ),
    )


# for singular pass version
class TrialOutcome(BaseModel):
    """A single outcome: one or more therapies tested against one or more
    conditions, with one resolved verdict, consolidated across all abstracts
    for this trial.

    Field order is deliberate and load-bearing: primary_endpoint (which measure
    governs) is named first, then evidence_quote captures THAT measure's reported
    result, and only THEN is conclusion classified from that quoted result. Do not
    reorder — classifying before the governing result is quoted produces
    inconsistent verdicts.

    EXCEPTION: if none of this trial's abstracts actually relate to the
    trial (see the system prompt's Relevance Check), this may instead be a
    single trial-level row with therapies=[], conditions=[],
    population=None, primary_endpoint=None, conclusion='inconclusive',
    evidence_quote drawn from the trial description, and source_pmid=None.
    That row must be the only outcome emitted for the trial in that
    scenario."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)
    therapies: list[Therapy] = Field(
        ...,
        description=(
            "All therapies that are part of the regimen this outcome's verdict "
            "applies to — including a background/standard-of-care/add-on-to drug "
            "if the abstracts report the result as the COMBINED effect of that "
            "regimen (e.g. 'empagliflozin as an add-on to liraglutide' results "
            "in therapies=[empagliflozin, liraglutide], since the regimen "
            "evaluated together is the combination, not just the newly added "
            "agent). A combination regimen belongs in ONE outcome with multiple "
            "therapies. Therapies tested as SEPARATE arms with results reported "
            "separately (e.g. drug A vs. drug C, each independently, for the "
            "same condition) must be split into SEPARATE outcomes — do not "
            "combine unrelated arms into one outcome just because they share a "
            "condition. In a head-to-head trial BOTH active arms get their own "
            "outcome, including the active comparator. Every non-empty therapy "
            "set MUST contain at least one real, named drug; a non-drug "
            "component (radiotherapy, device) may sit alongside a drug but never "
            "substitutes for one. Never emit an outcome whose therapy is a "
            "placebo, vehicle, or excipient. Exclude only drugs that are not "
            "part of the evaluated regimen at all. May be an EMPTY list only in "
            "the single trial-level row for a trial whose abstracts are all "
            "irrelevant to it — see class docstring."
        ),
    )
    conditions: list[Condition] = Field(
        ...,
        description=(
            "All conditions this outcome's verdict applies to. Most outcomes "
            "have exactly one condition; use multiple only if the abstracts "
            "report a single shared verdict across more than one condition for "
            "the same therapy/regimen. In basket or multi-cohort trials, emit "
            "one outcome PER indication rather than one pooled outcome covering "
            "all of them. May be an EMPTY list only in the same trial-level "
            "irrelevance row described in the class docstring."
        ),
    )
    population: str | None = Field(
        default=None,
        description=(
            "The specific patient subgroup this outcome's verdict applies to, "
            "when the abstracts report a result restricted to a subgroup rather "
            "than the whole enrolled population — e.g. 'patients without "
            "delayed dosing', 'patients undertreated based on age', 'elderly "
            "(>=65) subset'. Copy the population descriptor verbatim from the "
            "abstract; do not invent, infer, or normalise one. Leave null (or "
            "omit) when the verdict applies to the full enrolled/randomised "
            "population for this therapy/condition, which is the common case. "
            "This captures clinical/demographic/treatment-pattern subgroups "
            "that are NOT distinct disease entities — a molecular subtype or "
            "distinct anatomical site is instead its own 'conditions' entry, "
            "not a population. This field is DESCRIPTIVE ONLY and is NOT part "
            "of the outcome key: (therapies, conditions) remains the sole "
            "identifier, so two outcomes still may not share an identical "
            "(therapies, conditions) combination even if their populations "
            "differ. When per-subgroup results differ for one such key (e.g. "
            "one subgroup benefits, another does not, both from the same "
            "abstract), they still collapse to a SINGLE row per Pass D — the "
            "verdict is resolved from the governing endpoint's own result and "
            "this field cannot represent both subgroups, so record whichever "
            "subgroup the surviving evidence_quote describes, leaving the row "
            "visibly subgroup-driven for human review."
        ),
    )
    primary_endpoint: str | None = Field(
        default=None,
        description=(
            "STEP 1 of this row — decide this BEFORE evidence_quote and "
            "conclusion. The measure that governs this outcome's verdict, and it "
            "MUST equal the governing_endpoint chosen for this key in "
            "key_reconciliation. For an efficacy-category outcome (efficacy / "
            "lack_of_efficacy / effectiveness / ineffectiveness) or an "
            "'inconclusive' outcome, this is the governing efficacy measure, e.g. "
            "'overall survival', 'objective response rate', 'change in HbA1c from "
            "baseline', 'progression-free survival'. For a 'safety_tolerability' "
            "outcome, where no efficacy was assessed, name the governing "
            "safety/dose-finding measure instead, e.g. 'maximum tolerated dose', "
            "'recommended phase 2 dose', 'dose-limiting toxicities', 'incidence of "
            "adverse events', 'pharmacokinetic parameters'. When a source reports "
            "many endpoints for this key, this is still ONE outcome resolved on "
            "the SINGLE governing measure — not one outcome per endpoint. If the "
            "key has two genuine co-primary endpoints pointing in opposite "
            "directions (conclusion 'mixed'), name BOTH, separated by '; '. Null "
            "ONLY when 'conclusion' is 'no_results' (no readout of any kind "
            "exists), or for the trial-level 'no relevant abstracts' row."
        ),
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "STEP 2 of this row — write this BEFORE conclusion. The exact verbatim "
            "span reporting the RESULT of the governing measure named in "
            "primary_endpoint (e.g. the sentence stating whether all-cause "
            "mortality was reduced), for this specific therapy/condition "
            "combination only. This quote is what conclusion is classified from, "
            "so it must be the governing endpoint's own result — NOT a quote about "
            "a secondary or exploratory endpoint, and not paraphrased or combined "
            "across abstracts. EXCEPTION: for the trial-level 'no relevant "
            "abstracts' row, this is instead a verbatim span from the trial "
            "description indicating what the trial was designed to test — only "
            "usable when a real trial description was actually provided (not 'Not "
            "provided.')."
        ),
    )
    source_pmid: str | None = Field(
        default=None,
        description=(
            "PubMed ID of the abstract the evidence_quote was taken from. "
            "Required for every normal outcome. Null ONLY for the trial-level "
            "'no relevant abstracts' row, where evidence_quote is drawn from "
            "the trial description rather than a specific abstract."
        ),
    )
    conclusion: OutcomeCategory = Field(
        ...,
        description=(
            "STEP 3 of this row — classify LAST, strictly from the governing "
            "endpoint's result quoted in evidence_quote above. Do not classify "
            "from the overall impression of the paper or from any other endpoint. "
            + _OUTCOME_DESCRIPTION
        ),
    )
    outcome_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=_CONFIDENCE_DESCRIPTION,
    )


class TrialExtraction(BaseModel):
    """All outcomes extracted for a single clinical trial, consolidated
    across all of its associated abstracts."""

    model_config = ConfigDict(validate_by_name=True, alias_generator=str.lower)
    id: str = Field(..., description="Clinical trial identifier (NCT ID).")
    key_reconciliation: list[KeyReconciliation] = Field(
        ...,
        description=(
            "FILL THIS FIRST, before writing 'outcomes'. One entry per DISTINCT "
            "(therapies, conditions) key you will emit. This is the forced "
            "collapse step: (1) fix the set of distinct keys here — this count is "
            "the number of outcome rows and nothing after may change it; (2) for "
            "each key, inventory every reported endpoint in endpoints_considered, "
            "then name the ONE governing_endpoint. THEN emit exactly one 'outcomes' "
            "entry per entry in this list, with matching (therapies, conditions) "
            "and primary_endpoint = governing_endpoint. len(outcomes) MUST equal "
            "the number of distinct keys here. A GRADE / summary-of-findings table "
            "with N endpoint-rows is ONE key with N endpoints_considered — never N "
            "outcomes. This field is working scratch and is ignored downstream; "
            "producing it correctly is what prevents one-row-per-endpoint "
            "duplication."
        ),
    )
    outcomes: list[TrialOutcome] = Field(
        ...,
        description=(
            "One entry per distinct therapy-combination/condition-set tested in "
            "this trial — exactly one per key in key_reconciliation. Two outcomes "
            "must never share an identical (therapies, conditions) combination — "
            "see deduplication rules in the system prompt. Return an empty list if "
            "abstracts relate to the trial but no drug can be identified in any of "
            "them. If, instead, NONE of the abstracts relate to this trial at all, "
            "return a single trial-level inconclusive row per the Relevance "
            "Check rules in the system prompt — do not return an empty list "
            "in that case."
        ),
    )