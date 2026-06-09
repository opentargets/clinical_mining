import json

import pytest
from pydantic import ValidationError

from clinical_mining.schemas import (
    ClinicalReportExtraction,
    ExtractedDrug,
    ExtractedDisease,
)
from clinical_mining.data_sources.aact.llm_extractor import (
    _parse_single_record,
    EXTRACTION_SCHEMA,
)


def test_investigated_drug_required_fields():
    drug = ExtractedDrug(drug="ibuprofen", evidence_quote="patients received ibuprofen")
    assert drug.drug == "ibuprofen"
    assert drug.evidence_quote == "patients received ibuprofen"


def test_investigated_drug_missing_required_raises():
    with pytest.raises(ValidationError):
        ExtractedDrug()


def test_investigated_drug_dosages_is_list():
    drug = ExtractedDrug(
        drug="metformin",
        evidence_quote="metformin 500 mg twice daily",
        dosages=["500 mg twice daily", "1000 mg once daily"],
    )
    assert isinstance(drug.dosages, list)
    assert len(drug.dosages) == 2


def test_extracted_drug_modifier_fields_default_to_none():
    drug = ExtractedDrug(drug="metformin", evidence_quote="metformin was administered")
    assert drug.route is None
    assert drug.formulation is None


def test_extracted_drug_route_captures_inhaled():
    """Inhaled budesonide → drug='budesonide', route='inhaled'."""
    drug = ExtractedDrug(
        drug="budesonide",
        route="inhaled",
        evidence_quote="inhaled budesonide 400 mcg twice daily",
    )
    assert drug.route == "inhaled"
    assert drug.formulation is None


def test_extracted_drug_route_and_formulation_independent():
    """'oral metformin tablet' → route='oral', formulation='tablet'."""
    drug = ExtractedDrug(
        drug="metformin",
        route="oral",
        formulation="tablet",
        evidence_quote="oral metformin tablet 500 mg",
    )
    assert drug.drug == "metformin"
    assert drug.route == "oral"
    assert drug.formulation == "tablet"


def test_extracted_drug_modifiers_round_trip_json():
    """route/formulation survive JSON round-trip."""
    import json

    drug = ExtractedDrug(
        drug="metformin",
        route="oral",
        formulation="tablet",
        evidence_quote="oral metformin tablet",
    )
    data = json.loads(drug.model_dump_json())
    assert data["route"] == "oral"
    assert data["formulation"] == "tablet"
    restored = ExtractedDrug.model_validate(data)
    assert restored.route == "oral"
    assert restored.formulation == "tablet"


def test_extracted_disease_required_fields():
    disease = ExtractedDisease(
        name="type 2 diabetes", evidence_quote="patients with type 2 diabetes"
    )
    assert disease.name == "type 2 diabetes"
    assert disease.evidence_quote == "patients with type 2 diabetes"


def test_extracted_disease_evidence_quote_optional():
    """background_conditions may omit evidence_quote when no standalone span exists."""
    disease = ExtractedDisease(name="colorectal cancer")
    assert disease.name == "colorectal cancer"
    assert disease.evidence_quote is None


def test_extracted_disease_modifiers_are_separate_fields():
    disease = ExtractedDisease(
        name="peripheral neurotoxicity",
        evidence_quote="severe chronic oxaliplatin-induced peripheral neurotoxicity",
        severity="severe",
        stage="stage III",
        onset="chronic",
        etiology="oxaliplatin-induced",
    )
    assert disease.severity == "severe"
    assert disease.stage == "stage III"
    assert disease.onset == "chronic"
    assert disease.etiology == "oxaliplatin-induced"


def test_extracted_disease_etiology_defaults_to_none():
    disease = ExtractedDisease(name="lung cancer", evidence_quote="lung cancer")
    assert disease.etiology is None


def test_clinical_report_extraction_full():
    extraction = ClinicalReportExtraction(
        id="NCT04012606",
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(
                name="type 2 diabetes",
                evidence_quote="patients with type 2 diabetes were enrolled",
            )
        ],
        investigated_drugs=[
            ExtractedDrug(
                drug="metformin", evidence_quote="metformin 500 mg was administered"
            ),
        ],
    )
    assert extraction.id == "NCT04012606"
    assert extraction.drug_intent == "therapeutic"
    assert extraction.primary_indications[0].name == "type 2 diabetes"
    assert len(extraction.investigated_drugs) == 1


def test_clinical_report_extraction_supports_multiple_primary_indications():
    """A trial studying NHL, ALL, and CLL in parallel should list all three."""
    extraction = ClinicalReportExtraction(
        id="NCT07166419",
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(
                name="non-Hodgkin lymphoma", evidence_quote="Non-Hodgkin Lymphoma"
            ),
            ExtractedDisease(
                name="acute lymphoblastic leukemia",
                evidence_quote="Acute Lymphoblastic Leukemia",
            ),
            ExtractedDisease(
                name="chronic lymphocytic leukemia",
                evidence_quote="Chronic Lymphocytic Leukemia",
            ),
        ],
        investigated_drugs=[
            ExtractedDrug(
                drug="TriCAR19.20.22 T cells", evidence_quote="TriCAR19.20.22 T cells"
            ),
        ],
    )
    assert len(extraction.primary_indications) == 3


def test_clinical_report_extraction_diagnostic_drug_intent():
    """Diagnostic/imaging trials use drug_intent='diagnostic' to flag the relationship as detection."""
    extraction = ClinicalReportExtraction(
        id="NCT07218224",
        drug_intent="diagnostic",
        drug_intent_confidence=0.9,
        primary_indications=[
            ExtractedDisease(
                name="parathyroid adenomas",
                evidence_quote="localization of parathyroid adenomas",
            )
        ],
        investigated_drugs=[
            ExtractedDrug(
                drug="18F-fluorocholine", evidence_quote="18F-fluorocholine (FCH)"
            ),
        ],
    )
    assert extraction.drug_intent == "diagnostic"
    assert extraction.primary_indications[0].name == "parathyroid adenomas"


def test_clinical_report_extraction_prevention_drug_intent():
    """Prevention trials: primary_indications is the prevented event, not the chronic condition."""
    extraction = ClinicalReportExtraction(
        id="NCT00000620",
        drug_intent="prevention",
        drug_intent_confidence=0.85,
        primary_indications=[
            ExtractedDisease(
                name="cardiovascular events",
                evidence_quote="prevent major cardiovascular events",
            )
        ],
        background_conditions=[
            ExtractedDisease(
                name="type 2 diabetes",
                evidence_quote="adults with type 2 diabetes",
            )
        ],
        investigated_drugs=[
            ExtractedDrug(drug="simvastatin", evidence_quote="simvastatin"),
        ],
    )
    assert extraction.drug_intent == "prevention"
    assert extraction.background_conditions[0].name == "type 2 diabetes"


def test_clinical_report_extraction_optional_fields_default_none():
    extraction = ClinicalReportExtraction(
        id="NCT00000001",
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(name="asthma", evidence_quote="asthma patients")
        ],
        investigated_drugs=[
            ExtractedDrug(drug="budesonide", evidence_quote="budesonide inhaler"),
        ],
    )
    assert extraction.background_conditions is None
    assert extraction.comparator_drugs is None
    assert extraction.supportive_drugs is None
    assert extraction.conclusion is None


def test_clinical_report_extraction_from_json():
    json_str = """{
        "id": "NCT00000001",
        "drug_intent": "therapeutic",
    "drug_intent_confidence": 0.95,
        "primary_indications": [{
            "name": "headache",
            "evidence_quote": "patients with chronic headache"
        }],
        "investigated_drugs": [
            {"drug": "aspirin", "evidence_quote": "aspirin 100mg was given"}
        ]
    }"""
    extraction = ClinicalReportExtraction.model_validate_json(json_str)
    assert extraction.id == "NCT00000001"
    assert extraction.drug_intent == "therapeutic"
    assert extraction.primary_indications[0].name == "headache"
    assert extraction.investigated_drugs[0].drug == "aspirin"


import polars as pl
import tempfile
import os
from pathlib import Path


def _make_sample_parquet(path: str, n: int = 20) -> pl.DataFrame:
    """Helper: write a minimal clinical report parquet for testing."""
    df = pl.DataFrame(
        {
            "id": [f"NCT{i:08d}" for i in range(n)],
            "type": ["CLINICAL_TRIAL"] * (n - 2) + ["DRUG_LABEL", "REGULATORY_AGENCY"],
            "clinicalStage": ["PHASE_2"] * n,
            "drugs": [[{"drugFromSource": "aspirin", "drugId": None}]] * n,
            "diseases": [[{"diseaseFromSource": "headache", "diseaseId": None}]] * n,
            "trial_official_title": [f"Study {i}" for i in range(n)],
            "trial_description": [f"Description {i}" for i in range(n)],
            "trial_phase": ["PHASE2"] * n,
            "trial_overall_status": ["COMPLETED"] * n,
            "trial_primary_purpose": ["TREATMENT"] * n,
            "trial_study_type": ["INTERVENTIONAL"] * n,
            "trial_number_of_arms": [2] * n,
            "trial_why_stopped": [None] * n,
            "trial_literature": [None] * n,
            "trial_start_date": [None] * n,
        }
    )
    df.write_parquet(path)
    return df


TRIAL_FIELDS = {
    "trialOfficialTitle": "Official Title",
    "trialDescription": "Description",
    "trialDetailedDescription": "Detailed Description",
    "trialPhase": "Phase",
    "trialOverallStatus": "Overall Status",
    "trialPrimaryPurpose": "Primary Purpose",
    "trialStudyType": "Study Type",
    "trialNumberOfArms": "Number of Arms",
    "trialWhyStopped": "Why Stopped",
    "trialStartDate": "Start Date",
}


def test_build_prompt_contains_id_and_trial_fields():
    from clinical_mining.data_sources.aact.llm_extractor import build_prompt

    row = {
        "id": "NCT04012606",
        "trialOfficialTitle": "A Phase 2 Study",
        "trialDescription": "Tests aspirin",
        "trialDetailedDescription": "Detailed protocol.",
        "trialPhase": "PHASE_2",
        "trialOverallStatus": "COMPLETED",
        "trialPrimaryPurpose": "TREATMENT",
        "trialStudyType": "INTERVENTIONAL",
        "trialNumberOfArms": 2,
        "trialWhyStopped": None,
        "trialLiterature": None,
        "trialStartDate": "2020-01-01",
    }
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert "NCT04012606" in prompt
    assert "A Phase 2 Study" in prompt
    assert "Detailed protocol." in prompt
    assert "PHASE_2" in prompt
    assert "null" in prompt  # None values rendered as null


def test_build_prompt_handles_missing_trial_fields():
    from clinical_mining.data_sources.aact.llm_extractor import build_prompt

    row = {"id": "NCT00000001"}  # no trial_* fields
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert "NCT00000001" in prompt


def test_build_prompt_with_publications():
    from clinical_mining.data_sources.aact.llm_extractor import build_prompt

    row = {
        "id": "NCT04012606",
        "trialOfficialTitle": "A Phase 2 Study",
        "trialDescription": "Tests aspirin",
        "trialDetailedDescription": None,
        "trialPhase": "PHASE_2",
        "trialOverallStatus": "COMPLETED",
        "trialPrimaryPurpose": "TREATMENT",
        "trialStudyType": "INTERVENTIONAL",
        "trialNumberOfArms": 2,
        "trialWhyStopped": None,
        "trialStartDate": "2020-01-01",
    }
    publications = [
        {"title": "Aspirin for pain", "abstractText": "Aspirin is effective."},
        {"title": "Second study", "abstractText": "More findings."},
    ]
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS, publications=publications)
    assert "Publications" in prompt
    assert "[1]" in prompt
    assert "Aspirin for pain" in prompt
    assert "Aspirin is effective." in prompt
    assert "[2]" in prompt
    assert "Second study" in prompt


def test_build_prompt_includes_interventions():
    from clinical_mining.data_sources.aact.llm_extractor import build_prompt

    row = {
        "id": "NCT00000001",
        "drugs": [
            {"drugFromSource": "aspirin", "drugId": None},
            {"drugFromSource": "ibuprofen", "drugId": None},
        ],
    }
    prompt = build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert "Interventions" in prompt
    assert "aspirin" in prompt
    assert "ibuprofen" in prompt


def test_build_prompt_omits_interventions_when_empty():
    from clinical_mining.data_sources.aact.llm_extractor import build_prompt

    row = {"id": "NCT00000001", "drugs": []}
    assert "Interventions" not in build_prompt(row, trial_fields=TRIAL_FIELDS)
    row2 = {"id": "NCT00000001"}
    assert "Interventions" not in build_prompt(row2, trial_fields=TRIAL_FIELDS)


def test_build_prompt_without_publications_omits_section():
    from clinical_mining.data_sources.aact.llm_extractor import build_prompt

    row = {"id": "NCT00000001"}
    assert "Publications" not in build_prompt(row, trial_fields=TRIAL_FIELDS)
    assert "Publications" not in build_prompt(
        row, trial_fields=TRIAL_FIELDS, publications=None
    )


def _make_extraction(
    nct_id: str,
    drug: str,
    condition: str,
    num_drugs: int = 1,
) -> ClinicalReportExtraction:
    return ClinicalReportExtraction(
        id=nct_id,
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(
                name=condition,
                evidence_quote=condition,
            )
        ],
        investigated_drugs=[
            ExtractedDrug(drug=drug, evidence_quote=drug) for _ in range(num_drugs)
        ],
    )


def _make_input_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "clinicalStage": [r.get("clinicalStage", "PHASE_2") for r in rows],
            "drugs": [[{"drugFromSource": r["drug"], "drugId": None}] for r in rows],
            "diseases": [
                [{"diseaseFromSource": r["disease"], "diseaseId": None}] for r in rows
            ],
        }
    )


def test_compute_drug_coverage_both_present():
    from scripts.validation_report import compute_drug_coverage

    extractions = [_make_extraction("NCT0001", "aspirin", "headache")]
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}]
    )
    result = compute_drug_coverage(extractions, input_df)
    assert result["input"] == 1
    assert result["output"] == 1
    assert result["total"] == 1
    assert result["input_pct"] == 100.0
    assert result["output_pct"] == 100.0


def test_compute_drug_coverage_missing_in_input():
    from scripts.validation_report import compute_drug_coverage

    extractions = [_make_extraction("NCT0001", "aspirin", "headache")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": None, "disease": "headache"}])
    result = compute_drug_coverage(extractions, input_df)
    assert result["input"] == 0
    assert result["output"] == 1


def test_compute_disease_coverage_both_present():
    from scripts.validation_report import compute_disease_coverage

    extractions = [_make_extraction("NCT0001", "aspirin", "headache")]
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}]
    )
    result = compute_disease_coverage(extractions, input_df)
    assert result["input"] == 1
    assert result["output"] == 1
    assert result["input_pct"] == 100.0


def test_compute_disease_coverage_missing_in_input():
    from scripts.validation_report import compute_disease_coverage

    extractions = [_make_extraction("NCT0001", "aspirin", "migraine")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": "aspirin", "disease": None}])
    result = compute_disease_coverage(extractions, input_df)
    assert result["input"] == 0
    assert result["output"] == 1


def test_compute_drugs_by_stage():
    from scripts.validation_report import compute_drugs_by_stage

    extractions = [
        _make_extraction("NCT0001", "aspirin", "headache", num_drugs=2),
        _make_extraction("NCT0002", "metformin", "diabetes", num_drugs=1),
    ]
    input_df = _make_input_df(
        [
            {
                "id": "NCT0001",
                "drug": "aspirin",
                "disease": "headache",
                "clinicalStage": "PHASE_2",
            },
            {
                "id": "NCT0002",
                "drug": "metformin",
                "disease": "diabetes",
                "clinicalStage": "PHASE_3",
            },
        ]
    )
    result = compute_drugs_by_stage(extractions, input_df)
    assert isinstance(result, pl.DataFrame)
    assert "clinicalStage" in result.columns
    assert "total_investigated_drugs" in result.columns
    phase2_row = result.filter(pl.col("clinicalStage") == "PHASE_2")
    assert phase2_row["total_investigated_drugs"][0] == 2


def test_compute_excipient_hits_detects_placebo():
    from scripts.validation_report import compute_excipient_hits

    extractions = [
        _make_extraction("NCT0001", "placebo", "headache"),
        _make_extraction("NCT0002", "aspirin", "headache"),
        _make_extraction("NCT0003", "sesame oil", "headache"),
    ]
    hits = compute_excipient_hits(extractions)
    hit_drugs = {h["drug"] for h in hits}
    assert "placebo" in hit_drugs
    assert "sesame oil" in hit_drugs
    assert "aspirin" not in hit_drugs


def test_compute_quote_grounding_skips_none_quotes():
    """background_conditions with evidence_quote=None should not count or crash."""
    from scripts.validation_report import compute_quote_grounding

    ext = ClinicalReportExtraction(
        id="NCT0001",
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(
                name="peripheral neurotoxicity",
                evidence_quote="peripheral neurotoxicity",
            )
        ],
        background_conditions=[
            ExtractedDisease(name="colorectal cancer")
        ],  # no evidence_quote
        investigated_drugs=[
            ExtractedDrug(drug="neurotropin", evidence_quote="neurotropin")
        ],
    )
    input_df = pl.DataFrame(
        {
            "id": ["NCT0001"],
            "clinicalStage": ["PHASE_3"],
            "drugs": [[{"drugFromSource": "neurotropin", "drugId": None}]],
            "diseases": [
                [{"diseaseFromSource": "peripheral neurotoxicity", "diseaseId": None}]
            ],
            "trialOfficialTitle": ["peripheral neurotoxicity neurotropin"],
            "trialDescription": [None],
            "trialDetailedDescription": [None],
        }
    )
    result = compute_quote_grounding([ext], input_df)
    # Only the 2 non-None quotes are counted; the None background_condition is skipped
    assert result["total_quotes"] == 2
    assert result["grounded"] == 2


def test_compute_quote_grounding_finds_present_quotes():
    from scripts.validation_report import compute_quote_grounding

    ext = ClinicalReportExtraction(
        id="NCT0001",
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[
            ExtractedDisease(
                name="headache",
                evidence_quote="patients with chronic headache were enrolled",
            )
        ],
        investigated_drugs=[
            ExtractedDrug(
                drug="aspirin",
                evidence_quote="aspirin 100mg daily was administered",
            )
        ],
    )
    input_df = pl.DataFrame(
        {
            "id": ["NCT0001"],
            "clinicalStage": ["PHASE_2"],
            "drugs": [[{"drugFromSource": "aspirin", "drugId": None}]],
            "diseases": [[{"diseaseFromSource": "headache", "diseaseId": None}]],
            "trialOfficialTitle": [
                "patients with chronic headache were enrolled; aspirin 100mg daily was administered"
            ],
            "trialDescription": [None],
            "trialDetailedDescription": [None],
        }
    )
    result = compute_quote_grounding([ext], input_df)
    assert result["total_quotes"] == 2
    assert result["grounded"] == 2
    assert result["grounded_pct"] == 100.0


def test_has_unmatched_drug_false_when_labels_overlap():
    from scripts.validation_report import build_comparison_df

    ext = _make_extraction("NCT0001", "Aspirin", "headache")
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}]
    )
    df = build_comparison_df([ext], input_df)
    assert df["has_unmatched_drug"][0] is False


def test_has_unmatched_drug_true_when_no_label_overlap():
    from scripts.validation_report import build_comparison_df

    ext = _make_extraction("NCT0001", "ibuprofen", "headache")
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}]
    )
    df = build_comparison_df([ext], input_df)
    assert df["has_unmatched_drug"][0] is True


def test_has_unmatched_drug_handles_partial_name_match():
    from scripts.validation_report import build_comparison_df

    # Input uses brand name "Entrectinib", LLM extracts INN "entrectinib" — substring match
    ext = _make_extraction("NCT0001", "entrectinib", "lung cancer")
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "Entrectinib", "disease": "lung cancer"}]
    )
    df = build_comparison_df([ext], input_df)
    assert df["has_unmatched_drug"][0] is False


def test_build_comparison_df_surfaces_drug_modifier():
    """route/formulation should appear as a comma-joined output_drug_modifier column."""
    from scripts.validation_report import build_comparison_df

    ext = ClinicalReportExtraction(
        id="NCT0001",
        drug_intent="therapeutic",
        drug_intent_confidence=0.95,
        primary_indications=[ExtractedDisease(name="cancer", evidence_quote="cancer")],
        investigated_drugs=[
            ExtractedDrug(
                drug="metformin",
                route="oral",
                formulation="tablet",
                evidence_quote="oral metformin tablet",
            ),
        ],
    )
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "metformin", "disease": "cancer"}]
    )
    df = build_comparison_df([ext], input_df)
    modifier = df["output_drug_modifier"][0]
    assert "oral" in modifier
    assert "tablet" in modifier


def test_build_comparison_df_drug_modifier_empty_when_plain():
    """A drug with no formulation/route/dosage_form yields an empty modifier column."""
    from scripts.validation_report import build_comparison_df

    ext = _make_extraction("NCT0001", "metformin", "diabetes")
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "metformin", "disease": "diabetes"}]
    )
    df = build_comparison_df([ext], input_df)
    assert df["output_drug_modifier"][0] == ""


def test_has_unmatched_disease_false_when_labels_overlap():
    from scripts.validation_report import build_comparison_df

    ext = _make_extraction("NCT0001", "aspirin", "Non-small cell lung cancer")
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "aspirin", "disease": "non-small cell lung cancer"}]
    )
    df = build_comparison_df([ext], input_df)
    assert df["has_unmatched_disease"][0] is False


def test_has_unmatched_disease_true_when_no_label_overlap():
    from scripts.validation_report import build_comparison_df

    ext = _make_extraction("NCT0001", "aspirin", "migraine")
    input_df = _make_input_df(
        [{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}]
    )
    df = build_comparison_df([ext], input_df)
    assert df["has_unmatched_disease"][0] is True


def test_search_drug_disease_plausibility_returns_bool():
    from unittest.mock import MagicMock, patch
    from scripts.validation_report import search_drug_disease_plausibility

    with patch("scripts.validation_report.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "Aspirin is used to treat headache.",
            "RelatedTopics": [],
        }
        mock_httpx.get.return_value = mock_response
        result = search_drug_disease_plausibility("aspirin", "headache")
        assert isinstance(result, bool)
        assert result is True

class TestParseSingleRecord:
    """Unit tests for _parse_single_record."""

    SAMPLE_LINE = r"""{"id": "batch_req_6a1d6fccc80c8190b99bfd7b4e3af5fc", "custom_id": "nct00031889", "response": {"status_code": 200, "request_id": "039b5f8b-14a7-4df2-ac05-d1432b689a97", "body": {"id": "resp_0b866af5f6b78e2e016a1d6d1815b881978a79910735ecbf3c", "object": "response", "created_at": 1780313368, "status": "completed", "background": false, "error": null, "output": [{"id": "msg_0b866af5f6b78e2e016a1d6d18f5d48197b49a83dc713967b9", "type": "message", "status": "completed", "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "{\"id\":\"nct00031889\",\"drug_intent\":\"therapeutic\",\"drug_intent_confidence\":0.95,\"primary_indications\":[{\"name\":\"prostate cancer\",\"severity\":null,\"stage\":\"stage IV\",\"onset\":null,\"etiology\":null,\"evidence_quote\":\"treating patients who have stage IV prostate cancer that has been previously treated with hormone therapy or surgery\"}],\"background_conditions\":[{\"name\":\"prostate cancer\",\"severity\":null,\"stage\":null,\"onset\":null,\"etiology\":null,\"evidence_quote\":\"failure of androgen suppression (luteinizing hormone-releasing hormone agonist or orchiectomy) in patients with stage IV prostate cancer\"}],\"investigated_drugs\":[{\"drug\":\"exemestane\",\"route\":\"oral\",\"formulation\":\"tablet\",\"synonyms\":null,\"dosages\":[\"once daily\"],\"evidence_quote\":\"Patients receive oral exemestane once daily.\"},{\"drug\":\"bicalutamide\",\"route\":\"oral\",\"formulation\":\"tablet\",\"synonyms\":null,\"dosages\":[\"once daily\"],\"evidence_quote\":\"Patients receive ... oral bicalutamide once daily.\"}],\"comparator_drugs\":null,\"supportive_drugs\":null,\"conclusion\":null}"}], "role": "assistant"}], "usage": {"input_tokens": 4710, "output_tokens": 258, "total_tokens": 4968}}}}"""

    # ── happy path ────────────────────────────────────────────────────────────

    def test_happy_path_top_level_fields(self):
        good, bad = _parse_single_record(json.loads(self.SAMPLE_LINE))

        assert bad is None
        assert good["id"] == "nct00031889"
        assert good["drug_intent"] == "therapeutic"
        assert good["drug_intent_confidence"] == pytest.approx(0.95)

    def test_primary_indications_parsed(self):
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        assert len(good["primary_indications"]) == 1
        ind = good["primary_indications"][0]
        assert ind["name"] == "prostate cancer"
        assert ind["stage"] == "stage IV"
        assert ind["severity"] is None
        assert ind["etiology"] is None

    def test_investigated_drugs_parsed(self):
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        drugs = good["investigated_drugs"]
        assert len(drugs) == 2
        names = {d["drug"] for d in drugs}
        assert names == {"exemestane", "bicalutamide"}

        exemestane = next(d for d in drugs if d["drug"] == "exemestane")
        assert exemestane["route"] == "oral"
        assert exemestane["formulation"] == "tablet"
        assert exemestane["dosages"] == ["once daily"]
        assert exemestane["synonyms"] is None

    def test_optional_list_fields_are_empty_not_null(self):
        """comparator_drugs and supportive_drugs are null in the payload —
        normalised to [] so List(Struct) columns stay consistent."""
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        assert good["comparator_drugs"] == []
        assert good["supportive_drugs"] == []

    def test_record_builds_valid_dataframe(self):
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        df = pl.DataFrame([good], schema=EXTRACTION_SCHEMA)
        assert df.shape == (1, len(EXTRACTION_SCHEMA))
        assert df["drug_intent_confidence"].dtype == pl.Float64

    # ── error paths ───────────────────────────────────────────────────────────

    def test_missing_text_path_returns_bad_record(self):
        broken = {"custom_id": "nct99999", "response": {"body": {}}}
        good, bad = _parse_single_record(broken)

        assert good is None
        assert bad["id"] == "nct99999"
        assert "missing_text_path" in bad["error"]

    def test_malformed_inner_json_returns_bad_record(self):
        outer = json.loads(self.SAMPLE_LINE)
        outer["response"]["body"]["output"][0]["content"][0]["text"] = "{not valid json"
        good, bad = _parse_single_record(outer)

        assert good is None
        assert "inner_json_error" in bad["error"]