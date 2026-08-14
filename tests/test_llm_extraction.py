import json

import polars as pl
import pytest
from pydantic import ValidationError

from clinical_mining.data_sources.aact.llm_extractor import (
    _parse_single_record,
)
from clinical_mining.schemas import (
    ClinicalReportExtractionSchema as ClinicalReportExtraction,
)
from clinical_mining.schemas import (
    ExtractedDisease,
    ExtractedDrug,
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


class TestParseSingleRecord:
    """Unit tests for _parse_single_record."""

    SAMPLE_LINE = r"""{"id": "batch_req_6a1d6fccc80c8190b99bfd7b4e3af5fc", "custom_id": "nct00031889", "response": {"status_code": 200, "request_id": "039b5f8b-14a7-4df2-ac05-d1432b689a97", "body": {"id": "resp_0b866af5f6b78e2e016a1d6d1815b881978a79910735ecbf3c", "object": "response", "created_at": 1780313368, "status": "completed", "background": false, "error": null, "output": [{"id": "msg_0b866af5f6b78e2e016a1d6d18f5d48197b49a83dc713967b9", "type": "message", "status": "completed", "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "{\"id\":\"nct00031889\",\"drug_intent\":\"therapeutic\",\"drug_intent_confidence\":0.95,\"primary_indications\":[{\"name\":\"prostate cancer\",\"severity\":null,\"stage\":\"stage IV\",\"onset\":null,\"etiology\":null,\"evidence_quote\":\"treating patients who have stage IV prostate cancer that has been previously treated with hormone therapy or surgery\"}],\"background_conditions\":[{\"name\":\"prostate cancer\",\"severity\":null,\"stage\":null,\"onset\":null,\"etiology\":null,\"evidence_quote\":\"failure of androgen suppression (luteinizing hormone-releasing hormone agonist or orchiectomy) in patients with stage IV prostate cancer\"}],\"investigated_drugs\":[{\"drug\":\"exemestane\",\"route\":\"oral\",\"formulation\":\"tablet\",\"synonyms\":null,\"dosages\":[\"once daily\"],\"evidence_quote\":\"Patients receive oral exemestane once daily.\"},{\"drug\":\"bicalutamide\",\"route\":\"oral\",\"formulation\":\"tablet\",\"synonyms\":null,\"dosages\":[\"once daily\"],\"evidence_quote\":\"Patients receive ... oral bicalutamide once daily.\"}],\"comparator_drugs\":null,\"supportive_drugs\":null,\"conclusion\":null}"}], "role": "assistant"}], "usage": {"input_tokens": 4710, "output_tokens": 258, "total_tokens": 4968}}}}"""

    # ── happy path ────────────────────────────────────────────────────────────

    def test_happy_path_top_level_fields(self):
        good, bad = _parse_single_record(json.loads(self.SAMPLE_LINE))

        assert bad is None
        assert good.id == "nct00031889"
        assert good.drug_intent == "therapeutic"
        assert good.drug_intent_confidence == pytest.approx(0.95)

    def test_primary_indications_parsed(self):
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        assert len(good.primary_indications) == 1
        ind = good.primary_indications[0]
        assert ind.name == "prostate cancer"
        assert ind.stage == "stage IV"
        assert ind.severity is None
        assert ind.etiology is None

    def test_investigated_drugs_parsed(self):
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        drugs = good.investigated_drugs
        assert len(drugs) == 2
        names = {d.drug for d in drugs}
        assert names == {"exemestane", "bicalutamide"}

        exemestane = next(d for d in drugs if d.drug == "exemestane")
        assert exemestane.route == "oral"
        assert exemestane.formulation == "tablet"
        assert exemestane.dosages == ["once daily"]
        assert exemestane.synonyms is None

    def test_optional_list_fields_are_none(self):
        """comparator_drugs and supportive_drugs are null in the payload."""
        good, _ = _parse_single_record(json.loads(self.SAMPLE_LINE))

        assert good.comparator_drugs is None
        assert good.supportive_drugs is None

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

    # ── multi-item output ─────────────────────────────────────────────────────

    def _message_item(self, text: str) -> dict:
        return {
            "id": "msg_test",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    def _valid_text(self) -> str:
        return json.loads(self.SAMPLE_LINE)["response"]["body"]["output"][0]["content"][
            0
        ]["text"]

    def test_single_output_item_still_parses(self):
        """A record with one output item is unaffected by the fallback."""
        outer = json.loads(self.SAMPLE_LINE)
        assert len(outer["response"]["body"]["output"]) == 1

        good, bad = _parse_single_record(outer)

        assert bad is None
        assert good.id == "nct00031889"
        assert len(good.investigated_drugs) == 2

    def test_truncated_first_item_falls_back_to_continuation(self):
        """A cut-off first item must not discard the complete later payload."""
        text = self._valid_text()
        outer = json.loads(self.SAMPLE_LINE)
        outer["response"]["body"]["output"] = [
            self._message_item(text[:287]),  # unterminated string
            self._message_item(text),
        ]

        good, bad = _parse_single_record(outer)

        assert bad is None
        assert good.id == "nct00031889"
        assert good.drug_intent == "therapeutic"
        assert {d.drug for d in good.investigated_drugs} == {
            "exemestane",
            "bicalutamide",
        }

    def test_multiple_texts_within_one_item_are_candidates(self):
        """Fragments split across content entries of a single item also count."""
        text = self._valid_text()
        outer = json.loads(self.SAMPLE_LINE)
        outer["response"]["body"]["output"][0]["content"] = [
            {"type": "output_text", "text": text[:287]},
            {"type": "output_text", "text": text},
        ]

        good, bad = _parse_single_record(outer)

        assert bad is None
        assert good.id == "nct00031889"

    def test_no_valid_payload_anywhere_returns_bad_record(self):
        """Every fragment truncated → still a bad record, with the same error."""
        text = self._valid_text()
        outer = json.loads(self.SAMPLE_LINE)
        outer["response"]["body"]["output"] = [
            self._message_item(text[:287]),
            self._message_item(text[:412]),
        ]

        good, bad = _parse_single_record(outer)

        assert good is None
        assert bad["id"] == "nct00031889"
        assert "inner_json_error" in bad["error"]

    def test_non_message_output_item_is_skipped(self):
        """Only items of type 'message' are considered."""
        text = self._valid_text()
        outer = json.loads(self.SAMPLE_LINE)
        outer["response"]["body"]["output"] = [
            {
                "id": "rs_test",
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "{not valid json"}],
            },
            self._message_item(text),
        ]

        good, bad = _parse_single_record(outer)

        assert bad is None
        assert good.id == "nct00031889"

    def test_only_non_message_items_returns_missing_text_path(self):
        outer = json.loads(self.SAMPLE_LINE)
        outer["response"]["body"]["output"] = [
            {"id": "rs_test", "type": "reasoning", "content": []},
        ]

        good, bad = _parse_single_record(outer)

        assert good is None
        assert "missing_text_path" in bad["error"]
