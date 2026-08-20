import polars as pl


def _make_studies() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nct_id": ["NCT0001"],
            "overall_status": ["COMPLETED"],
            "phase": ["PHASE2"],
            "study_type": ["INTERVENTIONAL"],
            "start_date": [None],
            "why_stopped": [None],
            "number_of_arms": [2],
            "official_title": ["A Study"],
        }
    )


def _make_interventions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nct_id": ["NCT0001"],
            "intervention_type": ["DRUG"],
            "name": ["Aspirin"],
        }
    )


def _make_conditions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nct_id": ["NCT0001"],
            "downcase_name": ["headache"],
        }
    )


def test_source_and_provider():
    from clinical_mining.data_sources.aact import extract_clinical_report

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
    )

    assert result.df["source"].to_list() == ["ClinicalTrials.gov"]
    assert result.df["provider"].to_list() == ["AACT"]


def test_detailed_descriptions_column_present():
    from clinical_mining.data_sources.aact import extract_clinical_report

    detailed_descriptions = pl.DataFrame(
        {
            "nct_id": ["NCT0001"],
            "description": ["A detailed protocol description."],
        }
    )
    detailed_descriptions_rename = detailed_descriptions.rename(
        {"description": "detailed_description"}
    )

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
        additional_metadata=[detailed_descriptions_rename],
    )

    assert "trialDetailedDescription" in result.df.columns


def test_no_detailed_descriptions_column_absent():
    from clinical_mining.data_sources.aact import extract_clinical_report

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
    )

    assert "trialDetailedDescription" not in result.df.columns


def test_detailed_description_value_preserved():
    from clinical_mining.data_sources.aact import extract_clinical_report

    detailed_descriptions = pl.DataFrame(
        {
            "nct_id": ["NCT0001"],
            "description": ["Detailed protocol text."],
        }
    )
    detailed_descriptions_rename = detailed_descriptions.rename(
        {"description": "detailed_description"}
    )

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
        additional_metadata=[detailed_descriptions_rename],
    )

    values = result.df["trialDetailedDescription"].to_list()
    assert values[0] == "Detailed protocol text."


def test_replace_with_llm_indications():
    """Test that LLM indications replace source indications for LLM-covered trials."""
    from clinical_mining.data_sources.aact.clinical_report import (
        replace_with_llm_indications,
    )

    _disease_struct = pl.Struct(
        {
            "name": pl.String,
            "severity": pl.String,
            "stage": pl.String,
            "onset": pl.String,
            "etiology": pl.String,
            "evidence_quote": pl.String,
        }
    )
    _drug_struct = pl.Struct(
        {
            "drug": pl.String,
            "route": pl.String,
            "formulation": pl.String,
            "synonyms": pl.List(pl.String),
            "dosages": pl.List(pl.String),
            "evidence_quote": pl.String,
        }
    )

    studies = pl.DataFrame(
        {
            "nct_id": ["NCT1", "NCT1", "NCT2", "NCT3", "NCT4"],
            "diseaseFromSource": [
                "colorectal cancer",
                "pain",
                "diabetes",
                "dementia",
                "heart failure",
            ],
            "drugFromSource": [
                "acetaminophen",
                "acetaminophen",
                "metformin",
                "carbamazepine",
                "lisinopril",
            ],
            "trial_phase": ["PHASE2", "PHASE2", "PHASE3", "PHASE1", "PHASE4"],
        }
    )

    def disease(name: str) -> dict:
        return {
            "name": name,
            "severity": None,
            "stage": None,
            "onset": None,
            "etiology": None,
            "evidence_quote": name,
        }

    def drug(name: str) -> dict:
        return {
            "drug": name,
            "route": None,
            "formulation": None,
            "synonyms": None,
            "dosages": None,
            "evidence_quote": name,
        }

    extractions = pl.DataFrame(
        {
            "id": ["NCT1", "NCT3", "nct4"],
            "primary_indications": [
                [disease("metastatic colorectal cancer")],
                [],
                [disease("congestive heart failure")],
            ],
            "investigated_drugs": [
                [drug("acetaminophen")],
                [drug("carbamazepine")],
                [drug("lisinopril")],
            ],
        },
        schema={
            "id": pl.String,
            "primary_indications": pl.List(_disease_struct),
            "investigated_drugs": pl.List(_drug_struct),
        },
    )

    result = replace_with_llm_indications(studies, extractions)

    # LLM-covered trial uses LLM indications only — "pain" row is gone
    nct1_diseases = result.filter(pl.col("nct_id") == "NCT1")[
        "diseaseFromSource"
    ].to_list()
    assert nct1_diseases == ["metastatic colorectal cancer"], (
        "Expected LLM indication to replace original source indications entirely"
    )

    # LLM-covered trial does not bleed original disease/drug pairs
    assert "pain" not in result["diseaseFromSource"].to_list(), (
        "Original indications for LLM-covered trials must be fully replaced"
    )

    # Uncovered trial gets null indications
    nct2 = result.filter(pl.col("nct_id") == "NCT2")
    assert nct2["diseaseFromSource"].to_list() == [None]
    assert nct2["drugFromSource"].to_list() == [None]

    # LLM-covered trial NCT3 should not have disease information
    # the result of the extraction was None
    nct3 = result.filter(pl.col("nct_id") == "NCT3")
    assert nct3["drugFromSource"].to_list() == ["carbamazepine"]
    assert nct3["diseaseFromSource"][0] is None

    # LLM integration is case insensitive
    nct4 = result.filter(pl.col("nct_id") == "NCT4")
    assert nct4["diseaseFromSource"].to_list() == ["congestive heart failure"]


def test_extract_clinical_report_with_sponsors():
    from clinical_mining.data_sources.aact import extract_clinical_report

    sponsors = pl.DataFrame(
        {
            "nct_id": ["NCT0001", "NCT0001", "NCT0002"],
            "agency_class": ["INDUSTRY", "NIH", "INDUSTRY"],
            "lead_or_collaborator": ["lead", "collaborator", "lead"],
            "name": ["Sponsor A", "Sponsor B", "Sponsor C"],
        }
    )

    lead_sponsors = sponsors.filter(pl.col("lead_or_collaborator") == "lead")

    aggregation_specs = {
        "sponsor": {
            "group_by": "nct_id",
            "alias": "sponsor",
            "struct": {
                "agencyClass": "agency_class",
                "name": "name",
            },
            "agg": "first",
        }
    }

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
        additional_metadata=[lead_sponsors],
        aggregation_specs=aggregation_specs,
    )

    assert "trialSponsor" in result.df.columns
    trial_sponsor_nct0001 = result.df.filter(pl.col("id") == "nct0001")[
        "trialSponsor"
    ].to_list()[0]
    assert trial_sponsor_nct0001 == {"agencyClass": "INDUSTRY", "name": "Sponsor A"}


def test_extract_clinical_report_with_study_references():
    from clinical_mining.data_sources.aact import extract_clinical_report

    study_references = pl.DataFrame(
        {
            "nct_id": ["NCT0001", "NCT0001", "NCT0001", "NCT0002"],
            "pmid": [12345678, 99999999, 12345678, 11111111],
            "reference_type": ["result", "background", "result", "result"],
        }
    )

    aggregation_specs = {
        "study_references": {
            "group_by": "nct_id",
            "alias": "literature",
            "struct": {
                "id": "pmid",
                "type": "reference_type",
            },
            "agg": "unique",
        }
    }

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
        additional_metadata=[study_references],
        aggregation_specs=aggregation_specs,
    )

    assert "trialLiterature" in result.df.columns
    assert sorted(result.df["trialLiterature"].to_list()[0], key=lambda x: x["id"]) == [
        {"id": "12345678", "type": "result"},
        {"id": "99999999", "type": "background"},
    ]
