import polars as pl


def _make_studies() -> pl.DataFrame:
    return pl.DataFrame({
        "nct_id": ["NCT0001"],
        "overall_status": ["COMPLETED"],
        "phase": ["PHASE2"],
        "study_type": ["INTERVENTIONAL"],
        "start_date": [None],
        "why_stopped": [None],
        "number_of_arms": [2],
        "official_title": ["A Study"],
    })


def _make_interventions() -> pl.DataFrame:
    return pl.DataFrame({
        "nct_id": ["NCT0001"],
        "intervention_type": ["DRUG"],
        "name": ["Aspirin"],
    })


def _make_conditions() -> pl.DataFrame:
    return pl.DataFrame({
        "nct_id": ["NCT0001"],
        "downcase_name": ["headache"],
    })


def test_detailed_descriptions_column_present():
    from clinical_mining.data_sources.aact import extract_clinical_report

    detailed_descriptions = pl.DataFrame({
        "nct_id": ["NCT0001"],
        "description": ["A detailed protocol description."],
    })

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
        detailed_descriptions=detailed_descriptions,
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

    detailed_descriptions = pl.DataFrame({
        "nct_id": ["NCT0001"],
        "description": ["Detailed protocol text."],
    })

    result = extract_clinical_report(
        studies=_make_studies(),
        interventions=_make_interventions(),
        conditions=_make_conditions(),
        detailed_descriptions=detailed_descriptions,
    )

    values = result.df["trialDetailedDescription"].to_list()
    assert values[0] == "Detailed protocol text."


def test_replace_with_llm_indications():
    """Test that LLM indications replace source indications for LLM-covered trials."""
    from clinical_mining.data_sources.aact.clinical_report import replace_with_llm_indications
    studies = pl.DataFrame({
        "nct_id": ["NCT1", "NCT1", "NCT2", "NCT3", "NCT4"],
        "diseaseFromSource": ["colorectal cancer", "pain", "diabetes", "dementia", "heart failure"],
        "drugFromSource": ["acetaminophen", "acetaminophen", "metformin", "carbamazepine", "lisinopril"],
        "trial_phase": ["PHASE2", "PHASE2", "PHASE3", "PHASE1", "PHASE4"],
    })

    llm_extraction_df = pl.DataFrame({
        "id": ["NCT1", "NCT3", "nct4"],
        "diseases": [["metastatic colorectal cancer"], None, ["congestive heart failure"]],
        "drugs": [["acetaminophen"], ["carbamazepine"], ["lisinopril"]],
    })

    result = replace_with_llm_indications(studies, llm_extraction_df)

    # LLM-covered trial uses LLM indications only — "pain" row is gone
    nct1_diseases = result.filter(pl.col("nct_id") == "NCT1")["diseaseFromSource"].to_list()
    assert nct1_diseases == ["metastatic colorectal cancer"], (
        "Expected LLM indication to replace original source indications entirely"
    )

    # LLM-covered trial does not bleed original disease/drug pairs
    assert "pain" not in result["diseaseFromSource"].to_list(), (
        "Original indications for LLM-covered trials must be fully replaced"
    )

    # Uncovered trial is completely untouched
    nct2 = result.filter(pl.col("nct_id") == "NCT2")
    assert nct2["diseaseFromSource"].to_list() == ["diabetes"]
    assert nct2["drugFromSource"].to_list() == ["metformin"]

    # LLM-covered trial NCT3 should not have disease information
    # the result of the extraction was None
    nct3 = result.filter(pl.col("nct_id") == "NCT3")
    assert nct3["drugFromSource"].to_list() == ["carbamazepine"]
    assert nct3["diseaseFromSource"][0] is None

    # LLM integration is case insensitive
    nct4 = result.filter(pl.col("nct_id") == "NCT4")
    assert nct4["diseaseFromSource"].to_list() == ["congestive heart failure"]
    