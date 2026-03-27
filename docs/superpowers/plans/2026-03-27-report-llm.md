# LLM Extraction from Clinical Trial Reports — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone async script that calls OpenRouter's `gpt-5.4-mini` to extract structured drug, disease, and intervention data from clinical trial reports, then validates results against input data.

**Architecture:** A single script (`scripts/llm_extraction.py`) reads a local parquet, samples 100 records with a fixed seed, calls OpenRouter concurrently (semaphore-limited), validates each response against a new `ClinicalReportExtraction` Pydantic schema, writes JSONL output, and prints a five-check validation report to stdout.

**Tech Stack:** Python 3.11+, polars, pydantic v2, httpx (async + sync), argparse, pytest + pytest-asyncio

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add `httpx`, `pytest`, `pytest-asyncio` dependencies |
| Modify | `src/clinical_mining/schemas.py` | Add 4 new extraction Pydantic classes |
| Create | `scripts/llm_extraction.py` | Full standalone script: load → sample → prompt → call → validate → write → report |
| Create | `tests/test_llm_extraction.py` | Unit tests for all testable functions |

---

### Task 1: Set up branch and add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b report_llm
```

Expected: `Switched to a new branch 'report_llm'`

- [ ] **Step 2: Add httpx to project dependencies and pytest to dev**

Edit `pyproject.toml`. Replace the `dependencies` and `[project.optional-dependencies]` sections:

```toml
dependencies = [
    "altair>=5.5.0",
    "connectorx>=0.4.3",
    "fastexcel>=0.16.0",
    "httpx>=0.28.0",
    "hydra-core>=1.3.2",
    "ipykernel>=6.29.5",
    "loguru>=0.7.3",
    "ontoma>=2.3.0",
    "pandas>=2.3.3",
    "pdfplumber>=0.11.8",
    "polars>=1.34.0",
    "polars-hash>=0.5.5",
    "pyarrow>=21.0.0",
    "pydantic>=2.11.7",
    "sqlalchemy>=2.0.42",
]

[project.optional-dependencies]
dev = [
    "ty>=0.0.15",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
oracle = [
    "cx-oracle>=8.3.0",
]
```

- [ ] **Step 3: Sync dependencies**

```bash
uv sync --extra dev
```

Expected: output shows `httpx`, `pytest`, `pytest-asyncio` installed.

- [ ] **Step 4: Verify httpx is importable**

```bash
uv run python -c "import httpx; print(httpx.__version__)"
```

Expected: prints a version string (e.g. `0.28.x`)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add httpx and pytest dependencies for LLM extraction POC"
```

---

### Task 2: Add extraction schemas to schemas.py

**Files:**
- Modify: `src/clinical_mining/schemas.py`
- Create: `tests/test_llm_extraction.py`

- [ ] **Step 1: Create the tests directory and write failing schema tests**

```bash
mkdir -p tests
```

Create `tests/test_llm_extraction.py` with the following content:

```python
import pytest
from pydantic import ValidationError
from clinical_mining.schemas import (
    ClinicalReportExtraction,
    ClinicalExtractedIntervention,
    ExtractedDrug,
    ExtractedDisease,
)


def test_extracted_drug_required_field():
    drug = ExtractedDrug(drugFromSource="ibuprofen")
    assert drug.drugFromSource == "ibuprofen"
    assert drug.mechanismOfAction is None
    assert drug.target is None


def test_extracted_drug_optional_fields():
    drug = ExtractedDrug(
        drugFromSource="aspirin",
        mechanismOfAction="COX inhibitor",
        target="COX-1",
    )
    assert drug.mechanismOfAction == "COX inhibitor"
    assert drug.target == "COX-1"


def test_extracted_drug_missing_required_raises():
    with pytest.raises(ValidationError):
        ExtractedDrug()


def test_extracted_disease_required_field():
    disease = ExtractedDisease(condition="diabetes")
    assert disease.condition == "diabetes"
    assert disease.backgroundCondition is None


def test_extracted_disease_optional_background():
    disease = ExtractedDisease(condition="fever", backgroundCondition="malaria")
    assert disease.backgroundCondition == "malaria"


def test_clinical_extracted_intervention():
    intervention = ClinicalExtractedIntervention(
        drugs=[ExtractedDrug(drugFromSource="metformin")],
        diseases=[ExtractedDisease(condition="type 2 diabetes")],
    )
    assert len(intervention.drugs) == 1
    assert len(intervention.diseases) == 1


def test_clinical_report_extraction_full():
    extraction = ClinicalReportExtraction(
        id="NCT04012606",
        interventions=[
            ClinicalExtractedIntervention(
                drugs=[ExtractedDrug(drugFromSource="metformin")],
                diseases=[ExtractedDisease(condition="type 2 diabetes")],
            )
        ],
    )
    assert extraction.id == "NCT04012606"
    assert len(extraction.interventions) == 1


def test_clinical_report_extraction_from_json():
    json_str = """{
        "id": "NCT00000001",
        "interventions": [
            {
                "drugs": [{"drugFromSource": "aspirin", "mechanismOfAction": null, "target": null}],
                "diseases": [{"condition": "headache", "backgroundCondition": null}]
            }
        ]
    }"""
    extraction = ClinicalReportExtraction.model_validate_json(json_str)
    assert extraction.id == "NCT00000001"
    assert extraction.interventions[0].drugs[0].drugFromSource == "aspirin"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_extraction.py -v
```

Expected: `ImportError` — `ClinicalReportExtraction` not yet defined.

- [ ] **Step 3: Add the four new classes to schemas.py**

Append the following to the end of `src/clinical_mining/schemas.py`:

```python


class ExtractedDrug(BaseModel):
    """Drug used in a clinical trial intervention.

    Targets specific molecules, medicines or compounds — avoids broad categories
    like 'chemotherapy' or 'antibiotics' but captures specific drugs within them.
    """

    drugFromSource: str = Field(
        ..., description="Drug used for the same intervention."
    )
    mechanismOfAction: str | None = Field(
        default=None,
        description="The mechanism of action by which the drug exercises its on-target effect.",
    )
    target: str | None = Field(
        default=None,
        description="Biological entity (e.g. gene, protein, cell) the drug modulates.",
    )


class ExtractedDisease(BaseModel):
    """Indication or disease associated with a clinical trial intervention."""

    condition: str = Field(..., description="The name of the disease or condition.")
    backgroundCondition: str | None = Field(
        default=None,
        description=(
            "Condition common to all patients in the intervention that is underlying "
            "all patients in the group. The drug is not intended to treat the "
            "backgroundCondition but it might modify the effect of the drug. "
            "E.g. in acetaminophen treating fever in malaria patients, malaria is "
            "the backgroundCondition."
        ),
    )


class ClinicalExtractedIntervention(BaseModel):
    """A single intervention arm in a clinical trial."""

    drugs: list[ExtractedDrug] = Field(
        ...,
        description=(
            "All compounds mentioned in this intervention. Targets specific molecules "
            "and medicines — avoids broad medicinal groups (chemotherapy, antibiotics, "
            "Chinese treatment, devices) but allows specific medicines within them."
        ),
    )
    diseases: list[ExtractedDisease] = Field(
        ..., description="All diseases or conditions mentioned in this intervention."
    )


class ClinicalReportExtraction(BaseModel):
    """LLM-extracted structured information from a clinical trial report."""

    id: str = Field(
        ...,
        description="The identifier for the clinical reference, e.g. NCT04012606.",
    )
    interventions: list[ClinicalExtractedIntervention] = Field(
        ..., description="The interventions associated with the report."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_extraction.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clinical_mining/schemas.py tests/test_llm_extraction.py
git commit -m "feat: add ClinicalReportExtraction schema classes"
```

---

### Task 3: Implement data loading and prompt building

**Files:**
- Create: `scripts/llm_extraction.py` (initial skeleton + two functions)
- Modify: `tests/test_llm_extraction.py`

- [ ] **Step 1: Add tests for load_and_sample and build_prompt**

Append to `tests/test_llm_extraction.py`:

```python
import polars as pl
import tempfile
import os
from pathlib import Path


def _make_sample_parquet(path: str, n: int = 20) -> pl.DataFrame:
    """Helper: write a minimal clinical report parquet for testing."""
    df = pl.DataFrame({
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
    })
    df.write_parquet(path)
    return df


def test_load_and_sample_filters_to_clinical_trials():
    from scripts.llm_extraction import load_and_sample
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.parquet")
        _make_sample_parquet(path, n=20)
        result = load_and_sample(path, sample=10, seed=42)
        assert all(result["type"] == "CLINICAL_TRIAL")


def test_load_and_sample_respects_seed():
    from scripts.llm_extraction import load_and_sample
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.parquet")
        _make_sample_parquet(path, n=20)
        result1 = load_and_sample(path, sample=5, seed=42)
        result2 = load_and_sample(path, sample=5, seed=42)
        assert result1["id"].to_list() == result2["id"].to_list()


def test_load_and_sample_different_seeds_differ():
    from scripts.llm_extraction import load_and_sample
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.parquet")
        _make_sample_parquet(path, n=20)
        result1 = load_and_sample(path, sample=5, seed=42)
        result2 = load_and_sample(path, sample=5, seed=99)
        assert result1["id"].to_list() != result2["id"].to_list()


def test_build_prompt_contains_id_and_trial_fields():
    from scripts.llm_extraction import build_prompt
    row = {
        "id": "NCT04012606",
        "trial_official_title": "A Phase 2 Study",
        "trial_description": "Tests aspirin",
        "trial_phase": "PHASE_2",
        "trial_overall_status": "COMPLETED",
        "trial_primary_purpose": "TREATMENT",
        "trial_study_type": "INTERVENTIONAL",
        "trial_number_of_arms": 2,
        "trial_why_stopped": None,
        "trial_literature": None,
        "trial_start_date": "2020-01-01",
    }
    prompt = build_prompt(row)
    assert "NCT04012606" in prompt
    assert "A Phase 2 Study" in prompt
    assert "PHASE_2" in prompt
    assert "null" in prompt  # None values rendered as null


def test_build_prompt_handles_missing_trial_fields():
    from scripts.llm_extraction import build_prompt
    row = {"id": "NCT00000001"}  # no trial_* fields
    prompt = build_prompt(row)
    assert "NCT00000001" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_extraction.py::test_load_and_sample_filters_to_clinical_trials tests/test_llm_extraction.py::test_build_prompt_contains_id_and_trial_fields -v
```

Expected: `ModuleNotFoundError` — `scripts.llm_extraction` not yet defined.

- [ ] **Step 3: Create scripts/llm_extraction.py with load_and_sample and build_prompt**

```bash
mkdir -p scripts
touch scripts/__init__.py
```

Create `scripts/llm_extraction.py`:

```python
"""LLM-based extraction of structured information from clinical trial reports.

Usage:
    uv run scripts/llm_extraction.py \\
      --input data/inputs/clinical_report/ \\
      --output data/outputs/llm_extraction/ \\
      --sample 100 \\
      --seed 42 \\
      --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

import httpx
import polars as pl

# Add src to path so clinical_mining is importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_mining.schemas import ClinicalReportExtraction

SYSTEM_PROMPT = (
    "You are an expert clinician interpreting clinical trial reports. "
    "Extract structured information about therapeutic interventions, drug molecules, "
    "mechanisms of action, and conditions from the trial data provided. "
    "Follow the output schema exactly."
)

MODEL = "openai/gpt-5.4-mini"

TRIAL_FIELD_LABELS: dict[str, str] = {
    "trial_official_title": "Official Title",
    "trial_description": "Description",
    "trial_phase": "Phase",
    "trial_overall_status": "Overall Status",
    "trial_primary_purpose": "Primary Purpose",
    "trial_study_type": "Study Type",
    "trial_number_of_arms": "Number of Arms",
    "trial_why_stopped": "Why Stopped",
    "trial_literature": "Literature",
    "trial_start_date": "Start Date",
}


def load_and_sample(input_path: str, sample: int, seed: int) -> pl.DataFrame:
    """Read parquet, filter to CLINICAL_TRIAL rows, return a reproducible random sample."""
    df = pl.read_parquet(input_path)
    df = df.filter(pl.col("type") == "CLINICAL_TRIAL")
    n = min(sample, len(df))
    return df.sample(n=n, seed=seed, shuffle=True)


def build_prompt(row: dict) -> str:
    """Serialise a record's trial_* fields into a labeled key-value prompt string."""
    lines = [f"Trial ID: {row['id']}"]
    for field, label in TRIAL_FIELD_LABELS.items():
        value = row.get(field)
        lines.append(f"{label}: {value if value is not None else 'null'}")
    return "\n".join(lines)
```

- [ ] **Step 4: Add a pytest conftest.py so scripts is importable**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Make scripts/ importable as a package during tests
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_extraction.py -k "load_and_sample or build_prompt" -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/llm_extraction.py tests/conftest.py tests/test_llm_extraction.py
git commit -m "feat: add data loading and prompt building for LLM extraction"
```

---

### Task 4: Implement the async OpenRouter client and extraction loop

**Files:**
- Modify: `scripts/llm_extraction.py`
- Modify: `tests/test_llm_extraction.py`

- [ ] **Step 1: Add tests for call_openrouter and extract_record**

Append to `tests/test_llm_extraction.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _make_openrouter_response(content: str) -> MagicMock:
    """Build a mock httpx.Response for an OpenRouter completion."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_response


VALID_EXTRACTION_JSON = json.dumps({
    "id": "NCT04012606",
    "interventions": [
        {
            "drugs": [{"drugFromSource": "aspirin", "mechanismOfAction": None, "target": None}],
            "diseases": [{"condition": "headache", "backgroundCondition": None}],
        }
    ],
})


def test_call_openrouter_sends_correct_payload():
    from scripts.llm_extraction import call_openrouter

    async def run():
        mock_client = AsyncMock()
        mock_client.post.return_value = _make_openrouter_response(VALID_EXTRACTION_JSON)
        schema = ClinicalReportExtraction.model_json_schema()
        result = await call_openrouter(mock_client, "test prompt", SYSTEM_PROMPT, MODEL, schema)
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == MODEL
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["content"] == "test prompt"
        assert payload["response_format"]["type"] == "json_schema"
        return result

    result = asyncio.run(run())
    assert result["choices"][0]["message"]["content"] == VALID_EXTRACTION_JSON


def test_extract_record_success():
    from scripts.llm_extraction import extract_record

    async def run():
        semaphore = asyncio.Semaphore(1)
        mock_client = AsyncMock()
        mock_client.post.return_value = _make_openrouter_response(VALID_EXTRACTION_JSON)
        schema = ClinicalReportExtraction.model_json_schema()
        row = {"id": "NCT04012606", "trial_official_title": "Test"}
        extraction, error = await extract_record(semaphore, mock_client, row, SYSTEM_PROMPT, MODEL, schema)
        return extraction, error

    extraction, error = asyncio.run(run())
    assert extraction is not None
    assert error is None
    assert extraction.id == "NCT04012606"


def test_extract_record_http_error_returns_error_dict():
    from scripts.llm_extraction import extract_record

    async def run():
        semaphore = asyncio.Semaphore(1)
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPError("connection failed")
        schema = ClinicalReportExtraction.model_json_schema()
        row = {"id": "NCT00000001", "trial_official_title": "Test"}
        extraction, error = await extract_record(semaphore, mock_client, row, SYSTEM_PROMPT, MODEL, schema)
        return extraction, error

    extraction, error = asyncio.run(run())
    assert extraction is None
    assert error is not None
    assert error["id"] == "NCT00000001"
    assert "connection failed" in error["error"]


def test_extract_record_invalid_json_returns_error():
    from scripts.llm_extraction import extract_record

    async def run():
        semaphore = asyncio.Semaphore(1)
        mock_client = AsyncMock()
        mock_client.post.return_value = _make_openrouter_response("not valid json {{{")
        schema = ClinicalReportExtraction.model_json_schema()
        row = {"id": "NCT00000002"}
        extraction, error = await extract_record(semaphore, mock_client, row, SYSTEM_PROMPT, MODEL, schema)
        return extraction, error

    extraction, error = asyncio.run(run())
    assert extraction is None
    assert error is not None
    assert error["id"] == "NCT00000002"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_extraction.py -k "call_openrouter or extract_record" -v
```

Expected: `ImportError` — `call_openrouter` and `extract_record` not yet defined.

- [ ] **Step 3: Add call_openrouter and extract_record to scripts/llm_extraction.py**

Append to `scripts/llm_extraction.py`:

```python

async def call_openrouter(
    client: httpx.AsyncClient,
    prompt: str,
    system_prompt: str,
    model: str,
    schema: dict,
) -> dict:
    """Make a single structured-output request to OpenRouter."""
    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ClinicalReportExtraction",
                    "schema": schema,
                    "strict": True,
                },
            },
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


async def extract_record(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    row: dict,
    system_prompt: str,
    model: str,
    schema: dict,
) -> tuple[ClinicalReportExtraction | None, dict | None]:
    """Extract structured information from a single record; return (result, error)."""
    async with semaphore:
        try:
            prompt = build_prompt(row)
            response = await call_openrouter(client, prompt, system_prompt, model, schema)
            content = response["choices"][0]["message"]["content"]
            extraction = ClinicalReportExtraction.model_validate_json(content)
            return extraction, None
        except Exception as e:
            return None, {"id": row["id"], "error": str(e)}


async def run_extraction(
    records: list[dict],
    system_prompt: str,
    model: str,
    concurrency: int,
) -> tuple[list[ClinicalReportExtraction], list[dict]]:
    """Run async extraction over all records with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    schema = ClinicalReportExtraction.model_json_schema()
    async with httpx.AsyncClient() as client:
        tasks = [
            extract_record(semaphore, client, row, system_prompt, model, schema)
            for row in records
        ]
        results = await asyncio.gather(*tasks)
    extractions = [r for r, _ in results if r is not None]
    errors = [e for _, e in results if e is not None]
    return extractions, errors
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_extraction.py -k "call_openrouter or extract_record" -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_extraction.py tests/test_llm_extraction.py
git commit -m "feat: add async OpenRouter client and extraction loop"
```

---

### Task 5: Implement validation functions

**Files:**
- Modify: `scripts/llm_extraction.py`
- Modify: `tests/test_llm_extraction.py`

- [ ] **Step 1: Add tests for all five validation checks**

Append to `tests/test_llm_extraction.py`:

```python

def _make_extraction(
    nct_id: str,
    drug: str,
    condition: str,
    background: str | None = None,
    num_interventions: int = 1,
) -> ClinicalReportExtraction:
    interventions = [
        ClinicalExtractedIntervention(
            drugs=[ExtractedDrug(drugFromSource=drug)],
            diseases=[ExtractedDisease(condition=condition, backgroundCondition=background)],
        )
        for _ in range(num_interventions)
    ]
    return ClinicalReportExtraction(id=nct_id, interventions=interventions)


def _make_input_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "id": [r["id"] for r in rows],
        "clinicalStage": [r.get("clinicalStage", "PHASE_2") for r in rows],
        "drugs": [[{"drugFromSource": r["drug"], "drugId": None}] for r in rows],
        "diseases": [[{"diseaseFromSource": r["disease"], "diseaseId": None}] for r in rows],
    })


def test_compute_drug_match_rate_full_match():
    from scripts.llm_extraction import compute_drug_match_rate
    extractions = [_make_extraction("NCT0001", "aspirin", "headache")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}])
    result = compute_drug_match_rate(extractions, input_df)
    assert result["matched"] == 1
    assert result["total"] == 1
    assert result["pct"] == 100.0


def test_compute_drug_match_rate_no_match():
    from scripts.llm_extraction import compute_drug_match_rate
    extractions = [_make_extraction("NCT0001", "ibuprofen", "headache")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}])
    result = compute_drug_match_rate(extractions, input_df)
    assert result["matched"] == 0
    assert result["pct"] == 0.0


def test_compute_drug_match_rate_case_insensitive():
    from scripts.llm_extraction import compute_drug_match_rate
    extractions = [_make_extraction("NCT0001", "Aspirin", "headache")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}])
    result = compute_drug_match_rate(extractions, input_df)
    assert result["matched"] == 1


def test_compute_disease_match_rate_full_match():
    from scripts.llm_extraction import compute_disease_match_rate
    extractions = [_make_extraction("NCT0001", "aspirin", "headache")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}])
    result = compute_disease_match_rate(extractions, input_df)
    assert result["matched"] == 1
    assert result["pct"] == 100.0


def test_compute_disease_match_rate_no_match():
    from scripts.llm_extraction import compute_disease_match_rate
    extractions = [_make_extraction("NCT0001", "aspirin", "migraine")]
    input_df = _make_input_df([{"id": "NCT0001", "drug": "aspirin", "disease": "headache"}])
    result = compute_disease_match_rate(extractions, input_df)
    assert result["matched"] == 0


def test_compute_background_stats_no_background():
    from scripts.llm_extraction import compute_background_stats
    extractions = [
        _make_extraction("NCT0001", "aspirin", "headache", background=None),
        _make_extraction("NCT0002", "metformin", "diabetes", background=None),
    ]
    result = compute_background_stats(extractions)
    assert result["with_bg"] == 0
    assert result["pct_with_bg"] == 0.0


def test_compute_background_stats_with_background():
    from scripts.llm_extraction import compute_background_stats
    extractions = [
        _make_extraction("NCT0001", "acetaminophen", "fever", background="malaria"),
        _make_extraction("NCT0002", "metformin", "diabetes", background=None),
    ]
    result = compute_background_stats(extractions)
    assert result["with_bg"] == 1
    assert result["pct_with_bg"] == 50.0


def test_compute_interventions_by_stage():
    from scripts.llm_extraction import compute_interventions_by_stage
    extractions = [
        _make_extraction("NCT0001", "aspirin", "headache", num_interventions=2),
        _make_extraction("NCT0002", "metformin", "diabetes", num_interventions=1),
    ]
    input_df = _make_input_df([
        {"id": "NCT0001", "drug": "aspirin", "disease": "headache", "clinicalStage": "PHASE_2"},
        {"id": "NCT0002", "drug": "metformin", "disease": "diabetes", "clinicalStage": "PHASE_3"},
    ])
    result = compute_interventions_by_stage(extractions, input_df)
    assert isinstance(result, pl.DataFrame)
    assert "clinicalStage" in result.columns
    assert "total_interventions" in result.columns
    phase2_row = result.filter(pl.col("clinicalStage") == "PHASE_2")
    assert phase2_row["total_interventions"][0] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_extraction.py -k "compute_" -v
```

Expected: `ImportError` — validation functions not yet defined.

- [ ] **Step 3: Add validation functions to scripts/llm_extraction.py**

Append to `scripts/llm_extraction.py`:

```python

def compute_drug_match_rate(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> dict:
    """% of extractions where ≥1 extracted drug matches the input drugs (case-insensitive)."""
    input_drugs: dict[str, set[str]] = {}
    for row in input_df.iter_rows(named=True):
        drugs = row.get("drugs") or []
        input_drugs[row["id"]] = {
            d["drugFromSource"].lower()
            for d in drugs
            if d.get("drugFromSource")
        }

    matched = sum(
        1
        for ext in extractions
        if {
            drug.drugFromSource.lower()
            for intervention in ext.interventions
            for drug in intervention.drugs
        }
        & input_drugs.get(ext.id, set())
    )
    total = len(extractions)
    return {"matched": matched, "total": total, "pct": matched / total * 100 if total else 0.0}


def compute_disease_match_rate(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> dict:
    """% of extractions where ≥1 extracted condition matches input diseases (case-insensitive)."""
    input_diseases: dict[str, set[str]] = {}
    for row in input_df.iter_rows(named=True):
        diseases = row.get("diseases") or []
        input_diseases[row["id"]] = {
            d["diseaseFromSource"].lower()
            for d in diseases
            if d.get("diseaseFromSource")
        }

    matched = sum(
        1
        for ext in extractions
        if {
            disease.condition.lower()
            for intervention in ext.interventions
            for disease in intervention.diseases
        }
        & input_diseases.get(ext.id, set())
    )
    total = len(extractions)
    return {"matched": matched, "total": total, "pct": matched / total * 100 if total else 0.0}


def compute_background_stats(extractions: list[ClinicalReportExtraction]) -> dict:
    """% of extractions with ≥1 backgroundCondition, and % with >1."""
    def _background_count(ext: ClinicalReportExtraction) -> int:
        return sum(
            1
            for intervention in ext.interventions
            for disease in intervention.diseases
            if disease.backgroundCondition
        )

    counts = [_background_count(ext) for ext in extractions]
    total = len(extractions)
    with_bg = sum(1 for c in counts if c >= 1)
    with_multiple_bg = sum(1 for c in counts if c > 1)
    return {
        "with_bg": with_bg,
        "with_multiple_bg": with_multiple_bg,
        "pct_with_bg": with_bg / total * 100 if total else 0.0,
        "pct_with_multiple_bg": with_multiple_bg / total * 100 if total else 0.0,
    }


def compute_interventions_by_stage(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> pl.DataFrame:
    """Count total interventions per clinical stage using the input DataFrame's stage labels."""
    counts = pl.DataFrame({
        "id": [ext.id for ext in extractions],
        "num_interventions": [len(ext.interventions) for ext in extractions],
    })
    return (
        counts
        .join(input_df.select(["id", "clinicalStage"]).unique(subset="id"), on="id", how="left")
        .group_by("clinicalStage")
        .agg(pl.col("num_interventions").sum().alias("total_interventions"))
        .sort("clinicalStage")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_extraction.py -k "compute_" -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_extraction.py tests/test_llm_extraction.py
git commit -m "feat: add validation functions for drug/disease match and background stats"
```

---

### Task 6: Implement web search plausibility check

**Files:**
- Modify: `scripts/llm_extraction.py`
- Modify: `tests/test_llm_extraction.py`

- [ ] **Step 1: Add test for web search plausibility**

Append to `tests/test_llm_extraction.py`:

```python

def test_search_drug_disease_plausibility_returns_bool():
    from scripts.llm_extraction import search_drug_disease_plausibility
    # Uses live DuckDuckGo — mock httpx.get for unit test
    with patch("scripts.llm_extraction.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "Aspirin is used to treat headache.",
            "RelatedTopics": [],
        }
        mock_httpx.get.return_value = mock_response
        result = search_drug_disease_plausibility("aspirin", "headache")
        assert isinstance(result, bool)
        assert result is True


def test_search_drug_disease_plausibility_no_results_returns_false():
    from scripts.llm_extraction import search_drug_disease_plausibility
    with patch("scripts.llm_extraction.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.json.return_value = {"Abstract": "", "RelatedTopics": []}
        mock_httpx.get.return_value = mock_response
        result = search_drug_disease_plausibility("unknowndrug123", "unknowncondition456")
        assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_extraction.py -k "plausibility" -v
```

Expected: `ImportError` — `search_drug_disease_plausibility` not yet defined.

- [ ] **Step 3: Add web search plausibility function to scripts/llm_extraction.py**

Append to `scripts/llm_extraction.py`:

```python

def search_drug_disease_plausibility(drug: str, disease: str) -> bool:
    """Query DuckDuckGo Instant Answer API to check if drug-disease pair is plausible.

    Returns True if the search returns a non-empty Abstract or related topics,
    suggesting the combination is known in medical literature.
    """
    query = f"{drug} {disease} clinical trial"
    response = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_redirect": "1"},
        timeout=10.0,
    )
    data = response.json()
    return bool(data.get("Abstract")) or bool(data.get("RelatedTopics"))


def compute_web_search_plausibility(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> list[dict]:
    """For unmatched drug-disease pairs, check plausibility via web search.

    An 'unmatched' pair is one where the extracted drug does not appear in the
    input drugs list AND the extracted disease does not appear in the input diseases list.
    Returns a list of dicts with keys: id, drug, disease, plausible.
    """
    input_drugs: dict[str, set[str]] = {}
    input_diseases: dict[str, set[str]] = {}
    for row in input_df.iter_rows(named=True):
        input_drugs[row["id"]] = {
            d["drugFromSource"].lower()
            for d in (row.get("drugs") or [])
            if d.get("drugFromSource")
        }
        input_diseases[row["id"]] = {
            d["diseaseFromSource"].lower()
            for d in (row.get("diseases") or [])
            if d.get("diseaseFromSource")
        }

    results = []
    for ext in extractions:
        known_drugs = input_drugs.get(ext.id, set())
        known_diseases = input_diseases.get(ext.id, set())
        for intervention in ext.interventions:
            for drug in intervention.drugs:
                for disease in intervention.diseases:
                    drug_name = drug.drugFromSource.lower()
                    disease_name = disease.condition.lower()
                    if drug_name not in known_drugs and disease_name not in known_diseases:
                        plausible = search_drug_disease_plausibility(
                            drug.drugFromSource, disease.condition
                        )
                        results.append({
                            "id": ext.id,
                            "drug": drug.drugFromSource,
                            "disease": disease.condition,
                            "plausible": plausible,
                        })
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_extraction.py -k "plausibility" -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_extraction.py tests/test_llm_extraction.py
git commit -m "feat: add web search plausibility check for unmatched drug-disease pairs"
```

---

### Task 7: Wire CLI, output writing, and validation report

**Files:**
- Modify: `scripts/llm_extraction.py`

- [ ] **Step 1: Append main() and CLI entrypoint to scripts/llm_extraction.py**

Append to `scripts/llm_extraction.py`:

```python

def write_outputs(
    extractions: list[ClinicalReportExtraction],
    errors: list[dict],
    output_dir: str,
) -> tuple[Path, Path]:
    """Write extraction JSONL and error log to output_dir. Returns (extraction_path, error_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    extraction_path = out / f"extraction_{today}.jsonl"
    with extraction_path.open("w") as f:
        for ext in extractions:
            f.write(ext.model_dump_json() + "\n")

    error_path = out / "errors.jsonl"
    with error_path.open("w") as f:
        for err in errors:
            f.write(json.dumps(err) + "\n")

    return extraction_path, error_path


def print_validation_report(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> None:
    """Print the five-check validation report to stdout."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    total = len(extractions)
    print(f"\nTotal extracted records: {total}")
    print(f"Total errors: {len(input_df) - total}")

    drug_stats = compute_drug_match_rate(extractions, input_df)
    print(
        f"\n[1] Drug match rate: {drug_stats['matched']}/{drug_stats['total']} "
        f"({drug_stats['pct']:.1f}%)"
    )

    disease_stats = compute_disease_match_rate(extractions, input_df)
    print(
        f"[2] Disease match rate: {disease_stats['matched']}/{disease_stats['total']} "
        f"({disease_stats['pct']:.1f}%)"
    )

    bg_stats = compute_background_stats(extractions)
    print(
        f"\n[4] Background condition coverage:\n"
        f"    ≥1 background condition: {bg_stats['with_bg']} ({bg_stats['pct_with_bg']:.1f}%)\n"
        f"    >1 background condition: {bg_stats['with_multiple_bg']} ({bg_stats['pct_with_multiple_bg']:.1f}%)"
    )

    stage_df = compute_interventions_by_stage(extractions, input_df)
    print("\n[5] Interventions by clinical stage:")
    for row in stage_df.iter_rows(named=True):
        print(f"    {row['clinicalStage']}: {row['total_interventions']}")

    print("\n[3] Web search plausibility (unmatched drug-disease pairs):")
    plausibility_results = compute_web_search_plausibility(extractions, input_df)
    if not plausibility_results:
        print("    No unmatched pairs to check.")
    else:
        plausible_count = sum(1 for r in plausibility_results if r["plausible"])
        print(
            f"    {plausible_count}/{len(plausibility_results)} unmatched pairs "
            f"found plausible via web search."
        )
        for r in plausibility_results[:10]:  # show first 10
            status = "✓" if r["plausible"] else "✗"
            print(f"    [{status}] {r['id']}: {r['drug']} / {r['disease']}")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured drug/disease information from clinical trial reports using an LLM."
    )
    parser.add_argument("--input", required=True, help="Local parquet file or directory")
    parser.add_argument("--output", required=True, help="Output directory for JSONL and error log")
    parser.add_argument("--sample", type=int, default=100, help="Number of records to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel LLM calls")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {args.input}...")
    df = load_and_sample(args.input, sample=args.sample, seed=args.seed)
    print(f"Sampled {len(df)} clinical trial records.")

    records = df.to_dicts()

    print(f"Running extraction with model {MODEL} (concurrency={args.concurrency})...")
    extractions, errors = asyncio.run(
        run_extraction(records, SYSTEM_PROMPT, MODEL, args.concurrency)
    )
    print(f"Extracted: {len(extractions)} succeeded, {len(errors)} failed.")

    extraction_path, error_path = write_outputs(extractions, errors, args.output)
    print(f"Results written to {extraction_path}")
    print(f"Errors written to {error_path}")

    print_validation_report(extractions, df)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest tests/test_llm_extraction.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Verify the script's CLI help works**

```bash
uv run scripts/llm_extraction.py --help
```

Expected: prints usage with `--input`, `--output`, `--sample`, `--seed`, `--concurrency` arguments.

- [ ] **Step 4: Commit**

```bash
git add scripts/llm_extraction.py
git commit -m "feat: wire CLI, output writing, and validation report"
```

---

### Task 8: Download sample data and run pilot

**Files:**
- No code changes — data download and smoke test only

- [ ] **Step 1: Create local data directories**

```bash
mkdir -p data/inputs/clinical_report data/outputs/llm_extraction
```

- [ ] **Step 2: Download the clinical report parquet from GCS**

```bash
gsutil -m cp "gs://open-targets-data-releases/26.03/output/clinical_report/*.parquet" \
  data/inputs/clinical_report/
```

Expected: one or more `.parquet` files in `data/inputs/clinical_report/`.

- [ ] **Step 3: Set the OpenRouter API key**

```bash
export OPENROUTER_API_KEY="<your-key-here>"
```

- [ ] **Step 4: Run the pilot extraction**

```bash
uv run scripts/llm_extraction.py \
  --input data/inputs/clinical_report/ \
  --output data/outputs/llm_extraction/ \
  --sample 100 \
  --seed 42 \
  --concurrency 10
```

Expected:
- Progress messages printed to stdout
- `data/outputs/llm_extraction/extraction_<date>.jsonl` created with up to 100 lines
- `data/outputs/llm_extraction/errors.jsonl` created (may be empty)
- Validation report printed with five checks

- [ ] **Step 5: Verify output format**

```bash
head -1 data/outputs/llm_extraction/extraction_$(date +%Y-%m-%d).jsonl | python3 -m json.tool
```

Expected: valid JSON matching the `ClinicalReportExtraction` schema.

- [ ] **Step 6: Copy output to GCS**

```bash
gsutil cp data/outputs/llm_extraction/extraction_$(date +%Y-%m-%d).jsonl \
  gs://ot-team/dochoa/
```

- [ ] **Step 7: Final commit**

```bash
git add data/.gitkeep 2>/dev/null || true
git commit -m "feat: complete LLM extraction POC pilot run"
```

---

## Self-Review Notes

- All 5 validation checks from the spec are covered: drug match (Task 5), disease match (Task 5), web search (Task 6), background conditions (Task 5), interventions by stage (Task 5)
- Schema optional fields (`mechanismOfAction`, `target`, `backgroundCondition`) correctly default to `None` — verified in Task 2 tests
- Sampling reproducibility tested with two seed scenarios (Task 3)
- Error handling covers HTTP errors and invalid JSON (Task 4)
- `OPENROUTER_API_KEY` is never passed as a CLI arg — read from environment
- `data/` directories are not committed (only created at runtime)
