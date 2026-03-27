# LLM Extraction from Clinical Trial Reports — Design Spec

**Date:** 2026-03-27
**Branch:** `report_llm`
**Status:** Approved

---

## Objective

Prototype LLM-based extraction of structured information (interventions, drugs, mechanisms of action, and conditions) from clinical trial reports produced by the clinical mining pipeline. The extracted data will improve coverage and granularity of drug-disease relationships beyond what rule-based NER provides.

---

## Scope

- Standalone POC script; no pipeline integration yet
- Processes 100 randomly sampled clinical trial records (pilot)
- Input: locally downloaded parquet from `gs://open-targets-data-releases/26.03/output/clinical_report/`
- Output: validated JSONL locally, then manually copied to `gs://ot-team/dochoa/`

---

## File Layout

```
scripts/
  llm_extraction.py           # main async script
src/clinical_mining/
  schemas.py                  # + ClinicalReportExtraction and supporting models
data/
  inputs/clinical_report/     # locally downloaded parquet (not committed)
  outputs/llm_extraction/     # extraction_<date>.jsonl + errors.jsonl
docs/superpowers/specs/
  2026-03-27-report-llm-design.md
```

---

## Schema (`schemas.py`)

Four new Pydantic classes added to the existing `schemas.py`:

```python
class ExtractedDrug(BaseModel):
    drugFromSource: str
    mechanismOfAction: str | None = None
    target: str | None = None

class ExtractedDisease(BaseModel):
    condition: str
    backgroundCondition: str | None = None

class ClinicalExtractedIntervention(BaseModel):
    drugs: list[ExtractedDrug]
    diseases: list[ExtractedDisease]

class ClinicalReportExtraction(BaseModel):
    id: str
    interventions: list[ClinicalExtractedIntervention]
```

Key design decisions:
- `mechanismOfAction`, `target`, and `backgroundCondition` are all optional (`None` default)
- `drugFromSource` field name mirrors the existing `AssociatedDrug.drugFromSource` for consistency
- Schema is passed to OpenRouter as a JSON schema constraint to enforce structured output

---

## Runtime Flow

1. Read local parquet, filter to `type == CLINICAL_TRIAL`
2. Random sample 100 records with fixed seed (default: 42)
3. For each record, build a prompt from `id` + all `trial_*` columns as labeled key-value pairs
4. Call OpenRouter async with semaphore (max 10 concurrent); timeout 30s per call
5. Validate response with `ClinicalReportExtraction`; on failure write to `errors.jsonl` and continue
6. Write validated records to `extraction_<date>.jsonl`
7. Run validation analysis, print summary to stdout

---

## LLM Configuration

- **Model:** `openai/gpt-5.4-mini` via OpenRouter
- **System prompt:** "You are an expert clinician interpreting clinical trial reports. Extract structured information about therapeutic interventions, drug molecules, mechanisms of action, and conditions from the trial data provided. Follow the output schema exactly."
- **Response format:** JSON schema mode using `ClinicalReportExtraction.model_json_schema()`
- **API key:** read from `OPENROUTER_API_KEY` environment variable

---

## Prompt Format

Each record's user message serialises all `trial_*` fields as labeled key-value pairs:

```
Trial ID: NCT04012606
Official Title: ...
Description: ...
Phase: PHASE_2
Overall Status: COMPLETED
Primary Purpose: TREATMENT
Study Type: INTERVENTIONAL
Number of Arms: 2
Why Stopped: null
Literature: [...]
```

---

## Error Handling

- Pydantic validation failures and HTTP errors are caught per-record
- Failed records are written to `errors.jsonl` with `id` and error message
- Script continues processing remaining records without interruption

---

## CLI Interface

```bash
uv run scripts/llm_extraction.py \
  --input data/inputs/clinical_report/ \
  --output data/outputs/llm_extraction/ \
  --sample 100 \
  --seed 42 \
  --concurrency 10
```

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Local parquet file or directory |
| `--output` | required | Output directory for JSONL and error log |
| `--sample` | 100 | Number of records to process |
| `--seed` | 42 | Random seed for reproducible sampling |
| `--concurrency` | 10 | Max parallel LLM calls |

`OPENROUTER_API_KEY` is read from the environment — never passed as a CLI argument.

---

## Validation Report (stdout)

Five checks printed as formatted tables after extraction completes:

| Check | Description |
|---|---|
| Drug match rate | % records where ≥1 extracted `drugFromSource` matches input `drugs.drugFromSource` (case-insensitive) |
| Disease match rate | % records where ≥1 extracted `condition` matches input `diseases.diseaseFromSource` (case-insensitive) |
| Web search plausibility | For unmatched drug-disease pairs: WebSearch `"<drug> <condition> clinical trial"` to assess plausibility |
| Background condition coverage | % records with ≥1 `backgroundCondition`; % with >1 |
| Interventions by clinical stage | Count of interventions grouped by `clinicalStage` from input |

---

## Dependencies

- `httpx` — async HTTP client for OpenRouter calls (new)
- `polars`, `pydantic` — already in project
- No GCS dependencies; data transferred manually via `gcloud storage`
