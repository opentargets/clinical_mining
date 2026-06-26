# AGENTS.md

Guidance for coding agents working in `clinical_mining`.

## Scope and intent

- Repository type: Python 3.11+ data pipeline for clinical-trial mining.
- Main package path: `src/clinical_mining`.
- Runtime entrypoint: `clinical_mining` console script (`clinical_mining.cli:main`).
- Pipeline behavior is config-driven via Hydra config in `src/clinical_mining/config.yaml`.
- Data processing is mostly Polars, with selective Spark usage for OnToma/NER workflows.

## Repository signals detected

- Package/dependency manager: `uv` (`uv.lock` present).
- Build backend: Hatchling (`hatchling.build` in `pyproject.toml`).
- Dev static checker: `ty` (in `project.optional-dependencies.dev`).
- Optional Oracle dependency group: `oracle` (`cx-oracle`).
- Test framework: `pytest` (in `tests/`).
- No repo-level config found for Ruff/Black/isort/mypy/pre-commit.

## Cursor/Copilot local rules

- No `.cursorrules` file found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` found.
- If any of these files are added later, treat them as highest-priority local instructions and update this document.

## Environment/setup commands

- Install default dependencies: `uv sync`
- Install dev dependencies: `uv sync --extra dev`
- Install Oracle extras when needed: `uv sync --extra oracle`
- Run CLI pipeline: `uv run clinical_mining`
- Override Hydra config at runtime (example): `uv run clinical_mining db_properties.user=<user> db_properties.password=<password>`

## Build commands

- Build wheel and sdist: `uv build`
- Quick import sanity check: `uv run python -c "import clinical_mining; print('ok')"`

## Lint/type-check commands

- Primary static check for source tree: `uv run ty check src`
- Check one file: `uv run ty check src/clinical_mining/utils/pipeline.py`
- If additional lint tooling is introduced later, prefer repo-configured commands over ad-hoc defaults.

## Test commands

- Run full test suite: `uv run pytest`
- Run one test file: `uv run pytest tests/test_llm_extraction.py`
- Run one test function: `uv run pytest tests/test_llm_extraction.py::test_build_prompt_contains_id_and_trial_fields`
- Run with keyword expression: `uv run pytest -k "publications and not slow"`
- Stop early after first failure: `uv run pytest -x`

## Execution model (pipeline)

### Config-driven recipes

- **`config.yaml`** — minimal shared infrastructure (database connections, path definitions, `workflow: null` placeholder)
- **`recipe/`** — workflow-specific YAML configs that extend the base via `+recipe=<name>`
- Run a recipe: `uv run clinical_mining +recipe=aact_llm_extractor`
- Combine recipes: `uv run clinical_mining +recipe=clinical_report_generation +recipe=aact_llm_extractor`

### Step definition format

Steps are defined as dict-keyed entries for stable CLI overrides:

```yaml
workflow:
  transform:
    generate:
      my_step:                        # step name (becomes data_store key)
        function: clinical_mining...  # full Python path
        parameters:
          input: $previous_step       # reference another step's output
          literal_value: 42           # literal values passed as-is
```

- Override from CLI: `workflow.transform.generate.my_step.parameters.param=value`
- The path follows: `workflow.transform.<section>.<step_name>.parameters.<param>`

### Output convention

- Any step whose name starts with `output_` is automatically persisted:
  - **Polars DataFrame** → written as Parquet to `${datasets.output_path}/<date>/<name>.parquet`
  - **Dict** → written as JSON to `${datasets.output_path}/<date>/<name>.json`
  - **None** → skipped (used for inspect/debug modes)

### Inspect mode (single-trial)

- When `filter_by_id` is used with a specific `id_value`, the pipeline produces a single prompt.
- The generic LLM engine auto-detects single-prompt runs and enters inspect mode: prints the prompt + result to stdout, returns `None` (CLI skips output writing).
- No `inspect_mode` flag needed — it's implicit from having exactly one prompt.

## LLM extraction engine

### Architecture

- **Generic engine** (`workflows/llm.py`): accepts pre-built prompts (`list[{"id","prompt"}]`) and any Pydantic model class for response validation. Domain-agnostic.
- **Domain-specific helpers** (`data_sources/aact/llm_extractor.py`): `filter_by_id`, `build_prompts`, `build_prompt`, `fetch_publications`.
- **Europe PMC integration** (`data_sources/europepmc.py`): `fetch_publications`, `build_publications_map` for batch-fetching abstracts by PMID.

### LLM workflow steps (typical recipe)

1. `clinical_report` — extract clinical trial data from AACT
2. `filtered_report` — filter to specific trial(s) via `filter_by_id`
3. `publications_map` — fetch Europe PMC abstracts for trial PMIDs (`enabled: true/false`)
4. `prompts` — build prompts per trial with `build_prompts(report, trial_fields, publications_map)`
5. `output_llm_extraction` — run generic LLM extraction with `run_extraction`

### Key parameters

- `model_class`: dotted path to Pydantic model (e.g., `clinical_mining.schemas.ClinicalReportExtractionSchema`)
- `system_prompt_path`: path to system prompt text file
- `model`: OpenAI model identifier
- `sample_size` (in upstream sampling step): number of trials to sample before prompt/publication generation (`None` = all)
- `concurrency`: max parallel API calls

### Adding a new data source for LLM extraction

1. Create helpers in `data_sources/<source>/llm_extractor.py` with `filter_by_id`, `build_prompts`, `build_prompt`.
2. Wire steps in a new recipe under `recipe/`.
3. Reuse the generic `workflows/llm.run_extraction` engine — no changes needed there.

## Code style conventions to follow

### Imports

- Prefer absolute imports from `clinical_mining...` for internal modules.
- Group imports in this order when practical:
  1. standard library
  2. third-party
  3. local package imports
- Prefer explicit symbol imports for heavily used schemas/constants.

### Formatting

- Follow PEP 8 (4-space indentation, readable line lengths).
- Keep Polars transformations readable with one operation per line in long chains.
- Favor multiline expression chaining for non-trivial pipelines.
- Keep comments sparse and focused on non-obvious logic.

### Typing

- Use Python 3.11+ typing syntax (`str | None`, `list[str]`, `dict[str, Any]`).
- Add type hints for public functions/methods.
- Keep return types explicit for transformation functions (`pl.DataFrame`, `ClinicalReport`, etc.).
- Use schema models/contracts (`pydantic.BaseModel`) for externally meaningful outputs.

### Naming

- Functions and variables: `snake_case`.
- Classes/enums: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Output/data boundary fields generally use camelCase (see `snake_to_camel` and schema fields).

### DataFrame patterns

- Prefer Polars-native transforms over Python loops.
- Use `pl.struct(...)` for nested drug/disease fields before aggregation.
- Use `explode` + `group_by(...).agg(...)` to normalize/re-aggregate entities.
- Deduplicate with `unique()` after joins/expansions where needed.
- Use `pl.coalesce(...)` to preserve existing IDs while filling mapped IDs.

### Spark/OnToma boundaries

- Keep Spark usage focused to OnToma/NER-intensive steps.
- Use existing conversion helpers (`utils/polars_helpers.py`) for Polars<->Spark transitions.
- Preserve ID columns (`drugId`, `diseaseId`) during mapping joins and fallback logic.
- Respect NER cache paths to avoid unnecessary recomputation.

### Schema and validation

- Validate external-facing DataFrames with `validate_schema(...)`.
- Preserve required fields from `ClinicalReportSchema` and `ClinicalIndicationSchema`.
- If adding fields, maintain backward compatibility unless explicitly refactoring contracts.

### Error handling

- Raise explicit exceptions for invalid config or unsupported formats (`ValueError`, `KeyError`, `ImportError`).
- Fail early on missing optional dependencies (e.g., Oracle extras) with install guidance.
- Catch-and-continue only for intentional partial-recovery paths.
- Include contextual details in recoverable-failure logs (source table/path/step).

### Logging

- Use `loguru.logger` for operational logs.
- Keep logs concise and actionable for long-running steps.
- Log section boundaries, key counts, and mapping recovery stats when helpful.

## Config and pipeline edit guidance

- Prefer implementing new logic as callable functions, then wire them in recipe YAML files.
- Preserve DAG semantics and section ordering.
- Use `$name` parameter references rather than duplicating data loads.
- Keep step names stable when possible (downstream references depend on them).
- Legacy `scripts/` directory is being phased out; new logic belongs in `src/clinical_mining/`.

## Security and secrets

- Never commit real DB credentials or private connection strings.
- Treat `db_properties.*.user` and `db_properties.*.password` as secrets.
- Use env vars / Hydra overrides for credentials and machine-specific paths.

## Contribution checklist for agents

- Run: `uv run ty check src`
- Run tests: `uv run pytest`
- If behavior changes, update relevant docs (`README.md`, config comments, or both).
- Keep edits minimal and consistent with existing Polars/Spark pipeline patterns.
- Avoid unrelated refactors while implementing focused changes.
