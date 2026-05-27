# Clinical Trial Mining

## Motivation

Mining clinical trials is essential for accelerating drug discovery and biomedical research. ClinicalTrials.gov contains a wealth of information on ongoing and completed studies, including interventions, conditions, and outcomes. Systematic extraction and integration of this data enables:

- Mapping of drug–disease relationships
- Identification of drug repurposing opportunities
- Analysis of intervention efficacy and safety

## Project Overview

This project provides tools to fetch, process, and annotate clinical trial data directly from the AACT (Aggregate Analysis of ClinicalTrials.gov) database. It is designed to facilitate large-scale mining and integration of clinical trials for drug discovery applications.

### Key Features

- **Direct Connection to AACT:** Uses a robust connector to securely access the AACT PostgreSQL database.
- **Automated Table Loading:** Loads and filters relevant tables (studies, interventions, conditions, etc.) using Polars for scalable processing.
- **Drug and Disease Mapping:** Integrates external drug and disease vocabularies to annotate interventions and indications in trials.
- **LLM Extraction:** Uses LLMs to extract structured drug–disease evidence from clinical trial records.
- **Config-Driven Workflows:** All pipelines are defined via YAML recipes — no code changes needed to reconfigure.

## Data Sources

1. **AACT Database**  

    AACT database is a PostgreSQL database containing clinical trial data from the ClinicalTrials.gov database. We use Polars to connect to the database and return queried data in a DataFrame format. Credentials and connection parameters are provided via configuration.

2. **ChEMBL Drug Indication Data**

    JSON file storing indications for drugs, and clinical candidate drugs, from a variety of sources (e.g., FDA, EMA, WHO ATC, ClinicalTrials.gov, INN, USAN).

3. **ChEMBL Clinical Trials Pipeline**

   The private DRUGBASE_CURATION database in ChEMBL stores metadata related to clinical trials. After processing data from ClinicalTrials.gov, their internal pipeline automatically assigns an EFO ID to each condition and a ChEMBL ID to each intervention mentioned in the trials. This database is used to map the conditions and interventions in the AACT database to ChEMBL and EFO IDs, and it **requires a valid ChEMBL user account** to access.

4. **PMDA Database**

   The PMDA (Pharmaceuticals and Medical Devices Agency) is the Japanese regulatory agency. We use their PDF of approved products to extract drug/disease associations. The PDF can be downloaded from their site: https://www.pmda.go.jp/english/review-services/reviews/approved-information/drugs/0002.html

## Usage

### Configuration

The project uses a **base config + recipes** pattern:

- **`config.yaml`** — minimal shared infrastructure (database connections, path definitions)
- **`recipe/`** — workflow-specific configurations that extend the base

Run `uv run clinical_mining --help` to see available recipes.

### Workflows

#### 1. Clinical Report Generation

Loads all data sources (AACT, ChEMBL, TTD, EMA, PMDA), generates clinical reports, maps entities to ChEMBL/EFO IDs, and produces the final clinical indication dataset.

```bash
uv run clinical_mining +recipe=clinical_report_generation
```

#### 2. LLM Extraction

Loads clinical trial data from AACT and uses an LLM (via OpenAI) to extract structured drug–disease evidence including drug intent, primary indications, investigated drugs, comparators, and supportive medications.

```bash
# Batch extraction with defaults
uv run clinical_mining +recipe=aact_llm_extractor

# Single-trial inspect mode (prints to stdout, no output written)
uv run clinical_mining +recipe=aact_llm_extractor \
  workflow.transform.generate.filtered_report.parameters.id_value=NCT00002742

# Override any config value
uv run clinical_mining +recipe=aact_llm_extractor \
  workflow.transform.generate.output_llm_extraction.parameters.model=gpt-4o

# Enable publications enrichment
uv run clinical_mining +recipe=aact_llm_extractor \
  workflow.transform.generate.publications_map.parameters.enabled=true
```
### Configuring a Workflow Step

Each step within a workflow is defined as a dictionary keyed by name:

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

To override step parameters from the command line, use the full path:

```bash
uv run clinical_mining +recipe=aact_llm_extractor \
  workflow.transform.generate.filtered_report.parameters.id_value=NCT00002742
```

The path follows the structure: `workflow.transform.<section>.<step_name>.parameters.<param>`

### Output

Any step whose name starts with `output_` is automatically persisted:

- **Polars DataFrame** → written as Parquet to `${datasets.output_path}/<date>/<name>.parquet`
- **Dict** → written as JSON to `${datasets.output_path}/<date>/<name>.json`
- **None** → skipped (used for inspect/debug modes)

### Environment Variables

| Variable | Required for |
|---|---|
| `AACT_USER` | AACT database access (optional for localhost) |
| `AACT_PASSWORD` | AACT database access (optional for localhost) |
| `OPENAI_API_KEY` | LLM extraction workflow |

### Contribute

If you need to add a new data transformation or a new input source, please submit a pull request with the new functionality. Once merged, your function can be integrated into the pipeline via the configuration file.
