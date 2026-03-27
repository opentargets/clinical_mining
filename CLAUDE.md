# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run clinical_mining

# Run with credential overrides
uv run clinical_mining db_properties.user=<user> db_properties.password=<password>
```

No test runner or linter is configured in pyproject.toml.

## Architecture Overview

`clinical_mining` is a configuration-driven ETL pipeline that extracts drug-disease relationships from multiple clinical data sources, harmonizes them into a common schema, and outputs a Parquet dataset of clinical indications with mapped entity IDs.

### Entry Point & Configuration

`cli.py` is the entry point (via Hydra). `config.yaml` defines the entire pipeline declaratively — data source locations, database credentials, and the ordered list of processing steps. The pipeline runs three sequential stages: `setup → generate → post_process`.

Each pipeline step references a Python function by dotted path and passes named parameters, where `$name` resolves to a previously computed step's output stored in a shared data dictionary. This makes the pipeline a lightweight DAG without explicit dependency graph code.

### Data Sources (`data_sources/`)

Six heterogeneous sources are extracted into a common `ClinicalReport` schema:
- **AACT** (`aact.py`) — ClinicalTrials.gov via PostgreSQL
- **ChEMBL** (`chembl/`) — Curated drug indications, warnings, and direct curation joins via PostgreSQL
- **PMDA** (`pmda.py`) — Japanese regulatory approvals parsed from PDFs via `pdfplumber`
- **EMA** (`ema.py`) — European Medicines Agency data
- **TTD** (`ttd.py`) — Therapeutic Target Database

### Schema & Stage Harmonization (`schemas.py`)

The pipeline normalizes 50+ phase variants from different sources into 13 canonical `ClinicalStage` values (e.g., `PHASE_3`, `APPROVAL`, `WITHDRAWAL`) with a numeric ranking for computing `maxClinicalStage`. The final output entity is `ClinicalIndication`, which groups evidence by drug-disease pair and records `mappingStatus` (`FULLY_MAPPED`, `DRUG_MAPPED`, `DISEASE_MAPPED`, `UNMAPPED`).

### Entity Mapping (`utils/mapping.py`)

Drug and disease names are mapped to ChEMBL IDs and EFO IDs via two complementary strategies:
1. **Direct ChEMBL curation join** — fast, high-confidence
2. **OnToma NER** — Spark NLP pipeline for extracting and resolving entity mentions; expensive and cached

The dual DataFrame model exists because Polars (used everywhere else) doesn't support SparkNLP. `utils/spark_helpers.py` manages the Spark session and chunked conversion from Polars.

### Dataset Layer (`dataset/`)

`clinical_report.py` and `clinical_indication.py` wrap raw source data into validated Polars DataFrames conforming to the Pydantic schemas in `schemas.py`. These are the primary aggregation and deduplication layer before output.
