"""Domain-specific LLM extraction helpers for AACT clinical trials."""

from __future__ import annotations

import json

import fsspec
import polars as pl
from loguru import logger

from clinical_mining.data_sources.pubmed import build_publications_map
from clinical_mining.dataset import ClinicalReportExtraction
from clinical_mining.schemas import (
    ClinicalReportExtractionSchema,
    ExtractedDisease,
    ExtractedDrug,
)
from clinical_mining.utils.text_cleaning import sanitise_nested
from clinical_mining.workflows.llm import _extractions_to_df


def filter_by_id(report: pl.DataFrame, id_value: str | None) -> pl.DataFrame:
    """Filter report by id column. Returns full report if id_value is None."""
    if id_value is None:
        return report
    return report.filter(pl.col("id") == id_value.lower())


def sample_report(
    report: pl.DataFrame, sample_size: int | None, seed: int = 42
) -> pl.DataFrame:
    """Sample trials before expensive downstream steps.

    Returns full report when sample_size is None or >= row count.
    """
    if sample_size is None or sample_size >= report.height:
        return report
    return report.sample(n=sample_size, seed=seed, shuffle=True)


def build_prompts(
    report: pl.DataFrame,
    trial_fields: dict,
    publications_map: dict | None = None,
) -> list[dict]:
    """Build prompts for each record in the report.

    Returns a list of dicts: [{"id": ..., "prompt": ...}, ...]
    """
    records = report.to_dicts()
    results = []
    for row in records:
        pubs = publications_map.get(row["id"]) if publications_map else None
        prompt = build_prompt(row, trial_fields=trial_fields, publications=pubs)
        results.append({"id": row["id"], "prompt": prompt})
    return results


def build_prompt(
    row: dict,
    trial_fields: dict,
    publications: list[dict] | None = None,
) -> str:
    """Build prompt from a single clinical trial record."""
    lines = [f"Trial ID: {row['id']}"]
    for field, label in trial_fields.items():
        value = row.get(field)
        lines.append(f"{label}: {value if value is not None else 'null'}")
    drugs = [
        d["drugFromSource"] for d in (row.get("drugs") or []) if d.get("drugFromSource")
    ]
    if drugs:
        lines.append(f"Interventions: {'; '.join(drugs)}")
    if publications:
        lines.append("\nPublications:")
        for i, pub in enumerate(publications, start=1):
            title = pub.get("title") or "No title"
            abstract = pub.get("abstractText") or "No abstract"
            lines.append(f"[{i}] Title: {title}")
            lines.append(f"    Abstract: {abstract}")
    return "\n".join(lines)


def fetch_publications(
    report: pl.DataFrame,
    max_publications: int,
    enabled: bool,
) -> dict | None:
    """Fetch Europe PMC abstracts for trials in the report."""
    if not enabled:
        return None

    records = report.to_dicts()
    pubs = build_publications_map(records, max_pubs=max_publications)
    return pubs


_DISEASE_STRUCT = pl.Struct(
    {
        "name": pl.String,
        "severity": pl.String,
        "stage": pl.String,
        "onset": pl.String,
        "etiology": pl.String,
        "evidence_quote": pl.String,
    }
)

_DRUG_STRUCT = pl.Struct(
    {
        "drug": pl.String,
        "route": pl.String,
        "formulation": pl.String,
        "synonyms": pl.List(pl.String),
        "dosages": pl.List(pl.String),
        "evidence_quote": pl.String,
    }
)

EXTRACTION_SCHEMA = {
    "id": pl.String,
    "drug_intent": pl.String,
    "drug_intent_confidence": pl.Float64,
    "primary_indications": pl.List(_DISEASE_STRUCT),
    "background_conditions": pl.List(_DISEASE_STRUCT),
    "investigated_drugs": pl.List(_DRUG_STRUCT),
    "comparator_drugs": pl.List(_DRUG_STRUCT),
    "supportive_drugs": pl.List(_DRUG_STRUCT),
    "conclusion": pl.String,
}


def _normalise_disease(raw: dict) -> dict:
    """Fill every ExtractedDisease field, defaulting missing ones to None."""
    return {
        "name": raw.get("name"),
        "severity": raw.get("severity"),
        "stage": raw.get("stage"),
        "onset": raw.get("onset"),
        "etiology": raw.get("etiology"),
        "evidence_quote": raw.get("evidence_quote"),
    }


def _normalise_drug(raw: dict) -> dict:
    """Fill every ExtractedDrug field, defaulting missing ones to None."""
    return {
        "drug": raw.get("drug"),
        "route": raw.get("route"),
        "formulation": raw.get("formulation"),
        "synonyms": raw.get("synonyms"),  # already list[str] | None
        "dosages": raw.get("dosages"),  # already list[str] | None
        "evidence_quote": raw.get("evidence_quote"),
    }


def _normalise_disease_list(items: list | None) -> list[dict]:
    if not items:
        return []
    return [_normalise_disease(x) for x in items if isinstance(x, dict)]


def _normalise_drug_list(items: list | None) -> list[dict]:
    if not items:
        return []
    return [_normalise_drug(x) for x in items if isinstance(x, dict)]


def _message_texts(outer: dict) -> list[str]:
    """Collect every text payload carried by a batch envelope.

    The batch API can split one response across several ``output`` items, a
    truncated fragment followed by a continuation, so ``output[0]`` is not
    reliably the complete payload. Every ``message`` item is walked, in order,
    and every text content entry is returned as a candidate.
    """
    return [
        content["text"]
        for item in outer["response"]["body"]["output"]
        if item.get("type") == "message"
        for content in (item.get("content") or [])
        if "text" in content
    ]


def _parse_single_record(
    outer: dict, path: str = "<test>", row_idx: int = 0
) -> tuple[ClinicalReportExtractionSchema | None, dict | None]:
    """Parse a single decoded JSONL envelope into a typed extraction object.

    The first candidate payload that decodes to a JSON object wins; a record is
    only rejected when none of them does.

    Returns (ClinicalReportExtractionSchema, None) on success,
    (None, bad_record_dict) on failure.
    """
    record_id = outer.get("custom_id", "")

    try:
        texts = _message_texts(outer)
        if not texts:
            raise ValueError("no message output with text content")
    except Exception as e:
        return None, {
            "file": path,
            "row_idx": row_idx,
            "id": record_id,
            "error": f"missing_text_path: {e}",
        }

    payload = None
    last_error: Exception | None = None
    for text in texts:
        try:
            candidate = sanitise_nested(json.loads(text))
        except json.JSONDecodeError as e:
            last_error = e
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
        last_error = ValueError(
            f"expected a JSON object, got {type(candidate).__name__}"
        )

    if payload is None:
        return None, {
            "file": path,
            "row_idx": row_idx,
            "id": record_id,
            "error": f"inner_json_error: {last_error}",
        }

    try:
        extraction = ClinicalReportExtractionSchema(
            id=record_id,
            drug_intent=payload.get("drug_intent"),
            drug_intent_confidence=payload.get("drug_intent_confidence"),
            primary_indications=[
                ExtractedDisease(**d)
                for d in _normalise_disease_list(payload.get("primary_indications"))
                if d.get("name")
            ],
            background_conditions=[
                ExtractedDisease(**d)
                for d in _normalise_disease_list(payload.get("background_conditions"))
                if d.get("name")
            ]
            or None,
            investigated_drugs=[
                ExtractedDrug(**d)
                for d in _normalise_drug_list(payload.get("investigated_drugs"))
                if d.get("drug")
            ],
            comparator_drugs=[
                ExtractedDrug(**d)
                for d in _normalise_drug_list(payload.get("comparator_drugs"))
                if d.get("drug")
            ]
            or None,
            supportive_drugs=[
                ExtractedDrug(**d)
                for d in _normalise_drug_list(payload.get("supportive_drugs"))
                if d.get("drug")
            ]
            or None,
            conclusion=payload.get("conclusion"),
        )
    except Exception as e:
        return None, {
            "file": path,
            "row_idx": row_idx,
            "id": record_id,
            "error": f"model_validation_error: {e}",
        }

    return extraction, None


def parse_batch_results(output_dir: str) -> ClinicalReportExtraction:
    """Read LLM extraction output files and return a validated dataset.

    Returns an empty dataset when no valid records are found.
    """

    fs, root = fsspec.core.url_to_fs(output_dir)
    all_paths = fs.find(root)
    output_files = sorted(p for p in all_paths if p.endswith("_output.jsonl"))
    if not output_files:
        raise ValueError(f"No *_output.jsonl files found under: {output_dir}")

    good_records: list[ClinicalReportExtractionSchema] = []
    bad_records: list[dict] = []
    total_rows = 0

    for path in output_files:
        with fs.open(path, "rt", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total_rows += 1

                try:
                    outer = json.loads(line)
                except json.JSONDecodeError as e:
                    bad_records.append(
                        {
                            "file": path,
                            "row_idx": row_idx,
                            "id": None,
                            "error": f"outer_json_error: {e}",
                        }
                    )
                    continue

                good, bad = _parse_single_record(outer, path=path, row_idx=row_idx)
                if good:
                    good_records.append(good)
                if bad:
                    bad_records.append(bad)

    if bad_records:
        logger.warning(
            "Dropped {} malformed rows while parsing batch outputs from {}",
            len(bad_records),
            output_dir,
        )
        logger.warning("Sample malformed rows: {}", bad_records[:5])

    logger.info(
        "Parsed {} rows from {} output files (good={}, bad={})",
        total_rows,
        len(output_files),
        len(good_records),
        len(bad_records),
    )

    return ClinicalReportExtraction(_extractions_to_df(good_records))
