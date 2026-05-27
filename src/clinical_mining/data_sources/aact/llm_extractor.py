"""Domain-specific LLM extraction helpers for AACT clinical trials."""

from __future__ import annotations

import polars as pl
from loguru import logger

from clinical_mining.data_sources.pubmed import build_publications_map


def filter_by_id(report: pl.DataFrame, id_value: str | None) -> pl.DataFrame:
    """Filter report by id column. Returns full report if id_value is None."""
    if id_value is None:
        return report
    return report.filter(pl.col("id") == id_value.lower())


def sample_report(report: pl.DataFrame, sample_size: int | None, seed: int = 42) -> pl.DataFrame:
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
    drugs = [d["drugFromSource"] for d in (row.get("drugs") or []) if d.get("drugFromSource")]
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

def parse_indication_batch_results(output_dir: str) -> pl.DataFrame:
    """Reads LLM extraction output files and returns extracted trial disease/drug data.
    
    The function reads each json file and parses the JSON line by line, extracting
    the trial ID, disease, and drug information.

    The JSON might be malformed, so we log the errors into a bad records dataframe and continue.

    Columns:
    - id: Trial ID
    - diseases: List of unique `primary_indications`
    - drugs: List of unique `investigated_drugs` and `supportive_drugs`
    """
    import json
    import fsspec

    # Recursive listing with fsspec to work with local dirs and GCS paths
    fs, root = fsspec.core.url_to_fs(output_dir)
    all_paths = fs.find(root)
    output_files = sorted(p for p in all_paths if p.endswith("_output.jsonl"))
    if not output_files:
        raise ValueError(f"No *_output.jsonl files found under: {output_dir}")
    good_records = []
    bad_records = []
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
                    bad_records.append({
                        "file": path,
                        "row_idx": row_idx,
                        "id": None,
                        "error": f"outer_json_error: {e}",
                    })
                    continue
                record_id = outer.get("custom_id")
                try:
                    text = outer["response"]["body"]["output"][0]["content"][0]["text"]
                except Exception as e:
                    bad_records.append({
                        "file": path,
                        "row_idx": row_idx,
                        "id": record_id,
                        "error": f"missing_text_path: {e}",
                    })
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as e:
                    bad_records.append({
                        "file": path,
                        "row_idx": row_idx,
                        "id": record_id,
                        "error": f"inner_json_error: {e}",
                    })
                    continue
                diseases = [
                    x.get("name")
                    for x in (payload.get("primary_indications") or [])
                    if isinstance(x, dict) and x.get("name")
                ]
                investigated = [
                    x.get("drug")
                    for x in (payload.get("investigated_drugs") or [])
                    if isinstance(x, dict) and x.get("drug")
                ]
                supportive = [
                    x.get("drug")
                    for x in (payload.get("supportive_drugs") or [])
                    if isinstance(x, dict) and x.get("drug")
                ]
                seen = set()
                drugs = []
                for d in investigated + supportive:
                    if d not in seen:
                        seen.add(d)
                        drugs.append(d)
                good_records.append({
                    "id": record_id,
                    "diseases": diseases,
                    "drugs": drugs,
                })

    if bad_records:
        logger.warning(
            "Dropped {} malformed rows while parsing batch outputs from {}",
            len(bad_records),
            output_dir,
        )
        logger.warning("Sample malformed rows (up to 10): {}", bad_records[:10])

    logger.info(
        "Parsed {} rows from {} output files (good={}, bad={})",
        total_rows,
        len(output_files),
        len(good_records),
        len(bad_records),
    )

    return pl.DataFrame(good_records)
