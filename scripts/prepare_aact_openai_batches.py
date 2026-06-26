from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import date
from importlib import import_module
from pathlib import Path

import polars as pl
from loguru import logger

from clinical_mining.data_sources.aact.clinical_report import extract_clinical_report
from clinical_mining.data_sources.aact.llm_extractor import (
    build_prompts,
    fetch_publications,
    filter_by_id,
    parse_batch_results,
    sample_report,
)
from clinical_mining.utils.db import construct_db_uri, load_db_table


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)


def _patch_schema(schema: dict) -> dict:
    schema = copy.deepcopy(schema)

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema


def _iter_chunks(items: list[dict], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield i // chunk_size, items[i : i + chunk_size]


def _write_batch_files(
    prompts: list[dict],
    system_prompt_path: str,
    model_class: str,
    out_dir: Path,
    batch_size: int,
    service_tier: str,
    model: str = "gpt-4.1-mini",
    date_prefix: str | None = None,
) -> None:
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
    model_cls = _import_class(model_class)
    schema = _patch_schema(model_cls.model_json_schema(by_alias=True))

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "model": model,
        "endpoint": "/v1/responses",
        "total_requests": len(prompts),
        "batch_size": batch_size,
        "files": [],
    }

    prefix = f"{date_prefix}_" if date_prefix else ""
    for idx, chunk in _iter_chunks(prompts, batch_size):
        out_file = out_dir / f"{prefix}responses_batch_{idx:04d}.jsonl"
        with out_file.open("w", encoding="utf-8") as handle:
            for entry in chunk:
                request_line = {
                    "custom_id": str(entry["id"]),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "instructions": system_prompt,
                        "input": entry["prompt"],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": model_cls.__name__,
                                "schema": schema,
                                "strict": True,
                            }
                        },
                        "service_tier": service_tier,
                        "store": False,
                    },
                }
                handle.write(json.dumps(request_line, ensure_ascii=False) + "\n")

        file_size = out_file.stat().st_size
        if file_size > 200 * 1024 * 1024:
            logger.warning(
                "{} exceeds 200MB Batch upload limit ({} bytes). Reduce --batch-size.",
                out_file,
                file_size,
            )

        manifest_files = manifest["files"]
        assert isinstance(manifest_files, list)
        manifest_files.append(
            {
                "file": out_file.name,
                "requests": len(chunk),
                "bytes": file_size,
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote {} requests into {} batch files under {}",
        len(prompts),
        len(manifest["files"]),
        out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load AACT data, build LLM prompts, fetch publications, and export "
            "OpenAI Batch API JSONL files for /v1/responses."
        )
    )
    parser.add_argument("--aact-user", default=os.getenv("AACT_USER", ""))
    parser.add_argument("--aact-password", default=os.getenv("AACT_PASSWORD", ""))
    parser.add_argument("--aact-uri", default="localhost:5432/aact")
    parser.add_argument("--aact-schema", default="ctgov")
    parser.add_argument("--aact-db-type", default="postgresql")
    parser.add_argument("--id-value", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-publications", type=int, default=1)
    parser.add_argument("--publications-enabled", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--out-dir", default="data/openai_batches")
    parser.add_argument(
        "--system-prompt-path", default="src/clinical_mining/prompts/aact_llm.txt"
    )
    parser.add_argument(
        "--model-class",
        default="clinical_mining.schemas.ClinicalReportExtractionSchema",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--service-tier", default="default", choices=["default", "auto", "flex"]
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Path to previous batch results directory. If provided, studies already "
            "present in these results are skipped when building prompts."
        ),
    )
    parser.add_argument(
        "--baseline-parquet",
        default=None,
        help=(
            "Path to a clinical report parquet file. When provided, the AACT report is "
            "filtered to only include IDs present in this file (source == 'AACT'). "
            "Use with --results-dir to target only studies that are in the baseline "
            "but missing from the results (e.g. the 8.5K gap)."
        ),
    )
    args = parser.parse_args()

    db_url = construct_db_uri(
        db_type=args.aact_db_type,
        db_uri=args.aact_uri,
        db_user=args.aact_user or None,
        db_password=args.aact_password or None,
    )
    logger.info("Loading AACT source tables from {}", args.aact_uri)

    studies = load_db_table(
        table_name="studies",
        db_url=db_url,
        db_schema=args.aact_schema,
        select_cols=["nct_id", "study_type", "phase", "official_title"],
    )
    interventions = load_db_table(
        table_name="interventions",
        db_url=db_url,
        db_schema=args.aact_schema,
        select_cols=["nct_id", "intervention_type", "name"],
    )
    conditions = load_db_table(
        table_name="conditions",
        db_url=db_url,
        db_schema=args.aact_schema,
        select_cols=["nct_id", "downcase_name"],
    )
    brief_summaries = load_db_table(
        table_name="brief_summaries",
        db_url=db_url,
        db_schema=args.aact_schema,
        select_cols=["nct_id", "description"],
    )
    detailed_descriptions = load_db_table(
        table_name="detailed_descriptions",
        db_url=db_url,
        db_schema=args.aact_schema,
        select_cols="nct_id, description as detailed_description",
    )
    study_references = load_db_table(
        table_name="study_references",
        db_url=db_url,
        db_schema=args.aact_schema,
        select_cols=["nct_id", "pmid", "reference_type"],
    )

    logger.info("Building clinical report")
    clinical_report = extract_clinical_report(
        studies=studies,
        interventions=interventions,
        conditions=conditions,
        additional_metadata=[brief_summaries, study_references, detailed_descriptions],
        aggregation_specs={"pmid": {"group_by": "nct_id", "alias": "literature"}},
    ).df

    if args.baseline_parquet is not None:
        baseline = pl.read_parquet(args.baseline_parquet)
        baseline_ids = (
            baseline.filter(pl.col("source") == "AACT")["id"]
            .str.to_lowercase()
            .to_list()
        )
        before = clinical_report.height
        clinical_report = clinical_report.filter(pl.col("id").is_in(baseline_ids))
        logger.info(
            "Filtered clinical report to {} baseline AACT IDs (was {})",
            clinical_report.height,
            before,
        )

    filtered = filter_by_id(report=clinical_report, id_value=args.id_value)
    sampled = sample_report(
        report=filtered, sample_size=args.sample_size, seed=args.seed
    )
    logger.info("Report rows after filter/sample: {}", sampled.height)

    if args.results_dir is not None:
        processed = parse_batch_results(args.results_dir)
        processed_ids = processed["id"].str.to_lowercase().unique().to_list()
        before = sampled.height
        sampled = sampled.filter(~pl.col("id").is_in(processed_ids))
        logger.info(
            "Skipping {} already-processed studies ({} remaining)",
            before - sampled.height,
            sampled.height,
        )
        if sampled.height == 0:
            logger.info("No remaining studies to process. Exiting.")
            return

    publications_map = fetch_publications(
        report=sampled,
        max_publications=args.max_publications,
        enabled=args.publications_enabled,
    )

    prompts = build_prompts(
        report=sampled,
        trial_fields={
            "trialOfficialTitle": "Official Title",
            "trialDescription": "Description",
            "trialDetailedDescription": "Detailed Description",
        },
        publications_map=publications_map,
    )
    logger.info("Built {} prompts", len(prompts))

    _write_batch_files(
        prompts=prompts,
        system_prompt_path=args.system_prompt_path,
        model_class=args.model_class,
        model=args.model,
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
        service_tier=args.service_tier,
        date_prefix=date.today().isoformat(),
    )


if __name__ == "__main__":
    main()
