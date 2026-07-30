#!/usr/bin/env python3
"""Run LLM extraction via the OpenAI Batch API.

Generates prompts from clinical-trial abstracts, writes batch JSONL files,
uploads them to OpenAI, creates a batch job, polls for completion, downloads
the results, and parses them into a validated dataset.

Usage:
    export OPENAI_API_KEY="sk-..."
    uv run python scripts/run_openai_batch.py --data-path full_llm_extraction_dataset.parquet --batch-output ./batch_output --final-output ./extracted_results.parquet

    # Quick fire-and-forget: upload all files + create all batch jobs, then exit:
    uv run python scripts/run_openai_batch.py --fire-only

    # Resume to collect results (after --fire-only or if interrupted):
    uv run python scripts/run_openai_batch.py --resume

    # Test with just the first batch file:
    uv run python scripts/run_openai_batch.py --max-batches 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from loguru import logger
from openai import OpenAI

from clinical_mining.data_sources.aact.llm_extractor import (
    build_prompts_nct_combined_abstracts,
    parse_batch_results,
)
from clinical_mining.workflows.llm import write_batch_files

MODEL = "gpt-4.1-mini"
MODEL_CLASS = "clinical_mining.schemas.TrialExtraction"
SYSTEM_PROMPT_PATH = "src/clinical_mining/prompts/outcome_analysis_llm.txt"
SERVICE_TIER = "auto"
BATCH_SIZE = 3500
POLL_INTERVAL = 30


def load_parquet(data_path: str) -> pl.DataFrame:
    logger.info("Loading parquet: {}", data_path)
    return pl.read_parquet(data_path)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(state: dict, path: Path) -> None:
    path.write_text(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM extraction via OpenAI Batch API")
    parser.add_argument("--data-path", default="benchmark_version_2_augmented.parquet")
    parser.add_argument("--batch-output", default="./batch_output")
    parser.add_argument("--final-output", default="./extracted_results.parquet")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit processing to the first N batch files (default: all)",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Stop after generating batch files (no API calls)",
    )
    parser.add_argument(
        "--fire-only", action="store_true",
        help=(
            "Upload all files and create all batch jobs then exit without polling. "
            "Run later with --resume to collect results."
        ),
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_output)
    batch_dir.mkdir(parents=True, exist_ok=True)
    state_path = batch_dir / ".state.json"

    if args.resume:
        state = load_state(state_path)
        if not state.get("jobs"):
            logger.warning("No job state to resume — starting fresh.")
            state = {}
    else:
        state = {}

    try:
        # ── Steps 1-3: Data, prompts, batch files (skip on --resume) ─────
        if not args.resume:
            df = load_parquet(args.data_path)

            logger.info("Generating prompts ...")
            prompts = build_prompts_nct_combined_abstracts(df)
            logger.info("Generated {} prompts.", len(prompts))

            batch_files = sorted(batch_dir.glob("responses_batch_*.jsonl"))
            if not batch_files:
                logger.info("Writing batch files to {}", batch_dir)
                write_batch_files(
                    prompts=prompts,
                    system_prompt_path=SYSTEM_PROMPT_PATH,
                    model_class=MODEL_CLASS,
                    out_dir=batch_dir,
                    batch_size=BATCH_SIZE,
                    service_tier=SERVICE_TIER,
                    model=MODEL,
                )
            else:
                logger.info("Batch files already exist — skipping write.")

            del prompts

            if args.skip_upload:
                logger.info("--skip-upload set; done.")
                save_state(state, state_path)
                return

        # ── Step 4: Determine which batch files to process ──────────────
        all_jsonl_files = sorted(batch_dir.glob("responses_batch_*.jsonl"))
        if not all_jsonl_files:
            raise FileNotFoundError(
                f"No responses_batch_*.jsonl files found in {batch_dir}"
            )

        if args.max_batches is not None:
            jsonl_files = all_jsonl_files[: args.max_batches]
            logger.info(
                "Processing {} of {} batch files (--max-batches={})",
                len(jsonl_files), len(all_jsonl_files), args.max_batches,
            )
        else:
            jsonl_files = all_jsonl_files
            logger.info("Processing all {} batch files.", len(jsonl_files))

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # ── Build/update job entries in state ───────────────────────────
        if "jobs" not in state:
            state["jobs"] = []

        for fpath in jsonl_files:
            if not any(j["file_name"] == fpath.name for j in state["jobs"]):
                state["jobs"].append({"file_name": fpath.name})

        # ── Phase 1: Upload all files → create all batch jobs ────────────
        logger.info("Phase 1: Uploading files and creating batch jobs ...")

        for job in state["jobs"]:
            fpath = batch_dir / job["file_name"]
            if fpath not in jsonl_files:
                continue

            if "file_id" not in job:
                logger.info("Uploading {} ...", fpath.name)
                with fpath.open("rb") as fh:
                    upload = client.files.create(file=fh, purpose="batch")
                job["file_id"] = upload.id
                logger.info("  -> file id: {}", upload.id)
                save_state(state, state_path)
            else:
                logger.info(
                    "{} already uploaded (file_id={}) — skipping.",
                    fpath.name, job["file_id"],
                )

            if "batch_id" not in job:
                batch = client.batches.create(
                    input_file_id=job["file_id"],
                    endpoint="/v1/responses",
                    completion_window="24h",
                )
                job["batch_id"] = batch.id
                job["status"] = batch.status
                logger.info(
                    "Batch created: file={}  batch_id={}  status={}",
                    fpath.name, batch.id, batch.status,
                )
                save_state(state, state_path)
            else:
                logger.info(
                    "{} already has batch (batch_id={}) — skipping create.",
                    fpath.name, job["batch_id"],
                )

        if args.fire_only:
            logger.info(
                "--fire-only: All batch jobs created. "
                "Re-run with --resume later to collect results."
            )
            save_state(state, state_path)
            return

        # ── Phase 2: Poll incomplete batches concurrently, download ─────
        incomplete_jobs = [
            job for job in state["jobs"]
            if job.get("status") != "completed"
        ]

        if not incomplete_jobs:
            logger.info("All batches already completed — skipping poll.")
        else:
            logger.info(
                "Phase 2: Polling {} incomplete batches concurrently ...",
                len(incomplete_jobs),
            )

            def _poll_and_download(job: dict) -> None:
                fpath = batch_dir / job["file_name"]
                logger.info(
                    "Polling batch {} for {} ...",
                    job["batch_id"], fpath.name,
                )
                while True:
                    batch = client.batches.retrieve(job["batch_id"])
                    rc = batch.request_counts
                    logger.info(
                        "  {}  status={}  requests={{total={}, completed={}, failed={}}}",
                        fpath.name, batch.status, rc.total, rc.completed, rc.failed,
                    )

                    if batch.status == "completed":
                        if not batch.output_file_id:
                            raise RuntimeError(
                                f"Batch {job['batch_id']} completed "
                                "but no output_file_id."
                            )
                        job["status"] = "completed"
                        job["output_file_id"] = batch.output_file_id

                        output_path = batch_dir / f"{fpath.stem}_output.jsonl"
                        if not output_path.exists():
                            logger.info("Downloading {} ...", output_path.name)
                            content = client.files.content(batch.output_file_id)
                            output_path.write_bytes(content.read())
                            logger.info(
                                "Downloaded {} ({} bytes).",
                                output_path.name, output_path.stat().st_size,
                            )
                        else:
                            logger.info(
                                "{} already downloaded — skipping.",
                                output_path.name,
                            )
                        return

                    if batch.status in ("failed", "cancelled", "expired"):
                        detail = (
                            batch.errors.data if batch.errors else "unknown"
                        )
                        raise RuntimeError(
                            f"Batch {job['batch_id']} '{batch.status}': {detail}"
                        )

                    time.sleep(POLL_INTERVAL)

            with ThreadPoolExecutor(max_workers=len(incomplete_jobs)) as ex:
                futures = {
                    ex.submit(_poll_and_download, job): job
                    for job in incomplete_jobs
                }
                for future in as_completed(futures):
                    future.result()
                save_state(state, state_path)

        # ── Parse results ───────────────────────────────────────────────
        logger.info("Parsing batch results ...")
        dataset = parse_batch_results(str(batch_dir))
        result_df = dataset.df
        logger.info("Parsed {} records.", len(result_df))

        final_path = Path(args.final_output)
        result_df.write_parquet(final_path)
        logger.info("Wrote {} rows to {}", len(result_df), final_path)

        state["complete"] = True
        save_state(state, state_path)
        logger.info("Done.")

    except Exception:
        save_state(state, state_path)
        logger.error(
            "Script failed — state saved to {}. Re-run with --resume to continue.",
            state_path,
        )
        raise


if __name__ == "__main__":
    main()
