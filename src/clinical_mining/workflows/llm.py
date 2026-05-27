"""Generic LLM extraction engine.

Accepts a list of pre-built prompts and any Pydantic model for response validation.
Returns a Polars DataFrame of extractions (for parquet output) or None (inspect mode).

Flex processing (service_tier="flex") is enabled by default for ~50% cost
reduction; requests automatically fall back to "auto" on capacity shortage.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import traceback
from importlib import import_module
from pathlib import Path

import polars as pl
from loguru import logger
from openai import AsyncOpenAI, APIStatusError
from pydantic import BaseModel

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def run_extraction(
    prompts: list[dict],
    model_class: str,
    system_prompt_path: str,
    model: str,
    openai_key: str,
    service_tier: str = "default",
    concurrency: int = 50,
    max_retries: int = 2,
    errors_dir: str | Path | None = None,
) -> pl.DataFrame | None:
    """Extract structured information from pre-built prompts using an LLM.

    Args:
        prompts: List of dicts with keys "id" and "prompt".
        model_class: Dotted path to a Pydantic model class for response
            validation (e.g. "mypackage.models.TrialExtraction").
        system_prompt_path: Path to the system prompt text file.
        model: OpenAI model identifier. Must support the Responses API and,
            if service_tier="flex", flex processing (see OpenAI pricing page).
        openai_key: OpenAI API key.
        service_tier: OpenAI service tier. "flex" cuts costs by ~50% at the
            expense of higher latency; requests fall back automatically to
            "auto" if flex capacity is unavailable. Not available for all models, pass "auto" or "default"
            to opt out.
        concurrency: Maximum number of parallel API calls. Tune against your
            tier's rate limits to avoid sustained 429s.
        max_retries: Retry attempts for transient SDK-level errors (408, 429
            rate-limit, 5xx). Passed directly to the SDK client.
        errors_dir: Directory for errors.jsonl. Defaults to the working
            directory.

    Returns:
        Polars DataFrame with one row per successful extraction, or None in
        single-prompt inspect mode.
    """
    if not openai_key:
        raise ValueError("openai_key must be a non-empty string.")

    model_cls = _import_class(model_class)
    system_prompt = Path(system_prompt_path).read_text()

    timeout = (
        900 if service_tier == "flex" else 600
    )  # 15 minutes for flex, 10 minutes for auto as per OpenAI recommendations
    client = AsyncOpenAI(api_key=openai_key, timeout=timeout, max_retries=max_retries)

    # ── Inspect mode: single prompt ───────────────────────────────────────────
    if len(prompts) == 1:
        entry = prompts[0]
        logger.info("\n── Prompt sent to model ─────────────────────────────────────")
        logger.info(entry["prompt"])
        logger.info("─────────────────────────────────────────────────────────────\n")

        extractions, errors = asyncio.run(
            _run_async(
                prompts,
                model_cls,
                system_prompt,
                model,
                client,
                concurrency,
                service_tier,
            )
        )
        if errors:
            logger.error("Error: %s", errors[0]["error"])
            raise RuntimeError(errors[0]["error"])

        logger.info(
            json.dumps(
                json.loads(extractions[0].model_dump_json(exclude_none=True)),
                indent=2,
            )
        )
        return None

    # ── Batch mode ────────────────────────────────────────────────────────────
    logger.info(
        "Running extraction: model=%s  concurrency=%d  service_tier=%s",
        model,
        concurrency,
        service_tier,
    )
    extractions, errors = asyncio.run(
        _run_async(
            prompts, model_cls, system_prompt, model, client, concurrency, service_tier
        )
    )
    logger.info(
        "Extraction complete: %d succeeded, %d failed.", len(extractions), len(errors)
    )

    if errors:
        error_path = (
            Path(errors_dir) / "errors.jsonl" if errors_dir else Path("errors.jsonl")
        )
        error_path.write_text("\n".join(json.dumps(e) for e in errors) + "\n")

        for err in errors[:5]:
            logger.error(
                "Failed  id=%-30s  type=%s  message=%s",
                err.get("id"),
                err.get("error_type"),
                err.get("error_message"),
            )
        if len(errors) > 5:
            logger.error(
                "... plus %d more errors — see %s", len(errors) - 5, error_path
            )

    return _extractions_to_df(extractions)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _import_class(dotted_path: str) -> type[BaseModel]:
    """Dynamically import a Pydantic model class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)


def _patch_schema(schema: dict) -> dict:
    """Patch a JSON schema for OpenAI structured-output compatibility.

    OpenAI's structured-output mode requires every object node to declare
    ``additionalProperties: false`` and enumerate all properties under
    ``required``. Pydantic's ``model_json_schema`` does not emit these by
    default, so we add them in a single recursive pass.
    """
    schema = copy.deepcopy(schema)

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema


async def _run_async(
    prompts: list[dict],
    model_cls: type[BaseModel],
    system_prompt: str,
    model: str,
    client: AsyncOpenAI,
    concurrency: int,
    service_tier: str,
) -> tuple[list[BaseModel], list[dict]]:
    semaphore = asyncio.Semaphore(concurrency)
    schema = _patch_schema(model_cls.model_json_schema(by_alias=True))

    tasks = [
        _extract_record(
            semaphore,
            client,
            entry,
            model_cls,
            system_prompt,
            model,
            schema,
            service_tier,
        )
        for entry in prompts
    ]
    results = await asyncio.gather(*tasks)

    extractions = [r for r, _ in results if r is not None]
    errors = [e for _, e in results if e is not None]
    return extractions, errors


async def _extract_record(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    entry: dict,
    model_cls: type[BaseModel],
    system_prompt: str,
    model: str,
    schema: dict,
    service_tier: str,
) -> tuple[BaseModel | None, dict | None]:
    """Call the Responses API for a single record.

    Transient errors (408 timeout, rate-limit 429, 5xx) are retried
    automatically by OpenAI.
    One custom layer is added on top: when ``service_tier="flex"`` and the API
    returns a ``resource_unavailable`` 429 (capacity shortage, not a rate
    limit), the request is retried once under ``service_tier="auto"`` so it can
    still complete, at standard cost, rather than failing permanently.
    """
    # If flex is requested, try flex first then fall back to auto on capacity
    # shortage. For any other tier, there is only one attempt.
    tiers_to_try = ["flex", "auto"] if service_tier == "flex" else [service_tier]

    for tier in tiers_to_try:
        try:
            async with semaphore:
                response = await client.responses.create(
                    model=model,
                    instructions=system_prompt,
                    input=entry["prompt"],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": model_cls.__name__,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    service_tier=tier,
                    store=False,  # extractions are stateless; no persistence needed
                )

            extraction = model_cls.model_validate_json(response.output_text)
            if hasattr(extraction, "id"):
                extraction.id = entry["id"]
            return extraction, None

        except APIStatusError as e:
            # Distinguish flex capacity shortage from ordinary rate-limit 429s.
            # Only the former warrants a tier fallback; the latter is already
            # retried by the SDK.
            is_flex_unavailable = e.status_code == 429 and "resource_unavailable" in (
                e.body or {}
            ).get("error", {}).get("code", "")
            if is_flex_unavailable and tier != tiers_to_try[-1]:
                logger.warning(
                    "Flex capacity unavailable for id=%s — retrying with standard processing.",
                    entry["id"],
                )
                continue  # move to "auto"

            return None, {
                "id": entry["id"],
                "error": str(e),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": e.status_code,
            }

        except Exception as e:
            return None, {
                "id": entry["id"],
                "error": str(e),
                "error_type": type(e).__name__,
                "error_message": str(e) or repr(e),
                "error_traceback": traceback.format_exc(),
            }

    # Unreachable in normal flow; guards against future loop changes.
    return None, {
        "id": entry["id"],
        "error": "all service tiers exhausted without a result",
        "error_type": "ServiceTierExhausted",
        "error_message": "Request failed on both flex and auto tiers.",
    }


def _extractions_to_df(extractions: list[BaseModel]) -> pl.DataFrame:
    """Serialise a list of Pydantic model instances into a Polars DataFrame.

    Nested structures (lists, dicts) are JSON-encoded to preserve
    Parquet round-trip compatibility.
    """
    rows = []
    for ext in extractions:
        row = ext.model_dump(exclude_none=True)
        row = {
            k: json.dumps(v) if isinstance(v, (list, dict)) else v
            for k, v in row.items()
        }
        rows.append(row)
    return pl.DataFrame(rows)



def write_batch_files(
    prompts: list[dict],
    system_prompt_path: str,
    model_class: str,
    out_dir: Path,
    batch_size: int,
    service_tier: str,
    model: str = "gpt-4.1-mini",
) -> None:
    """Prepare batches of JSON lines to submit to the OpenAI Batch API.
    
    Args:
        prompts (list[dict]): Preformed prompts with the query (e.g., the output of `data_sources.aact.llm_extractor.build_prompts`)
        system_prompt_path (str): Path to the system prompt file
        model_class (str): Pydantic class with the output schema (e.g., 'clinical_mining.schemas.ClinicalReportExtraction')
        out_dir (Path): Directory to write batch files
        batch_size (int): Number of requests per batch file
        service_tier (str): Service tier for the OpenAI Batch API (e.g., 'flex', 'auto')
        model (str): Model to use for the OpenAI Batch API (default: 'gpt-4.1-mini')
    """
    def _iter_chunks(items: list[dict], chunk_size: int):
        for i in range(0, len(items), chunk_size):
            yield i // chunk_size, items[i : i + chunk_size]

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

    for idx, chunk in _iter_chunks(prompts, batch_size):  # TODO: make it GCSFS friendly
        out_file = out_dir / f"responses_batch_{idx:04d}.jsonl"
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

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Wrote {} requests into {} batch files under {}",
        len(prompts),
        len(manifest["files"]),
        out_dir,
    )
