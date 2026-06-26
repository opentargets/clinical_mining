#!/usr/bin/env python
"""Run extract_clinical_report for AACT using the configuration files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_mining.data_sources.aact.clinical_report import extract_clinical_report
from clinical_mining.utils.db import construct_db_uri, load_db_table
from clinical_mining.utils.polars_helpers import filter_df, rename_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AACT extract_clinical_report using base config and recipe."
    )
    parser.add_argument(
        "--recipe",
        default="src/clinical_mining/recipe/clinical_report_generation.yaml",
        help="Path to recipe YAML file.",
    )
    parser.add_argument(
        "--config",
        default="src/clinical_mining/config.yaml",
        help="Path to base config YAML file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of rows loaded from AACT studies table (for faster runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading configurations...")
    base_cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(base_cfg)
    recipe_cfg = OmegaConf.load(args.recipe)

    # 1. Setup AACT DB Connection
    aact_props = base_cfg.db_properties.aact
    db_url = construct_db_uri(
        db_type=str(aact_props.type),
        db_uri=str(aact_props.uri),
        db_user=aact_props.get("user") or None,
        db_password=aact_props.get("password") or None,
    )
    db_schema = str(aact_props.schema)
    logger.info(
        f"Connected to AACT database at: {aact_props.uri} (schema: {db_schema})"
    )

    inputs_cfg = recipe_cfg.get("inputs", {})

    # Helper to load a table from db
    def load_table(name: str, limit: int | None = None) -> pl.DataFrame:
        source = inputs_cfg[name]
        select_cols = list(source.select_cols)
        logger.info(f"Loading input: {name} (cols: {select_cols})...")
        return load_db_table(
            table_name=name,
            db_url=db_url,
            db_schema=db_schema,
            select_cols=select_cols,
            limit=limit,
        )

    # 2. Load required inputs
    studies = load_table("studies", limit=args.limit)
    interventions = load_table("interventions")
    conditions = load_table("conditions")
    brief_summaries = load_table("brief_summaries")
    detailed_descriptions = load_table("detailed_descriptions")
    study_references = load_table("study_references")
    designs = load_table("designs")
    sponsors = load_table("sponsors")

    # 3. Filter sponsors to lead sponsors and rename detailed descriptions
    logger.info("Filtering sponsors to lead_sponsors...")
    lead_sponsors = filter_df(sponsors, "lead_or_collaborator == 'lead'")

    logger.info("Renaming detailed descriptions...")
    detailed_descriptions_rename = rename_columns(
        detailed_descriptions, {"description": "detailed_description"}
    )

    # 4. Prepare parameters for extract_clinical_report
    aact_report_cfg = recipe_cfg.workflow.transform.generate.aact_report.parameters

    # Map the variables as configured in the recipe
    additional_metadata_mapping = {
        "$brief_summaries": brief_summaries,
        "$study_references": study_references,
        "$designs": designs,
        "$lead_sponsors": lead_sponsors,
        "$detailed_descriptions_rename": detailed_descriptions_rename,
    }

    additional_metadata = [
        additional_metadata_mapping[ref]
        for ref in aact_report_cfg.additional_metadata
        if ref in additional_metadata_mapping
    ]

    # Convert DictConfig to a normal dictionary for aggregation_specs
    aggregation_specs = OmegaConf.to_container(
        aact_report_cfg.aggregation_specs, resolve=True
    )

    logger.info("Running extract_clinical_report...")
    report = extract_clinical_report(
        studies=studies,
        interventions=interventions,
        conditions=conditions,
        additional_metadata=additional_metadata,
        aggregation_specs=aggregation_specs,
    )

    # 5. Output inspection
    df = report.df
    logger.info("Report generated successfully!")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")

    # Filter to show some rows that actually have sponsors to verify
    sample_df = df.filter(pl.col("trialSponsor").is_not_null()).head(10)
    if sample_df.height > 0:
        logger.info("Sample rows with trialSponsor:")
        print(sample_df)
    else:
        logger.info("No rows with trialSponsor found in this sample.")
        print(df.head(5))
    logger.info(sample_df.schema)
    df.write_parquet("aact_report.parquet")


if __name__ == "__main__":
    main()
