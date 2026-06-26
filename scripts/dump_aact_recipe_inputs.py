from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from clinical_mining.utils.db import construct_db_uri, load_db_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump AACT db_table inputs configured in a recipe to parquet files."
        )
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
        "--output-dir",
        default="data/inputs/aact/27052026",
        help="Directory where parquet files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recipe_cfg = OmegaConf.load(args.recipe)
    base_cfg = OmegaConf.load(args.config)

    aact_props = base_cfg.db_properties.aact
    db_url = construct_db_uri(
        db_type=str(aact_props.type),
        db_uri=str(aact_props.uri),
    )
    db_schema = str(aact_props.schema)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_cfg = recipe_cfg.get("inputs", {})

    dumped_tables = 0
    for table_name, source in inputs_cfg.items():
        if source.get("format") != "db_table":
            continue
        if source.get("db", "aact") != "aact":
            continue

        select_cols = list(source.get("select_cols", []))
        if not select_cols:
            logger.warning("Skipping {}: no select_cols configured", table_name)
            continue

        logger.info("Loading table '{}' with {} columns", table_name, len(select_cols))
        df = load_db_table(
            table_name=table_name,
            db_url=db_url,
            db_schema=db_schema,
            select_cols=select_cols,
        )

        out_path = output_dir / f"{table_name}.parquet"
        df.write_parquet(out_path)
        dumped_tables += 1
        logger.info(
            "Wrote {} rows x {} cols to {}",
            df.height,
            df.width,
            out_path,
        )

    logger.info("Done. Dumped {} AACT table(s) to {}", dumped_tables, output_dir)


if __name__ == "__main__":
    main()
