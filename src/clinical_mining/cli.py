from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig
import polars as pl

from clinical_mining.utils.pipeline import execute_step, normalise_steps
from clinical_mining.utils.db import construct_db_uri, load_db_table
from clinical_mining.utils.spark_helpers import spark_session


def _run_transform(workflow_cfg: DictConfig, cfg: DictConfig) -> dict[str, Any]:
    """Run the transform pipeline (setup → generate → post_process)."""
    data_store = {}

    date = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(cfg.datasets.output_path) / date
    output_dir.mkdir(parents=True, exist_ok=True)
    data_store["output_dir"] = output_dir

    db_urls = {}
    for db_name, props in cfg.db_properties.items():
        db_urls[db_name] = construct_db_uri(
            db_type=props.type,
            db_uri=props.uri,
            db_user=props.get("user"),
            db_password=props.get("password"),
        )

    inputs = cfg.get("inputs", {})
    for name, source in inputs.items():
        logger.info(f"Loading input: {name}")
        if source.format == "db_table":
            db_name = source.get("db", "aact")
            if db_name not in db_urls:
                raise KeyError(
                    f"Input '{name}' references db '{db_name}', but it is not defined under db_properties"
                )
            data_store[name] = load_db_table(
                table_name=name,
                db_url=db_urls[db_name],
                select_cols=list(source.select_cols),
                db_schema=cfg.db_properties[db_name].schema,
            )
        elif "spark" in name:
            if data_store.get("spark_session") is None:
                data_store["spark_session"] = spark_session()
            data_store[name] = data_store["spark_session"].read.load(
                source.path, format=source.format
            )
        elif source.format == "parquet":
            data_store[name] = pl.read_parquet(source.path)
        elif source.format == "json":
            data_store[name] = pl.read_ndjson(source.path)
        else:
            raise ValueError(f"unsupported file format: {source.format}")

    for section_name in ["setup", "generate", "post_process"]:
        logger.info(f"\n----- Running {section_name.upper()} section -----")
        section = workflow_cfg.get(section_name, {})
        steps = normalise_steps(section)
        if not steps:
            logger.info(f"No steps found in {section_name} section, skipping...")
            continue
        for name, step in steps:
            logger.info(f"Executing step: {name}")
            execute_step(step, data_store)

    # Write outputs: anything prefixed output_

    outputs = {k: v for k, v in data_store.items() if k.startswith("output_")}
    for k, v in outputs.items():
        if v is None:
            logger.info(f"Step '{k}' returned None (inspect mode), skipping output")
            continue
        output_name = k.removeprefix("output_")
        if isinstance(v, pl.DataFrame):
            v.unique().write_parquet(output_dir / f"{output_name}.parquet")
            logger.info(f"output {output_name} written to {output_dir}")
        elif isinstance(v, dict):
            import json
            out_path = output_dir / f"{output_name}.json"
            with out_path.open("w") as f:
                json.dump(v, f, indent=2, default=str)
            logger.info(f"output {output_name} written to {out_path}")
        else:
            logger.warning(f"Unknown output type for '{k}': {type(v)}")

    return data_store


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).parent), config_name="config"
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Main function to run clinical mining workflows."""
    workflow = cfg.get("workflow")
    if not workflow:
        logger.info("No workflow defined. Add a recipe to activate a workflow (e.g. +recipe=<name>).")
        return {}

    for name, section in workflow.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running workflow: {name}")
        logger.info(f"{'=' * 50}")

        if name == "transform":
            _run_transform(section, cfg)
        else:
            logger.warning(f"Unknown workflow '{name}', skipping")

    return {}


if __name__ == "__main__":
    main()
