from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import polars as pl

from clinical_mining.utils.pipeline import execute_step
from clinical_mining.utils.db import construct_db_uri, load_db_table
from clinical_mining.utils.spark_helpers import spark_session


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).parent), config_name="config"
)
def main(cfg: DictConfig) -> pl.DataFrame:
    """Main function to run the clinical mining pipeline."""
    data_store = {}

    # Construct database URI from config
    aact_url = construct_db_uri(
        db_type=cfg.db_properties.aact.type,
        db_uri=cfg.db_properties.aact.uri,
        db_user=cfg.db_properties.aact.user,
        db_password=cfg.db_properties.aact.password,
    )

    # Load all data sources
    for name, source in cfg.inputs.items():
        logger.info(f"Loading input: {name}")
        if source.format == "db_table":
            data_store[name] = load_db_table(
                table_name=name,
                db_url=aact_url,
                select_cols=list(source.select_cols),
                db_schema=cfg.db_properties.aact.schema,
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

    # Run pipeline sections
    for section in ["setup", "generate", "post_process"]:
        logger.info(f"\n----- Running {section.upper()} section -----")
        steps = cfg.pipeline.get(section, [])
        if not steps:
            logger.info(f"No steps found in {section} section, skipping...")
            continue
        for step in steps:
            logger.info(f"Executing step: {step.name}")
            execute_step(step, data_store)

    # Write the final output
    date = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(cfg.datasets.output_path) / date
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {k: v for k, v in data_store.items() if k.startswith("output_")}
    for k, v in outputs.items():
        output_name = k.removeprefix("output_")
        v.unique().write_parquet(output_dir / f"{output_name}.parquet")
        logger.info(f"output {output_name} written to {output_dir}")

    return data_store


if __name__ == "__main__":
    main()
