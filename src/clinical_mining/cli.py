from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
import polars as pl

from clinical_mining.utils.pipeline import execute_step
from clinical_mining.utils.db import construct_db_url, load_db_table


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).parent), config_name="config"
)
def main(cfg: DictConfig) -> pl.DataFrame:
    """Main function to run the clinical mining pipeline."""
    data_store = {}

    # Construct database URI from config
    aact_url = construct_db_url(
        db_type=cfg.db_properties.aact.type,
        db_uri=cfg.db_properties.aact.uri,
        db_user=cfg.db_properties.aact.user,
        db_password=cfg.db_properties.aact.password,
    )

    # Load all data sources
    for name, source in cfg.inputs.items():
        print(f"Loading input: {name}")
        if source.type == "db_table":
            data_store[name] = load_db_table(
                table_name=name,
                db_url=aact_url,
                select_cols=list(source.select_cols),
                db_schema=cfg.db_properties.aact.schema,
            )
        elif source.type == "file":
            if source.format == "json":
                data_store[name] = pl.read_ndjson(source.path)
            else:
                raise ValueError(f"Unsupported file format: {source.format}")

    # Run pipeline sections
    for section in ["setup", "generate", "post_process"]:
        print(f"\n----- Running {section.upper()} section -----")
        for step in cfg.pipeline.get(section, []):
            print(f"Executing step: {step.name}")
            execute_step(step, data_store)

    # Write the final output
    date = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(cfg.datasets.output_path) / date
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = data_store["final_df"].unique()
    final_df.write_parquet(output_dir / "df.parquet")

    print(f"Output written to {output_dir}")

    return final_df


if __name__ == "__main__":
    main()
