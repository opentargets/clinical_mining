from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
import polars as pl
from clinical_mining.utils.pipeline import execute_step


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).parent), config_name="config"
)
def main(cfg: DictConfig) -> pl.DataFrame:
    """Main function to run the clinical mining pipeline."""
    data_store = {}

    # Construct database URI from config
    db_uri = f"postgresql://{cfg.db_properties.user}:{cfg.db_properties.password}@{cfg.db_properties.url}"

    # Load all data sources
    for name, source in cfg.inputs.items():
        print(f"Loading input: {name}")
        # TODO: implement load_table / print_table_schema?
        if source.type == "db_table":
            query = f"SELECT {', '.join(source.select_cols)} FROM {cfg.db_properties.schema}.{name}"
            if cfg.mode.debug and "nct_id" in source.select_cols:
                study_id = cfg.mode.debug_study_id
                query += f" WHERE nct_id = '{study_id}'"
                print(f"  [DEBUG MODE] Filtering {name} for study: {study_id}")
            data_store[name] = pl.read_database_uri(query=query, uri=db_uri)
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
