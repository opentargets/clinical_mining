from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pyspark.sql import DataFrame

from clinical_mining.utils.pipeline import execute_step
from clinical_mining.utils.spark import SparkSession


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).parent), config_name="config"
)
def main(cfg: DictConfig) -> DataFrame:
    """Main function to run the clinical mining pipeline."""
    spark = SparkSession(
        db_url=cfg.db_properties.url,
        user=cfg.db_properties.user,
        password=cfg.db_properties.password,
        schema=cfg.db_properties.schema,
    )

    data_store = {}

    ## Load all data sources
    for name, source in cfg.inputs.items():
        print(f"Loading input: {name}")
        if source.type == "db_table":
            data_store[name] = spark.load_table(
                name, select_cols=list(source.select_cols)
            )
        elif source.type == "file":
            data_store[name] = spark.session.read.format(source.format).load(
                source.path
            )

    ## Run pipeline sections
    # Setup - Generate intermediate DataFrames
    for step in cfg.pipeline.setup:
        print(f"Setup step: {step.name}")
        execute_step(step, data_store, spark)

    # Generate Drug/Indication DataFrames
    for step in cfg.pipeline.generate:
        print(f"Data Generation step: {step.name}")
        execute_step(step, data_store, spark)

    # Post-Process - Generate final DataFrame
    for step in cfg.pipeline.post_process:
        print(f"Post-Processing step: {step.name}")
        execute_step(step, data_store, spark)

    # Write the final output
    date = datetime.now().strftime("%Y-%m-%d")
    final_df = data_store["final_df"].distinct()
    final_df.write.parquet(f"{cfg.datasets.output_path}/{date}", mode="overwrite")
    spark.stop()
    return final_df


if __name__ == "__main__":
    main()
