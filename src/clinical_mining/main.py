from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pyspark.sql import DataFrame

from clinical_mining.utils.spark import SparkSession
from clinical_mining.utils.utils import assign_approval_status
from clinical_mining.utils.utils import call_with_dependencies
from clinical_mining.data_sources.aact.aact import (
    extract_drug_indications,
    extract_clinical_trials,
)
from clinical_mining.data_sources.chembl.indications import extract_chembl_indications
from clinical_mining.data_sources.aact.mapping_utils import (
    assign_drug_id,
    assign_disease_id,
    process_molecule,
    process_disease,
)


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).parent), config_name="config"
)
def main(
    cfg: DictConfig,
) -> DataFrame:
    print(cfg)
    spark = SparkSession(
        db_url=cfg.get("db_properties", {}).get("url"),
        user=cfg.get("db_properties", {}).get("user"),
        password=cfg.get("db_properties", {}).get("password"),
        schema=cfg.get("db_properties", {}).get("schema"),
    )

    #### DATA LOADING
    data_sources = {}
    for source in cfg.inputs:
        name = source.name
        source_type = source.type
        print(f"Loading {name} from {source_type}...")
        if source_type == "db_table":
            data_sources[name] = spark.load_table(
                name,
                select_cols=list(source.select_cols),
            )
        elif source_type == "file":
            data_sources[name] = spark.session.read.format(source.format).load(
                source.path
            )

    # Prepare dynamic arguments for functions that need them
    data_sources["additional_metadata"] = [
        data_sources[source.name]
        for source in cfg.inputs
        if source.get("role") == "metadata"
    ]
    data_sources["molecule"] = process_molecule(
        spark, cfg["datasets"]["molecule_path"]
    )
    data_sources["disease"] = process_disease(
        spark, cfg["datasets"]["disease_path"]
    )

    # Define the pipeline steps, specifying the output name for each
    pipeline_steps = [
        {"name": "trials", "func": extract_clinical_trials},
        {"name": "aact_indications", "func": extract_drug_indications},
        {"name": "chembl_indications", "func": extract_chembl_indications},
        {
            "name": "indications",
            "func": lambda aact_indications, chembl_indications: aact_indications.unionByName(
                chembl_indications, allowMissingColumns=True
            ).persist(),
        },
        {"name": "trials_mapped_drug", "func": assign_drug_id},
        {"name": "trials_mapped_drug_disease", "func": assign_disease_id},
        {"name": "final_df", "func": assign_approval_status},
    ]

    # Execute the pipeline
    for step in pipeline_steps:
        output_name = step["name"]
        func = step["func"]
        print(f"Executing step: {output_name}")
        data_sources[output_name] = call_with_dependencies(func, data_sources)

    df = data_sources["final_df"]

    df.write.parquet(f"{cfg['datasets']['output_path']}/{date}", mode="overwrite")
    spark.stop()
    return df


if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d")
    main()
