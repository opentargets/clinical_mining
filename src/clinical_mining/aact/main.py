from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pyspark.sql import DataFrame
import pyspark.sql.functions as f

from clinical_mining.utils.db import AACTConnector
from clinical_mining.aact.utils import (
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
    db = AACTConnector(
        db_url=cfg["db_properties"]["url"],
        user=cfg["db_properties"]["user"],
        password=cfg["db_properties"]["password"],
        schema=cfg["db_properties"]["schema"],
    )

    ## Load Data
    studies = db.load_table(
        "studies",
        select_cols=list(cfg["db_tables"]["studies"]),
    ).distinct()

    interventions = (
        db.load_table(
            "interventions",
            select_cols=list(cfg["db_tables"]["interventions"]),
        )
        .filter(f.col("intervention_type") == "DRUG")
        # Placebo is listed as an intervention
        .filter(~f.lower("name").contains("placebo"))
        .distinct()
    )

    browse_interventions = (
        db.load_table(
            "browse_interventions",
            select_cols=list(cfg["db_tables"]["browse_interventions"]),
        )
        # filter by interventions of interest
        .join(interventions.select("nct_id"), "nct_id")
        # Get the direct mapping
        .filter(f.col("mesh_type") == "mesh-list")
        .selectExpr("nct_id", "downcase_mesh_term as mesh_term")
        .distinct()
    )
    conditions = db.load_table(
        "conditions",
        select_cols=list(cfg["db_tables"]["conditions"]),
    ).distinct()
    browse_conditions = (
        db.load_table(
            "browse_conditions",
            select_cols=list(cfg["db_tables"]["browse_conditions"]),
        )
        .filter(f.col("mesh_type") == "mesh-list")
        .selectExpr("nct_id", "downcase_mesh_term as mesh_term")
        .distinct()
    )

    study_references = db.load_table(
        "study_references",
        select_cols=list(cfg["db_tables"]["study_references"]),
    ).distinct()

    study_design = db.load_table(
        "designs",
        select_cols=list(cfg["db_tables"]["designs"]),
    ).distinct()
    
    ## Annotate condition in a study
    joined_conditions = (
        conditions.selectExpr("nct_id", "downcase_name")
        .join(browse_conditions.selectExpr("nct_id", "mesh_term"), "nct_id", "left")
        .withColumn(
            "disease_label", f.coalesce(f.col("mesh_term"), f.col("downcase_name"))
        )
        .drop("downcase_name", "mesh_term")
        .distinct()
    )

    ## Putting things together
    joined_interventions = (
        interventions.selectExpr("nct_id", "name")
        .join(browse_interventions.selectExpr("nct_id", "mesh_term"), "nct_id", "left")
        .withColumn("drug_label", f.coalesce(f.col("mesh_term"), f.col("name")))
        .drop("name", "mesh_term")
        .distinct()
    )

    int_studies = (
        studies.filter(f.col("study_type") == "INTERVENTIONAL")
        .join(
            joined_interventions.selectExpr("nct_id", "drug_label as drug_name"),
            on="nct_id",
            how="inner",
        )
        .join(
            joined_conditions.selectExpr("nct_id", "disease_label as disease_name"),
            on="nct_id",
            how="inner",
        )
        .join(
            study_design.selectExpr("nct_id", "primary_purpose as purpose"),
            on="nct_id",
            how="inner",
        )
        .join(
            study_references, 
            on="nct_id",
            how="left",
        )
        .drop("study_type")
        .distinct()
    ).persist()

    ## Drug mapping
    trials_mapped_drug = assign_drug_id(
        int_studies, process_molecule(db.spark, cfg["datasets"]["molecule_path"])
    )

    ## Indication mapping
    trials_mapped_drug_disease = assign_disease_id(
        trials_mapped_drug, process_disease(db.spark, cfg["datasets"]["disease_path"])
    )
    trials_mapped_drug_disease.write.parquet(
        f"{cfg['datasets']['output_path']}_{date}", mode="overwrite"
    )
    db.spark.stop()
    return trials_mapped_drug_disease


if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d")
    main()
