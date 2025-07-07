from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pyspark.sql import DataFrame

from clinical_mining.utils.db import AACTConnector
from clinical_mining.utils.utils import assign_approval_status
from clinical_mining.data_sources.aact.aact import extract_aact_indications
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
    db = AACTConnector(
        db_url=cfg["db_properties"]["url"],
        user=cfg["db_properties"]["user"],
        password=cfg["db_properties"]["password"],
        schema=cfg["db_properties"]["schema"],
    )

    #### DATA LOADING
    studies = db.load_table(
        "studies",
        select_cols=list(cfg["db_tables"]["studies"]),
    )
    interventions = (
        db.load_table(
            "interventions",
            select_cols=list(cfg["db_tables"]["interventions"]),
        )
    )
    browse_interventions = (
        db.load_table(
            "browse_interventions",
            select_cols=list(cfg["db_tables"]["browse_interventions"]),
        )
    )
    conditions = db.load_table(
        "conditions",
        select_cols=list(cfg["db_tables"]["conditions"]),
    )
    browse_conditions = (
        db.load_table(
            "browse_conditions",
            select_cols=list(cfg["db_tables"]["browse_conditions"]),
        )
    )
    study_references = db.load_table(
        "study_references",
        select_cols=list(cfg["db_tables"]["study_references"]),
    )
    study_design = db.load_table(
        "designs",
        select_cols=list(cfg["db_tables"]["designs"]),
    )
    chembl_indications_raw = db.spark.session.read.json(cfg["datasets"]["chembl_indications_path"]) ## FIXME DB naming
    

    ##### DRUG/DISEASE EXTRACTION
    aact_indications = extract_aact_indications(
        studies,
        interventions,
        conditions,
        browse_conditions,
        browse_interventions,
        study_references,
        study_design,
    )
    chembl_indications = extract_chembl_indications(chembl_indications_raw)
    indications = aact_indications.unionByName(chembl_indications, allowMissingColumns=True).persist()

    ##### DRUG/DISEASE MAPPING
    trials_mapped_drug = assign_drug_id(
        indications, process_molecule(db.spark, cfg["datasets"]["molecule_path"]), verbose=False
    )

    trials_mapped_drug_disease = assign_disease_id(
        trials_mapped_drug, process_disease(db.spark, cfg["datasets"]["disease_path"]), verbose=False
    )


    ##### APPROVAL ASSIGNMENT
    df = assign_approval_status(trials_mapped_drug_disease)
    
    df.write.parquet(
        f"{cfg['datasets']['output_path']}/{date}", mode="overwrite"
    )
    db.spark.stop()
    return df


if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d")
    main()
