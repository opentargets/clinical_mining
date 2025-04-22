from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f

from src.aact.tables import (
    Studies,
    Interventions,
    Browse_Interventions,
    Conditions,
    Browse_Conditions,
)
from src.utils.db import (
    connect_to_db,
    select_table,
)
from src.aact.utils import (
    assign_drug_id,
    assign_disease_id,
    process_molecule,
    process_disease,
)
from datetime import datetime

spark = SparkSession.builder.getOrCreate()


DATABASE_URL = (
    "postgresql+psycopg2://irenelopez:Ephemeral2023@"
    "aact-db.ctti-clinicaltrials.org:5432/aact"
)
DISEASE_PATH = "data/disease"
MOLECULE_PATH = "data/molecule"
OUTPUT_PATH = "aact_trials_"


def main(DATABASE_URL: str, disease_path: str, molecule_path: str) -> DataFrame:
    engine = connect_to_db(DATABASE_URL)

    studies = select_table(Studies, engine).distinct()

    interventions = (
        select_table(Interventions, engine)
        .drop("id")
        .filter(f.col("intervention_type") == "DRUG")
        # Placebo is listed as an intervention
        .filter(~f.lower("name").contains("placebo"))
        .distinct()
    )

    browse_interventions = (
        select_table(Browse_Interventions, engine)
        # filter by interventions of interest
        .join(interventions.select("nct_id"), "nct_id")
        # Get the direct mapping
        .filter(f.col("mesh_type") == "mesh-list")
        .selectExpr("nct_id", "downcase_mesh_term as mesh_term")
        .distinct()
    )
    joined_interventions = (
        interventions.selectExpr("nct_id", "name")
        .join(browse_interventions.selectExpr("nct_id", "mesh_term"), "nct_id", "left")
        .withColumn("drug_label", f.coalesce(f.col("mesh_term"), f.col("name")))
        .drop("name", "mesh_term")
        .distinct()
    )

    """### Annotate condition in a study"""

    conditions = select_table(Conditions, engine).drop("id").distinct()
    conditions.show()

    from sqlmodel import Session, select

    with Session(engine) as session:  # TODO: pass filter to select_table
        statement = (
            # For the code not to break, we apply the filter with SQL
            select(Browse_Conditions).where(Browse_Conditions.mesh_type == "mesh-list")
        )
        records = session.exec(statement).all()

    browse_conditions = (
        spark.createDataFrame([e.model_dump() for e in records])
        .selectExpr("nct_id", "downcase_mesh_term as mesh_term")
        .distinct()
    )

    joined_conditions = (
        conditions.selectExpr("nct_id", "downcase_name")
        .join(browse_conditions.selectExpr("nct_id", "mesh_term"), "nct_id", "left")
        .withColumn(
            "disease_label", f.coalesce(f.col("mesh_term"), f.col("downcase_name"))
        )
        .drop("downcase_name", "mesh_term")
        .distinct()
    )

    """### Putting things together"""

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
        .distinct()
    ).persist()

    ## Drug mapping
    trials_mapped_drug = assign_drug_id(int_studies, process_molecule(molecule_path))

    ## Indication mapping

    trials_mapped_drug_disease = assign_disease_id(
        trials_mapped_drug, process_disease(disease_path)
    )
    return trials_mapped_drug_disease


if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d")
    trials = main(DATABASE_URL, DISEASE_PATH, MOLECULE_PATH)
    trials.write.parquet(
        f"{OUTPUT_PATH}_{date}", mode="overwrite"
    )
