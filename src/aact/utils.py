
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f

def assign_drug_id(df, molecule):
    df_w_id = (
        df.withColumn("drug_name", f.lower("drug_name"))
        .join(
            molecule.withColumn("drug_name", f.explode("drug_names")),
            "drug_name",
            "left",
        )
        .drop("drug_names")
    )
    print(f"Unmapped rows: {df_w_id.filter(f.col('drug_id').isNull()).count()}")
    print(f"Mapped rows: {df_w_id.filter(f.col('drug_id').isNotNull()).count()}")
    return df_w_id


def assign_disease_id(df, diseases):
    df_w_id = (
        df.withColumn("disease_name", f.lower("disease_name"))
        .join(
            diseases.withColumn("disease_name", f.explode("disease_names")),
            "disease_name",
            "left",
        )
        .drop("disease_names")
    )
    print(f"Unmapped rows: {df_w_id.filter(f.col('disease_id').isNull()).count()}")
    print(f"Mapped rows: {df_w_id.filter(f.col('disease_id').isNotNull()).count()}")
    return df_w_id

def process_molecule(path: str) -> DataFrame:
    return (
    SparkSession.builder.getOrCreate()
    .read.parquet("molecule").select("id", f.lower("name").alias("name"), "synonyms")
    .withColumn("synonyms", f.transform(f.col("synonyms"), lambda x: f.lower(x)))
    .withColumn("drug_names", f.array_union(f.array(f.col("name")), f.col("synonyms")))
    .selectExpr("id as drug_id", "drug_names")
    .persist()
)

def process_disease(path: str) -> DataFrame:
    return (
        SparkSession.builder.getOrCreate()
        .read.parquet(path)
        .select("id", f.lower("name").alias("name"), "synonyms")
        .withColumn(
            "synonyms",
            f.transform(f.col("synonyms.hasExactSynonym"), lambda x: f.lower(x)),
        )
        .withColumn(
            "disease_names", f.array_union(f.array(f.col("name")), f.col("synonyms"))
        )
        .selectExpr("id as disease_id", "disease_names")
        .distinct()
        .persist()
    )
