from pandas import DataFrame
import polars as pl
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import pyspark.sql.functions as f
from ontoma import OnToma
from ontoma import OpenTargetsDisease, OpenTargetsDrug
from clinical_mining.data_sources.chembl.curation import extract_chembl_ct_curation
from ontoma.dataset.raw_entity_lut import RawEntityLUT


def initialise_spark():
    """OnToma works on Spark dataframes."""
    # Stop any existing Spark sessions to avoid conflicts
    try:
        SparkSession.getActiveSession().stop()
    except Exception:
        pass
    config = (
        SparkConf()
        .set("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.0")
        .set("spark.driver.memory", "12g")
        .set("spark.executor.memory", "12g")
        .set("spark.driver.maxResultSize", "8g")
        .set("spark.sql.adaptive.enabled", "true")
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer.max", "1g")
        .set("spark.executor.instances", "1")
        .set("spark.executor.cores", "4")
        .set("spark.driver.host", "localhost")
        .set("spark.driver.bindAddress", "localhost")
        .set("spark.ui.enabled", "false")
        .set("spark.sql.shuffle.partitions", "200")
        .set("spark.default.parallelism", "4")
    )
    return (
        SparkSession.builder.appName("clinical_mining_entity_mapping")
        .master("local[*]")
        .config(conf=config)
        .getOrCreate()
    )


def _create_curation_lut(
    chembl_curation_df, entity_id_col: str, entity_name_col: str, entity_type: str
) -> RawEntityLUT:
    """Create a curation lookup table for a specific entity type.

    Args:
        chembl_curation_df: Spark DataFrame with ChEMBL curation data
        entity_id_col: Column name for entity ID (e.g., 'disease_id', 'drug_id')
        entity_name_col: Column name for entity name (e.g., 'disease_name', 'drug_name')
        entity_type: Entity type code ('DS' for disease, 'CD' for compound/drug)

    Returns:
        RawEntityLUT configured for the specific entity type
    """
    return RawEntityLUT(
        _df=chembl_curation_df.select(
            f.col(entity_id_col).alias("entityId"),
            f.col(entity_name_col).alias("entityLabel"),
            f.lit(1.0).alias("entityScore"),
            f.lit("term").alias("nlpPipelineTrack"),
            f.lit(entity_type).alias("entityType"),
            f.lit("label").alias("entityKind"),
        )
        .filter(f.col("entityId").isNotNull())
        .distinct(),
        _schema=RawEntityLUT.get_schema(),
    )


def _prepare_unified_query_df(associations_df, drug_column: str, disease_column: str):
    """Prepare a unified DataFrame with both drugs and diseases in a single column.

    Args:
        associations_df: Spark DataFrame with clinical associations
        drug_column: Name of the column containing drug names
        disease_column: Name of the column containing disease names

    Returns:
        Spark DataFrame with columns: query_label, entity_type
    """
    disease_df = (
        associations_df.filter(f.col(disease_column).isNotNull())
        .select(
            f.col(disease_column).alias("query_label"), f.lit("DS").alias("entity_type")
        )
        .distinct()
    )
    drug_df = (
        associations_df.filter(f.col(drug_column).isNotNull())
        .select(
            f.col(drug_column).alias("query_label"), f.lit("CD").alias("entity_type")
        )
        .distinct()
    )
    return disease_df.union(drug_df)


def _process_mapping_results(mapped_spark_df) -> pl.DataFrame:
    """Process OnToma mapping results into separate disease/drug ID columns.

    Args:
        mapped_spark_df: Spark DataFrame with OnToma mapping results

    Returns:
        Polars DataFrame with disease_ids and drug_ids columns
    """
    return (
        pl.from_pandas(mapped_spark_df.toPandas())
        .with_columns(
            [
                pl.when(pl.col("entity_type") == "DS")
                .then(pl.col("mapped_ids"))
                .alias("disease_ids"),
                pl.when(pl.col("entity_type") == "CD")
                .then(pl.col("mapped_ids"))
                .alias("drug_ids"),
            ]
        )
        .drop("mapped_ids")
    )


def map_entities(
    spark: SparkSession,
    clinical_associations: DataFrame,
    disease_index: DataFrame,
    drug_index: DataFrame,
    chembl_curation: DataFrame,
    drug_column_name: str,
    disease_column_name: str,
    drug_id_column_name: str = "drug_id",
    disease_id_column_name: str = "disease_id",
) -> DataFrame:
    """Map drug and disease entities to their standardized IDs using OnToma.

    This function takes a DataFrame with clinical associations and maps the drug and
    disease names to their corresponding standardized IDs using OnToma entity mapping.

    Args:
        spark (SparkSession): Spark session for OnToma
        clinical_associations: Polars DataFrame with clinical associations
        drug_column_name: Name of column containing drug names to map
        disease_column_name: Name of column containing disease names to map
        disease_index_path: Path to disease index parquet file
        drug_index_path: Path to drug molecule index parquet file
        chembl_curation_path: Path to ChEMBL curation parquet file
        drug_id_column_name: Name for output drug ID column
        disease_id_column_name: Name for output disease ID column

    Returns:
        Polars DataFrame with added drug and disease ID columns
    """
    # Convert to Spark DataFrame for OnToma processing
    # associations_spark = spark.createDataFrame(
    #     clinical_associations.to_pandas()
    # ).repartition(200)
    associations_spark = clinical_associations

    # Create entity lookup tables
    disease_label_lut = OpenTargetsDisease.as_label_lut(disease_index)
    drug_label_lut = OpenTargetsDrug.as_label_lut(drug_index)

    # Create curation lookup tables using the helper function
    curation_lut_disease = _create_curation_lut(
        chembl_curation, "disease_id", "disease_name", "DS"
    )
    curation_lut_drug = _create_curation_lut(
        chembl_curation, "drug_id", "drug_name", "CD"
    )

    # Prepare unified query DataFrame
    query_df = _prepare_unified_query_df(
        associations_spark, drug_column_name, disease_column_name
    )

    # Optimize partitioning to prevent large tasks
    query_df = query_df.repartition(100, "entity_type")

    # Initialize OnToma with all lookup tables
    ontoma = OnToma(
        spark=spark,
        entity_lut_list=[
            disease_label_lut,
            drug_label_lut,
            curation_lut_disease,
            curation_lut_drug,
        ],
    )
    mapped_df = ontoma.map_entities(
        df=query_df,
        result_col_name="mapped_ids",
        entity_col_name="query_label",
        entity_kind="label",
        type_col=f.col("entity_type"),
    )

    # Process mapping results
    polars_mapped_df = _process_mapping_results(mapped_df)

    # Join mapped IDs back to original associations
    result_df = (
        clinical_associations.drop(
            [
                col
                for col in [drug_id_column_name, disease_id_column_name]
                if col in clinical_associations.columns
            ]
        )  # Remove existing ID columns if present
        .join(
            # Add mapped disease IDs
            polars_mapped_df.filter(pl.col("entity_type") == "DS").select(
                ["query_label", "disease_ids"]
            ),
            left_on=disease_column_name,
            right_on="query_label",
            how="left",
            suffix="_disease",
        )
        .join(
            # Add mapped drug IDs
            polars_mapped_df.filter(pl.col("entity_type") == "CD").select(
                ["query_label", "drug_ids"]
            ),
            left_on=drug_column_name,
            right_on="query_label",
            how="left",
            suffix="_drug",
        )
        .explode("disease_ids")
        .explode("drug_ids")
        .rename(
            {"disease_ids": disease_id_column_name, "drug_ids": drug_id_column_name}
        )
    )

    # Drop any remaining query_label columns if they exist
    columns_to_drop = [col for col in result_df.columns if "query_label" in col]
    print(columns_to_drop)
    if columns_to_drop:
        result_df = result_df.drop(columns_to_drop)

    spark.stop()
    return result_df


disease_path = "/Users/irenelopez/EBI/repos/clinical_mining/data/inputs/disease"
drug_path = "/Users/irenelopez/EBI/repos/clinical_mining/data/inputs/drug_molecule"
chembl_curation_path = "/Users/irenelopez/EBI/repos/clinical_mining/data/outputs/chembl_indications/0910.parquet"
associations_path = "/Users/irenelopez/EBI/repos/clinical_mining/data/outputs/clinical_trials/2025-09-08"
output_path = "/Users/irenelopez/EBI/repos/clinical_mining/data/outputs/experimental/mapped_clinical_trials/2025-10-14-pipeline.parquet"

spark = initialise_spark()
# clinical_associations = pl.read_parquet(associations_path)
clinical_associations = spark.read.parquet(associations_path)
disease_index = spark.read.parquet(disease_path)
drug_index = spark.read.parquet(drug_path)
chembl_curation = spark.read.parquet(chembl_curation_path) # TODO: from extract_chembl_ct_curation, expect polars


mapped_associations = map_entities(
    spark,
    clinical_associations=clinical_associations,
    disease_index=disease_index,
    drug_index=drug_index,
    chembl_curation=chembl_curation,
    drug_column_name="drug_name",
    disease_column_name="disease_name",
)
mapped_associations.write_parquet(output_path)

spark.stop()
