from pandas import DataFrame
import polars as pl
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from ontoma import OnToma
from ontoma import OpenTargetsDisease, OpenTargetsDrug
from ontoma.dataset.raw_entity_lut import RawEntityLUT
from clinical_mining.utils.polars_helpers import convert_polars_to_spark


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
    clinical_associations: pl.DataFrame,
    disease_index: DataFrame,
    drug_index: DataFrame,
    chembl_curation: pl.DataFrame,
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
        disease_index: Spark DataFrame with disease index
        drug_index: Spark DataFrame with drug molecule index
        chembl_curation: Polars DataFrame with ChEMBL curation extracted in `extract_chembl_ct_curation`
        drug_id_column_name: Name for output drug ID column
        disease_id_column_name: Name for output disease ID column

    Returns:
        Polars DataFrame with added drug and disease ID columns
    """
    # Convert only essential columns to Spark DataFrame for OnToma processing
    associations_spark = convert_polars_to_spark(
        clinical_associations.select(drug_column_name, disease_column_name), spark
    )
    chembl_curation_spark = convert_polars_to_spark(chembl_curation, spark)

    # Create entity lookup tables
    disease_label_lut = OpenTargetsDisease.as_label_lut(disease_index)
    drug_label_lut = OpenTargetsDrug.as_label_lut(drug_index)

    # Create curation lookup tables using the helper function
    curation_lut_disease = _create_curation_lut(
        chembl_curation_spark, "disease_id", "disease_name", "DS"
    )
    curation_lut_drug = _create_curation_lut(
        chembl_curation_spark, "drug_id", "drug_name", "CD"
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

    # Note: Spark session should be managed by the calling code, not stopped here
    return result_df
