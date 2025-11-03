from pandas import DataFrame
import polars as pl
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from ontoma import OnToma, OpenTargetsDisease, OpenTargetsDrug
from ontoma.dataset.raw_entity_lut import RawEntityLUT
from ontoma.ner.drug import extract_drug_entities
from clinical_mining.utils.polars_helpers import convert_polars_to_spark
from datetime import datetime
from pathlib import Path


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
    ner_extract_drug: bool = True,
    ner_batch_size: int = 256,
    ner_cache_path: str = f".cache/ner/{datetime.now().strftime('%Y%m%d')}.parquet",
) -> DataFrame:
    """Map drug and disease entities to their standardized IDs using OnToma.

    This function uses a tiered approach for drug mapping:
    1. First attempt: Dictionary mapping using curation and drug indices
    2. Second attempt (if use_ner_fallback=True): NER extraction for unmapped drugs

    Disease mapping always uses dictionary approach only.

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
        ner_extract_drug: If True, apply NER to unmapped drug labels (default: True)
        ner_batch_size: Batch size for NER extraction (default: 256)
        ner_cache_path: Path to parquet file for caching NER results (default: .cache/ner/{datetime.now().strftime('%Y%m%d')}.parquet)

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
    ).repartition(50, "entity_type")

    # STEP 1: Dictionary mapping (curation + indices)
    # This maps ~50% of drugs and diseases
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

    # STEP 2: NER fallback for unmapped drugs
    # Apply NER to the remaining ~50% unmapped drugs
    if ner_extract_drug:
        unmapped_drugs = (
            mapped_df.filter(
                (f.col("entity_type") == "CD")
                & (f.col("mapped_ids").isNull() | (f.size("mapped_ids") == 0))
            )
            .select("query_label")
            .distinct()
        )

        unmapped_count = unmapped_drugs.count()
        if unmapped_count > 0:
            print(f"apply ner to {unmapped_count} unmapped drug labels...")

            # Simple caching: try to load, else extract and save
            if ner_cache_path:
                cache_file = Path(ner_cache_path)
                if cache_file.exists():
                    print(f"load ner from cache: {cache_file}")
                    ner_extracted_raw = spark.read.parquet(str(cache_file))

                # Run NER if not cached
                else:
                    ner_extracted_raw = extract_drug_entities(
                        spark=spark,
                        df=unmapped_drugs,
                        input_col="query_label",
                        output_col="extracted_drugs",
                        use_regex=True,
                        use_biobert=True,
                        use_drugtemist=True,
                        batch_size=ner_batch_size,
                    )

                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    print(f"save ner to cache: {cache_file}")
                    ner_extracted_raw.write.toPandas().to_parquet(str(cache_file))

            # Explode extracted drugs
            ner_extracted = (
                ner_extracted_raw.filter(f.size("extracted_drugs") > 0)
                .select(
                    f.col("query_label"),
                    f.explode("extracted_drugs").alias("clean_label"),
                )
            )

            # Map the cleaned labels with new OnToma instance (faster and safer for clean drug names)
            ontoma_clean = OnToma(spark=spark, entity_lut_list=[drug_label_lut])
            ner_mapped = ontoma_clean.map_entities(
                df=ner_extracted,
                result_col_name="mapped_ids",
                entity_col_name="clean_label",
                entity_kind="label",
                type_col=f.lit("CD"),
            )

            # Aggregate: original label → list of mapped IDs
            ner_aggregated = (
                ner_mapped.filter(
                    f.col("mapped_ids").isNotNull() & (f.size("mapped_ids") > 0)
                )
                .groupBy("query_label")
                .agg(
                    f.array_distinct(f.flatten(f.collect_list("mapped_ids"))).alias(
                        "ner_mapped_ids"
                    )
                )
            )

            # Merge: prefer dictionary, fallback to NER
            mapped_df = (
                mapped_df.alias("dict")
                .join(
                    ner_aggregated.select("query_label", "ner_mapped_ids").alias("ner"),
                    on="query_label",
                    how="left",
                )
                .select(
                    f.col("dict.query_label"),
                    f.col("dict.entity_type"),
                    f.when(
                        f.col("dict.mapped_ids").isNotNull()
                        & (f.size("dict.mapped_ids") > 0),
                        f.col("dict.mapped_ids"),
                    )
                    .otherwise(f.col("ner.ner_mapped_ids"))
                    .alias("mapped_ids"),
                )
            )

            recovered = ner_aggregated.count()
            print(f"ner recovered {recovered}/{unmapped_count} ({recovered / unmapped_count * 100:.1f}%)")

    # Convert to Polars and join back
    polars_mapped_df = _process_mapping_results(mapped_df)

    return (
        clinical_associations.join(
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
            {
                "disease_ids": f"new_{disease_id_column_name}",
                "drug_ids": f"new_{drug_id_column_name}",
            }
        )
        .with_columns(
            pl.coalesce(
                pl.col(disease_id_column_name), pl.col(f"new_{disease_id_column_name}")
            ).alias(disease_id_column_name),
            pl.coalesce(
                pl.col(drug_id_column_name), pl.col(f"new_{drug_id_column_name}")
            ).alias(drug_id_column_name),
        )
        .drop(f"new_{disease_id_column_name}", f"new_{drug_id_column_name}")
    )
