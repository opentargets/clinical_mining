from datetime import datetime
from pathlib import Path

from loguru import logger
from ontoma import OnToma, OpenTargetsDisease, OpenTargetsDrug
from ontoma.dataset.raw_entity_lut import RawEntityLUT
from ontoma.ner.drug import extract_drug_entities
import polars as pl
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f

from clinical_mining.utils.polars_helpers import convert_polars_to_spark


def _create_curation_lut(
    chembl_curation_df, entity_id_col: str, entity_name_col: str, entity_type: str
) -> RawEntityLUT:
    """Create a curation lookup table for a specific entity type.

    Args:
        chembl_curation_df: Spark DataFrame with ChEMBL curation data
        entity_id_col: Column name for entity ID (e.g., 'diseaseId', 'drugId')
        entity_name_col: Column name for entity name (e.g., 'diseaseFromSource', 'drugFromSource')
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


def _prepare_unified_query_df(df, drug_column: str, disease_column: str):
    """Prepare a unified DataFrame with both drugs and diseases in a single column.

    Args:
        df: Spark DataFrame with a disease and a drug column
        drug_column: Name of the column containing drug names
        disease_column: Name of the column containing disease names

    Returns:
        Spark DataFrame with columns: query_label, entity_type
    """
    disease_df = (
        df.filter(f.col(disease_column).isNotNull())
        .select(
            f.col(disease_column).alias("query_label"), f.lit("DS").alias("entity_type")
        )
        .distinct()
    )
    drug_df = (
        df.filter(f.col(drug_column).isNotNull())
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
        Polars DataFrame with diseaseIds and drugIds columns
    """
    return (
        pl.from_pandas(
            mapped_spark_df.filter(
                (f.col("mapped_ids").isNotNull()) & (f.size("mapped_ids") > 0)
            ).toPandas()
        )
        .with_columns(
            [
                pl.when(pl.col("entity_type") == "DS")
                .then(pl.col("mapped_ids"))
                .alias("diseaseIds"),
                pl.when(pl.col("entity_type") == "CD")
                .then(pl.col("mapped_ids"))
                .alias("drugIds"),
            ]
        )
        .drop("mapped_ids")
    )


def map_entities(
    spark: SparkSession,
    df: pl.DataFrame,
    disease_index: DataFrame,
    drug_index: DataFrame,
    chembl_curation: pl.DataFrame | None = None,
    drug_column_name: str = "drugFromSource",
    disease_column_name: str = "diseaseFromSource",
    drug_id_column_name: str = "drugId",
    disease_id_column_name: str = "diseaseId",
    ner_extract_drug: bool = True,
    ner_batch_size: int = 256,
    ner_cache_path: str | None = f".cache/ner/{datetime.now().strftime('%Y%m%d')}.parquet",
) -> pl.DataFrame:
    """Map drug and disease entities to their standardised IDs using OnToma.

    This function uses a tiered approach for drug mapping:
    1. First attempt: Dictionary mapping using curation and drug indices
    2. Second attempt (if ner_extract_drug=True): NER extraction for unmapped drugs

    Disease mapping always uses dictionary approach only.

    Args:
        spark (SparkSession): Spark session for OnToma
        df: Polars DataFrame with drugs and diseases to map
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
        pl.DataFrame: Polars DataFrame with a diseaseId and drugId populated if there is an OnToma match
    """
    # Prepare unified query DataFrame and with essential columns for OnToma processing
    query_df = _prepare_unified_query_df(
        convert_polars_to_spark(
            df.select(drug_column_name, disease_column_name).unique(),
            spark,
        ),
        drug_column_name,
        disease_column_name,
    ).repartition(50, "entity_type")

    # Create entity lookup tables
    disease_label_lut = OpenTargetsDisease.as_label_lut(disease_index)
    drug_label_lut = OpenTargetsDrug.as_label_lut(drug_index)
    lut_tables = [disease_label_lut, drug_label_lut]

    # Create curation lookup tables using the helper function
    if chembl_curation is not None:
        chembl_curation_spark = convert_polars_to_spark(chembl_curation, spark)
        curation_lut_disease = _create_curation_lut(
            chembl_curation_spark, "diseaseId", "diseaseFromSource", "DS"
        )
        curation_lut_drug = _create_curation_lut(
            chembl_curation_spark, "drugId", "drugFromSource", "CD"
        )
        lut_tables.extend([curation_lut_disease, curation_lut_drug])

    # STEP 1: Dictionary mapping (curation + indices)
    # This maps ~50% of drugs and diseases
    ontoma = OnToma(
        spark=spark,
        entity_lut_list=lut_tables,
    )
    mapping_results = ontoma.map_entities(
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
            mapping_results.filter(
                (f.col("entity_type") == "CD")
                & (f.col("mapped_ids").isNull() | (f.size("mapped_ids") == 0))
            )
            .select("query_label")
            .distinct()
        )

        unmapped_count = unmapped_drugs.count()
        if unmapped_count > 0:
            logger.info(f"found {unmapped_count} unmapped drug labels...")
            cache_file = Path(ner_cache_path) if ner_cache_path else None

            if cache_file is not None and cache_file.exists():
                logger.info(f"loading ner cache from: {cache_file}")
                cached_ner = spark.read.parquet(str(cache_file))
                cached_labels = cached_ner.select("query_label").distinct()

                # Find new labels not in cache
                new_labels = unmapped_drugs.join(
                    cached_labels, on="query_label", how="left_anti"
                )
                new_count = new_labels.count()

                if new_count > 0:
                    logger.info(
                        f"applying ner to {new_count} new drug labels (using cache for {unmapped_count - new_count})..."
                    )
                    new_ner_results = extract_drug_entities(
                        spark=spark,
                        df=new_labels,
                        input_col="query_label",
                        output_col="extracted_drugs",
                        use_regex=True,
                        use_biobert=True,
                        use_drugtemist=True,
                        batch_size=ner_batch_size,
                    )

                    # Update cache with merged results
                    logger.info(f"updating cache: {cache_file}")
                    ner_extracted_raw = cached_ner.union(new_ner_results)
                    ner_extracted_raw.toPandas().to_parquet(str(cache_file))
                else:
                    logger.info(
                        f"all {unmapped_count} labels found in cache, skipping ner extraction"
                    )
                    ner_extracted_raw = cached_ner
            else:
                logger.info(
                    f"no cache found, applying ner to all {unmapped_count} drug labels..."
                )
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

                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"saving ner cache to: {cache_file}")
                    ner_extracted_raw.toPandas().to_parquet(str(cache_file))

            # Explode extracted drugs
            ner_extracted = ner_extracted_raw.filter(
                f.size("extracted_drugs") > 0
            ).select(
                f.col("query_label"),
                f.explode("extracted_drugs").alias("clean_label"),
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
            mapping_results = (
                mapping_results.alias("dict")
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
            logger.info(
                f"ner recovered {recovered}/{unmapped_count} ({recovered / unmapped_count * 100:.1f}%)"
            )

    # Convert to Polars and join back
    polars_mapping_results = _process_mapping_results(mapping_results)

    # Some upstream datasets may not yet have ID columns (e.g. `diseaseId`/`drugId`).
    # Ensure they exist before we attempt to coalesce them with newly mapped IDs.
    missing_id_cols: list[pl.Expr] = []
    if disease_id_column_name not in df.columns:
        missing_id_cols.append(pl.lit(None).cast(pl.String).alias(disease_id_column_name))
    if drug_id_column_name not in df.columns:
        missing_id_cols.append(pl.lit(None).cast(pl.String).alias(drug_id_column_name))
    if missing_id_cols:
        df = df.with_columns(*missing_id_cols)

    return (
        df.join(
            # Add mapped disease IDs
            polars_mapping_results.filter(pl.col("entity_type") == "DS").select(
                ["query_label", "diseaseIds"]
            ),
            left_on=disease_column_name,
            right_on="query_label",
            how="left",
            suffix="_disease",
        )
        .join(
            # Add mapped drug IDs
            polars_mapping_results.filter(pl.col("entity_type") == "CD").select(
                ["query_label", "drugIds"]
            ),
            left_on=drug_column_name,
            right_on="query_label",
            how="left",
            suffix="_drug",
        )
        .explode("diseaseIds")
        .explode("drugIds")
        .rename(
            {
                "diseaseIds": f"new_{disease_id_column_name}",
                "drugIds": f"new_{drug_id_column_name}",
            }
        )
        .with_columns(
            pl.coalesce(
                pl.col(disease_id_column_name),
                pl.col(f"new_{disease_id_column_name}"),
            ).alias(disease_id_column_name),
            pl.coalesce(
                pl.col(drug_id_column_name), pl.col(f"new_{drug_id_column_name}")
            ).alias(drug_id_column_name),
        )
        .drop(f"new_{disease_id_column_name}", f"new_{drug_id_column_name}")
    )
