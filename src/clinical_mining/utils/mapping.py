from datetime import datetime

from loguru import logger
from ontoma import OnToma, OpenTargetsDisease, OpenTargetsDrug
from ontoma.ner.drug import extract_drug_entities
import polars as pl
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f

from clinical_mining.utils.polars_helpers import convert_polars_to_spark


def _normalise_label_expr(col_name: str) -> pl.Expr:
    """Normalise entity labels for robust matching."""
    return (
        pl.col(col_name)
        .cast(pl.String)
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.to_lowercase()
    )


def _apply_chembl_curation_mapping(
    df: pl.DataFrame,
    chembl_curation: pl.DataFrame,
    drug_column_name: str,
    disease_column_name: str,
    drug_id_column_name: str,
    disease_id_column_name: str,
) -> pl.DataFrame:
    """Map IDs directly from ChEMBL curation using study ID + label joins.

    For duplicate curated matches on the same key, keep all curated IDs and explode
    to one output row per curated ID combination.
    """
    required_df_cols = {"id", drug_column_name, disease_column_name}
    if not required_df_cols.issubset(df.columns):
        missing = sorted(required_df_cols.difference(df.columns))
        logger.warning(
            "skipping chembl curation join, missing df columns: {}",
            missing,
        )
        return df

    required_curation_cols = {
        "studyId",
        "drugFromSource",
        "drugId",
        "diseaseFromSource",
        "diseaseId",
    }
    if not required_curation_cols.issubset(chembl_curation.columns):
        missing = sorted(required_curation_cols.difference(chembl_curation.columns))
        logger.warning(
            "skipping chembl curation join, missing curation columns: {}",
            missing,
        )
        return df

    df_norm = df.with_columns(
        _normalise_label_expr("id").alias("__study_id_norm"),
        _normalise_label_expr(drug_column_name).alias("__drug_label_norm"),
        _normalise_label_expr(disease_column_name).alias("__disease_label_norm"),
    )

    curation_drug_lut = (
        chembl_curation.select("studyId", "drugFromSource", "drugId")
        .drop_nulls(["studyId", "drugFromSource", "drugId"])
        .with_columns(
            _normalise_label_expr("studyId").alias("__study_id_norm"),
            _normalise_label_expr("drugFromSource").alias("__drug_label_norm"),
            pl.col("drugId").alias("__curated_drug_id"),
        )
        .group_by(["__study_id_norm", "__drug_label_norm"])
        .agg(
            pl.col("__curated_drug_id")
            .drop_nulls()
            .unique()
            .sort()
            .alias("__curated_drug_ids")
        )
    )

    curation_disease_lut = (
        chembl_curation.select("studyId", "diseaseFromSource", "diseaseId")
        .drop_nulls(["studyId", "diseaseFromSource", "diseaseId"])
        .with_columns(
            _normalise_label_expr("studyId").alias("__study_id_norm"),
            _normalise_label_expr("diseaseFromSource").alias("__disease_label_norm"),
            pl.col("diseaseId").alias("__curated_disease_id"),
        )
        .group_by(["__study_id_norm", "__disease_label_norm"])
        .agg(
            pl.col("__curated_disease_id")
            .drop_nulls()
            .unique()
            .sort()
            .alias("__curated_disease_ids")
        )
    )

    joined = (
        df_norm.join(
            curation_drug_lut,
            on=["__study_id_norm", "__drug_label_norm"],
            how="left",
        )
        .join(
            curation_disease_lut,
            on=["__study_id_norm", "__disease_label_norm"],
            how="left",
        )
    )

    drug_filled_rows = joined.filter(
        pl.col(drug_id_column_name).is_null() & pl.col("__curated_drug_ids").is_not_null()
    ).height
    disease_filled_rows = joined.filter(
        pl.col(disease_id_column_name).is_null()
        & pl.col("__curated_disease_ids").is_not_null()
    ).height

    mapped = (
        joined.with_columns(
            pl.when(pl.col(drug_id_column_name).is_not_null())
            .then(pl.concat_list([pl.col(drug_id_column_name)]))
            .otherwise(pl.col("__curated_drug_ids"))
            .alias("__resolved_drug_ids"),
            pl.when(pl.col(disease_id_column_name).is_not_null())
            .then(pl.concat_list([pl.col(disease_id_column_name)]))
            .otherwise(pl.col("__curated_disease_ids"))
            .alias("__resolved_disease_ids"),
        )
        .with_columns(
            pl.when(pl.col("__resolved_drug_ids").is_null())
            .then(pl.lit([None], dtype=pl.List(pl.String)))
            .otherwise(pl.col("__resolved_drug_ids"))
            .alias("__resolved_drug_ids"),
            pl.when(pl.col("__resolved_disease_ids").is_null())
            .then(pl.lit([None], dtype=pl.List(pl.String)))
            .otherwise(pl.col("__resolved_disease_ids"))
            .alias("__resolved_disease_ids"),
        )
        .explode("__resolved_drug_ids")
        .explode("__resolved_disease_ids")
        .with_columns(
            pl.col("__resolved_drug_ids").alias(drug_id_column_name),
            pl.col("__resolved_disease_ids").alias(disease_id_column_name),
        )
        .drop(
            "__study_id_norm",
            "__drug_label_norm",
            "__disease_label_norm",
            "__curated_drug_ids",
            "__curated_disease_ids",
            "__resolved_drug_ids",
            "__resolved_disease_ids",
        )
    )

    logger.info(
        "chembl curation direct join filled drugId for {} rows and diseaseId for {} rows before explode",
        drug_filled_rows,
        disease_filled_rows,
    )

    return mapped


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

    This function uses a tiered mapping approach:
    1. Direct mapping from ChEMBL curation using study ID + source labels (when available)
    2. Dictionary mapping with Open Targets indices for remaining unmapped labels
    3. NER fallback for still-unmapped drug labels (optional)

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
    # Some upstream datasets may not yet have ID columns (e.g. `diseaseId`/`drugId`).
    # Ensure they exist before we attempt to coalesce them with newly mapped IDs.
    missing_id_cols: list[pl.Expr] = []
    if disease_id_column_name not in df.columns:
        missing_id_cols.append(pl.lit(None).cast(pl.String).alias(disease_id_column_name))
    if drug_id_column_name not in df.columns:
        missing_id_cols.append(pl.lit(None).cast(pl.String).alias(drug_id_column_name))
    if missing_id_cols:
        df = df.with_columns(*missing_id_cols)

    # Fill IDs from ChEMBL curation first using deterministic study+label joins.
    if chembl_curation is not None:
        df = _apply_chembl_curation_mapping(
            df=df,
            chembl_curation=chembl_curation,
            drug_column_name=drug_column_name,
            disease_column_name=disease_column_name,
            drug_id_column_name=drug_id_column_name,
            disease_id_column_name=disease_id_column_name,
        )

    # Build dictionary queries only for unresolved labels.
    disease_queries = (
        df.filter(
            pl.col(disease_column_name).is_not_null()
            & pl.col(disease_id_column_name).is_null()
        )
        .select(pl.col(disease_column_name).alias("query_label"))
        .with_columns(pl.lit("DS").alias("entity_type"))
        .unique()
    )
    drug_queries = (
        df.filter(pl.col(drug_column_name).is_not_null() & pl.col(drug_id_column_name).is_null())
        .select(pl.col(drug_column_name).alias("query_label"))
        .with_columns(pl.lit("CD").alias("entity_type"))
        .unique()
    )

    query_inputs = [q for q in [disease_queries, drug_queries] if q.height > 0]
    if not query_inputs:
        logger.info("all entities already mapped after curation and existing IDs")
        return df

    query_df = convert_polars_to_spark(pl.concat(query_inputs), spark).repartition(
        50, "entity_type"
    )

    # Create entity lookup tables
    disease_label_lut = OpenTargetsDisease.as_label_lut(disease_index)
    drug_label_lut = OpenTargetsDrug.as_label_lut(drug_index)
    lut_tables = [disease_label_lut, drug_label_lut]

    # STEP 1: Dictionary mapping with Open Targets indices
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

            cached_ner = None
            labels_to_process = unmapped_drugs

            if ner_cache_path is not None:
                try:
                    cached_ner = spark.read.parquet(ner_cache_path)
                    cached_labels = cached_ner.select("query_label").distinct()
                    labels_to_process = unmapped_drugs.join(
                        cached_labels, on="query_label", how="left_anti"
                    )
                    logger.info(f"loaded ner cache from: {ner_cache_path}")
                except Exception:
                    logger.info(f"no existing cache found at: {ner_cache_path}, will compute all labels")

            new_count = labels_to_process.count()

            if new_count > 0:
                logger.info(
                    f"applying ner to {new_count} new drug labels"
                    + (f" (using cache for {unmapped_count - new_count})" if cached_ner is not None else "")
                )
                new_ner_results = extract_drug_entities(
                    spark=spark,
                    df=labels_to_process,
                    input_col="query_label",
                    output_col="extracted_drugs",
                    use_regex=True,
                    use_biobert=True,
                    use_drugtemist=True,
                    batch_size=ner_batch_size,
                )
                ner_extracted_raw = (
                    cached_ner.union(new_ner_results) if cached_ner is not None else new_ner_results
                )

                if ner_cache_path is not None:
                    logger.info(f"updating cache: {ner_cache_path}")
                    ner_extracted_raw.toPandas().to_parquet(ner_cache_path)
            else:
                logger.info(f"all {unmapped_count} labels found in cache, skipping ner extraction")
                ner_extracted_raw = cached_ner

            if ner_extracted_raw is None:
                logger.info("ner cache returned no rows, skipping ner fallback")
                ner_aggregated = spark.createDataFrame([], "query_label string, ner_mapped_ids array<string>")
            else:
                assert ner_extracted_raw is not None
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
