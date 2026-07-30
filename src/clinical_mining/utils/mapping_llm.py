"""Map drug/disease labels to Open Targets IDs.

Input is the output of the reshape step: one row per
(trialId, drugFromSource, diseaseFromSource, primary_endpoint), with
`drugId` / `diseaseId` present but null. Combination labels have already been
split upstream, so NER is used here only to normalise each component label to
an entity mention before lookup — not to decompose it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pyspark.sql.functions as f
from loguru import logger
from ontoma import OnToma, OpenTargetsDisease, OpenTargetsDrug
from ontoma.ner.disease import extract_disease_entities
from ontoma.ner.drug import extract_drug_entities
from pyspark.sql import SparkSession

from clinical_mining.utils.polars_helpers import convert_polars_to_spark
from clinical_mining.utils.spark_helpers import spark_session

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

OUTPUT_COLUMNS = [
    "evidenceId",
    "trialId",
    "drugFromSource",
    "drugId",
    "diseaseFromSource",
    "diseaseId",
    "primary_endpoint",
    "population",
    "evidence_quote",
    "source_pmid",
    "conclusion",
    "conclusion_axis",
    "conclusion_direction",
    "conclusion_confidence",
    "is_contested",
]


def load_ot_indices(spark: SparkSession, disease_index_path: str, drug_index_path: str):
    disease_index = spark.read.parquet(disease_index_path)
    drug_index = spark.read.parquet(drug_index_path)
    return disease_index, drug_index


def _normalize_labels(
    spark: SparkSession,
    labels: pl.DataFrame,
    label_col: str,
    ner_extractor_fn,
    ner_cache_path: str | None,
    **ner_kwargs,
):
    """Run NER over distinct labels; return [label_col, normalizedLabel], one row per mention.

    A label with 0 recognised entities (drug *classes* like "antihistamine",
    unusual phrasings) falls back to itself, so nothing is dropped — it just
    goes to lookup unnormalised, as before.
    """
    labels_spark = convert_polars_to_spark(labels, spark)
    labels_to_process = labels_spark.select(label_col).distinct()
    cached_ner = None

    if ner_cache_path is not None:
        try:
            cached_ner = spark.read.parquet(ner_cache_path)
            labels_to_process = labels_to_process.join(
                cached_ner.select(label_col).distinct(), on=label_col, how="left_anti"
            )
        except Exception:
            logger.info("no NER cache found at: {}", ner_cache_path)

    if labels_to_process.count() > 0:
        new_ner_results = ner_extractor_fn(
            spark=spark,
            df=labels_to_process,
            input_col=label_col,
            output_col="normalizedLabel",
            **ner_kwargs,
        )
        ner_raw = (
            cached_ner.unionByName(
                new_ner_results.select(label_col, "normalizedLabel")
            )
            if cached_ner is not None
            else new_ner_results
        )
        if ner_cache_path is not None:
            Path(ner_cache_path).parent.mkdir(parents=True, exist_ok=True)
            ner_raw.select(label_col, "normalizedLabel").toPandas().to_parquet(
                ner_cache_path
            )
    elif cached_ner is not None:
        ner_raw = cached_ner
    else:
        ner_raw = spark.createDataFrame(
            [], f"{label_col} string, normalizedLabel array<string>"
        )

    zero_entity = ner_raw.filter(f.size("normalizedLabel") == 0).count()
    if zero_entity > 0:
        logger.warning(
            "{} distinct {} labels had 0 entities recognised by NER — "
            "falling back to the raw label for lookup",
            zero_entity,
            label_col,
        )

    fallback = f.array(f.col(label_col))
    return (
        labels_spark.join(
            ner_raw.select(label_col, "normalizedLabel").distinct(),
            on=label_col,
            how="left",
        )
        .withColumn(
            "normalizedLabel",
            f.when(
                f.col("normalizedLabel").isNull() | (f.size("normalizedLabel") == 0),
                fallback,
            ).otherwise(f.col("normalizedLabel")),
        )
        .select(label_col, f.explode("normalizedLabel").alias("normalizedLabel"))
        .distinct()
    )


def _map_labels(
    spark: SparkSession,
    trials: pl.DataFrame,
    label_col: str,
    id_col: str,
    entity_type: str,
    label_lut,
    ner_extractor_fn,
    ner_cache_path: str | None,
    **ner_kwargs,
) -> pl.DataFrame:
    """NER-normalise then look up distinct labels; return [label_col, id_col: List[String]]."""
    null_count = trials.select(pl.col(label_col).is_null().sum()).item()
    if null_count > 0:
        logger.warning("{} rows have null {}", null_count, label_col)

    labels = trials.select(label_col).drop_nulls().unique()
    normalized = _normalize_labels(
        spark=spark,
        labels=labels,
        label_col=label_col,
        ner_extractor_fn=ner_extractor_fn,
        ner_cache_path=ner_cache_path,
        **ner_kwargs,
    )

    ontoma = OnToma(spark=spark, entity_lut_list=[label_lut])
    mapped = ontoma.map_entities(
        df=normalized.select("normalizedLabel")
        .distinct()
        .withColumn("entity_type", f.lit(entity_type)),
        result_col_name="mapped_ids",
        entity_col_name="normalizedLabel",
        entity_kind="label",
        type_col=f.col("entity_type"),
    )

    by_label = (
        normalized.join(
            mapped.select("normalizedLabel", "mapped_ids").distinct(),
            on="normalizedLabel",
            how="left",
        )
        .withColumn(
            "mapped_ids",
            f.when(f.col("mapped_ids").isNotNull(), f.col("mapped_ids")).otherwise(
                f.array()
            ),
        )
        .groupBy(label_col)
        .agg(
            f.filter(
                f.array_distinct(f.flatten(f.collect_list("mapped_ids"))),
                lambda x: x.isNotNull(),
            ).alias(id_col)
        )
    )

    total = by_label.count()
    unmapped = by_label.filter(f.size(id_col) == 0).count()
    if unmapped > 0:
        logger.warning(
            "{}/{} distinct {} labels did not map to any {} "
            "(often classes rather than named entities)",
            unmapped,
            total,
            label_col,
            id_col,
        )

    ambiguous = by_label.filter(f.size(id_col) > 1).count()
    if ambiguous > 0:
        logger.warning(
            "{}/{} distinct {} labels mapped to >1 candidate ID; exploding to one row per ID",
            ambiguous,
            total,
            label_col,
        )

    return pl.from_pandas(by_label.toPandas())


def _explodable(col: str) -> pl.Expr:
    """Empty/missing ID list -> [null], so unmapped rows survive the explode."""
    return (
        pl.when(pl.col(col).is_null() | (pl.col(col).list.len() == 0))
        .then(pl.lit([None], dtype=pl.List(pl.String)))
        .otherwise(pl.col(col))
        .alias(col)
    )


def _recompute_is_contested(df: pl.DataFrame) -> pl.DataFrame:
    """Re-derive the contested flag on the post-mapping grain.

    Upstream it was computed over drugId/diseaseId while both were still null,
    so synonymous labels ("aspirin" / "acetylsalicylic acid") sat in separate
    partitions and any disagreement between them was invisible. Coalescing to
    the source label keeps unmapped rows from collapsing into a single null
    bucket per trial.
    """
    return (
        df.with_columns(
            _drug_key=pl.coalesce([pl.col("drugId"), pl.col("drugFromSource")]),
            _disease_key=pl.coalesce([pl.col("diseaseId"), pl.col("diseaseFromSource")]),
        )
        .with_columns(
            is_contested=(
                pl.col("conclusion")
                .drop_nulls()
                .n_unique()
                .over(["trialId", "_drug_key", "_disease_key", "primary_endpoint"])
                > 1
            ).fill_null(False)
        )
        .drop("_drug_key", "_disease_key")
    )


def map_clinical_trial_entities(
    spark: SparkSession,
    trials: pl.DataFrame,
    disease_index,
    drug_index,
    ner_batch_size: int = 256,
    ner_cache_path: str | None = None,
    ner_cache_path_disease: str | None = None,
) -> pl.DataFrame:
    required_cols = {"trialId", "drugFromSource", "diseaseFromSource"}
    missing = required_cols.difference(trials.columns)
    if missing:
        raise ValueError(f"trials df is missing required columns: {sorted(missing)}")

    if ner_cache_path is None:
        ner_cache_path = str(
            _PROJECT_ROOT
            / ".cache"
            / "ner"
            / f"clinical_trials_drug_{datetime.now().strftime('%Y%m%d')}.parquet"
        )
    if ner_cache_path_disease is None:
        ner_cache_path_disease = str(
            _PROJECT_ROOT
            / ".cache"
            / "ner"
            / f"clinical_trials_disease_{datetime.now().strftime('%Y%m%d')}.parquet"
        )

    trials = trials.unique(maintain_order=True)

    drug_mapped = _map_labels(
        spark=spark,
        trials=trials,
        label_col="drugFromSource",
        id_col="drugId",
        entity_type="CD",
        label_lut=OpenTargetsDrug.as_label_lut(drug_index),
        ner_extractor_fn=extract_drug_entities,
        ner_cache_path=ner_cache_path,
        use_regex=True,
        use_biobert=True,
        use_drugtemist=True,
        batch_size=ner_batch_size,
    )
    disease_mapped = _map_labels(
        spark=spark,
        trials=trials,
        label_col="diseaseFromSource",
        id_col="diseaseId",
        entity_type="DS",
        label_lut=OpenTargetsDisease.as_label_lut(disease_index),
        ner_extractor_fn=extract_disease_entities,
        ner_cache_path=ner_cache_path_disease,
    )

    result = (
        trials.drop("drugId", "diseaseId", strict=False)
        .with_row_index("evidenceId")
        .join(drug_mapped.unique(subset=["drugFromSource"]), on="drugFromSource", how="left")
        .join(
            disease_mapped.unique(subset=["diseaseFromSource"]),
            on="diseaseFromSource",
            how="left",
        )
        .with_columns(_explodable("drugId"), _explodable("diseaseId"))
        .explode("drugId")
        .explode("diseaseId")
        .unique(maintain_order=True)
        .pipe(_recompute_is_contested)
    )

    ordered_cols = [c for c in OUTPUT_COLUMNS if c in result.columns]
    extras = [c for c in result.columns if c not in ordered_cols]
    return result.select([*ordered_cols, *extras])


if __name__ == "__main__":
    spark = spark_session()

    disease_index, drug_index = load_ot_indices(
        spark,
        disease_index_path=str(_PROJECT_ROOT / "disease.parquet"),
        drug_index_path=str(_PROJECT_ROOT / "drug.parquet"),
    )

    trials = pl.read_parquet(
        str(_PROJECT_ROOT / "extracted_results_reshaped.parquet")
    )

    mapped = map_clinical_trial_entities(
        spark=spark,
        trials=trials,
        disease_index=disease_index,
        drug_index=drug_index,
    )

    output_path = _PROJECT_ROOT / "mapped_trials.parquet"
    mapped.write_parquet(str(output_path))
    logger.info("wrote {} rows to {}", mapped.height, output_path)
    spark.stop()