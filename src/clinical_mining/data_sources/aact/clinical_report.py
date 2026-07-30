"""Utils to transform AACT database to drug/indication relationships."""

from typing import Any

import polars as pl

from clinical_mining.dataset import ClinicalReport
from clinical_mining.schemas import ClinicalReportType, ClinicalSource
from clinical_mining.data_sources.pubmed import fetch_publications 
from clinical_mining.data_sources.aact.llm_extractor import parse_batch_results


def process_interventions(interventions: pl.DataFrame) -> pl.DataFrame:
    """Extract relevant MeSH terms for interventional studies."""
    INTERVENTION_WHITELIST = ["DRUG", "COMBINATION_PRODUCT", "BIOLOGICAL"]
    return (
        interventions.with_columns(drugFromSource=pl.col("name").str.to_lowercase())
        .filter(pl.col("intervention_type").is_in(INTERVENTION_WHITELIST))
        .filter(~pl.col("drugFromSource").str.starts_with("placebo"))
        .drop("intervention_type", "name")
        .unique()
    )


def process_conditions(conditions: pl.DataFrame) -> pl.DataFrame:
    """Extract relevant MeSH terms for conditions."""
    return (
        conditions.with_columns(
            diseaseFromSource=pl.col("downcase_name").str.to_lowercase()
        )
        .filter(~pl.col("diseaseFromSource").str.contains("healthy"))
        .drop("downcase_name")
        .unique()
    )


def replace_with_llm_indications(
    studies: pl.DataFrame,
    extractions: pl.DataFrame,
) -> pl.DataFrame:
    """Replace diseaseFromSource/drugFromSource with LLM-extracted indications.

    The *extractions* DataFrame must conform to the
    ``ClinicalReportExtractionSchema`` (``id``, ``primary_indications``,
    ``investigated_drugs``, …). Disease and drug names are extracted from
    the ``.name`` and ``.drug`` struct fields respectively.

    Trials not present in *extractions* get null indications.
    """
    llm_extracted = (
        extractions.select(
            nct_id=pl.col("id"),
            diseaseFromSource=pl.col("primary_indications").list.eval(
                pl.element().struct.field("name")
            ),
            drugFromSource=pl.col("investigated_drugs").list.eval(
                pl.element().struct.field("drug")
            ),
        )
        .with_columns(
            diseaseFromSource=pl.when(pl.col("diseaseFromSource").list.len() == 0)
            .then(pl.lit(None, dtype=pl.List(pl.String)))
            .otherwise(pl.col("diseaseFromSource")),
            drugFromSource=pl.when(pl.col("drugFromSource").list.len() == 0)
            .then(pl.lit(None, dtype=pl.List(pl.String)))
            .otherwise(pl.col("drugFromSource")),
        )
        .explode("diseaseFromSource", empty_as_null=True)
        .explode("drugFromSource", empty_as_null=True)
    )

    return pl.concat(
        [
            # Trials not covered by LLM extraction get null indications
            studies.join(
                llm_extracted.select("nct_id"),
                left_on=pl.col("nct_id").str.to_uppercase(),
                right_on=pl.col("nct_id").str.to_uppercase(),
                how="anti",
            ).with_columns(
                pl.lit(None, dtype=pl.String).alias("drugFromSource"),
                pl.lit(None, dtype=pl.String).alias("diseaseFromSource"),
            ),
            # Trials covered by LLM extraction get LLM annotations
            studies.drop(["diseaseFromSource", "drugFromSource"])
            .unique()  # to avoid ID duplication due to exploded diseases/drugs
            .join(
                llm_extracted,
                left_on=pl.col("nct_id").str.to_uppercase(),
                right_on=pl.col("nct_id").str.to_uppercase(),
                how="inner",
            )
            .select(studies.columns),  # to keep the same column order
        ]
    )


def extract_clinical_report(
    studies: pl.DataFrame,
    interventions: pl.DataFrame,
    conditions: pl.DataFrame,
    additional_metadata: list[pl.DataFrame] | None = None,
    aggregation_specs: dict[str, dict[str, Any]] | None = None,
    llm_extractions: pl.DataFrame | None = None,
) -> ClinicalReport:
    """Return clinical trials with desired extra annotations from other tables.

    Args:
        studies: DataFrame with study information.
        interventions: DataFrame with intervention information.
        conditions: DataFrame with condition information.
        additional_metadata: List of DataFrames with additional metadata.
        aggregation_specs: Dictionary with aggregation specifications.
        llm_extractions: DataFrame with LLM extractions.

    Returns:
        ClinicalReport with desired extra annotations from other tables.
    """
    STUDY_TYPES = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
    interventions = process_interventions(interventions)
    conditions = process_conditions(conditions)
    studies = studies.join(interventions, on="nct_id", how="left").join(
        conditions, on="nct_id", how="left"
    )
    if llm_extractions is not None:
        studies = replace_with_llm_indications(studies, llm_extractions)
    if additional_metadata is not None:
        for metadata_df in additional_metadata:
            if aggregation_specs:
                for key, spec in aggregation_specs.items():
                    if "struct" in spec and isinstance(spec["struct"], dict):
                        struct_cols = list(spec["struct"].values())
                        if all(col in metadata_df.columns for col in struct_cols):
                            struct_expr = pl.struct(
                                [
                                    pl.col(str(col)).alias(str(field))
                                    for field, col in spec["struct"].items()
                                ]
                            )
                            agg_op = spec.get("agg", "first")
                            if agg_op == "first":
                                expr = struct_expr.first()
                            elif agg_op == "unique":
                                expr = struct_expr.unique()
                            else:
                                expr = struct_expr

                            metadata_df = metadata_df.group_by(spec["group_by"]).agg(
                                expr.alias(spec["alias"])
                            )
                            break
                    elif key in metadata_df.columns:
                        metadata_df = metadata_df.group_by(spec["group_by"]).agg(
                            pl.col(key).alias(spec["alias"])
                        )
                        break
            studies = studies.join(metadata_df, on="nct_id", how="left")

    trial_metadata_cols = [
        c
        for c in studies.columns
        if c not in ["nct_id", "drugFromSource", "diseaseFromSource"]
    ]

    reports = (
        studies.filter(pl.col("study_type").is_in(STUDY_TYPES))
        .filter(pl.col("drugFromSource").is_not_null())
        .rename({col: f"trial_{col}" for col in trial_metadata_cols})
        .rename({"nct_id": "id"})
        .with_columns(
            trial_phase=pl.when(
                (pl.col("trial_phase").is_not_null()) & (pl.col("trial_phase") != "NA")
            )
            .then(pl.col("trial_phase"))
            .otherwise(pl.lit(None))
        )
        .with_columns(
            source=pl.lit(ClinicalSource.AACT.value),
            url=pl.concat_str(
                [pl.lit("https://clinicaltrials.gov/study/"), pl.col("id")],
                separator="",
            ),
            hasExpertReview=pl.lit(False),
            type=pl.lit(ClinicalReportType.CLINICAL_TRIAL),
            phaseFromSource=pl.col("trial_phase"),
            disease=pl.struct(
                pl.lit(None, dtype=pl.String).alias("diseaseId"),
                pl.col("diseaseFromSource"),
            ),
            drug=pl.struct(
                pl.col("drugFromSource"),
                pl.lit(None, dtype=pl.String).alias("drugId"),
            ),
        )
        .drop(["diseaseFromSource", "drugFromSource"])
        .unique()
    )
    return ClinicalReport(
        df=(
            reports.group_by(
                [c for c in reports.columns if c not in ["disease", "drug"]]
            ).agg(
                pl.col("disease").unique().alias("diseases"),
                pl.col("drug").unique().alias("drugs"),
            )
        )
    )


def build_aact_dataset(studies, interventions, conditions, study_references):
    llm_batch_results = parse_batch_results(
        "gs://open-targets-pipeline-runs/ds/26.06.0-dev2/input/clinical_report/aact_extraction_batch_results/"
    )
    llm_indications = llm_batch_results.df.select(
        'id',
        pl.col('investigated_drugs').list.eval(pl.element().struct.field('drug').unique()).alias('drugs'),
        pl.col('primary_indications').list.eval(pl.element().struct.field('name').unique()).alias('diseases'),
    )

    #only keep RESULT for study references
    if study_references is not None:
        if 'reference_type' in study_references.columns:
            study_references = study_references.filter(pl.col('reference_type') != 'BACKGROUND')

    clinical_report = extract_clinical_report(
        studies=studies,
        interventions=interventions,
        conditions=conditions,
        additional_metadata=[study_references],
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
        llm_extractions=llm_indications,
    )

    report_df = clinical_report.df.rename({"id": "nct_id"})

    if "reference_type" not in report_df.columns:
        if "trial_reference_type" in report_df.columns:
            report_df = report_df.rename({"trial_reference_type": "reference_type"})
        else:
            report_df = report_df.with_columns(pl.lit(None, dtype=pl.String).alias("reference_type"))

    if "trialDescription" not in report_df.columns:
        for candidate in ("trialDescription", "trial_description", "trial_trialDescription"):
            if candidate in report_df.columns:
                report_df = report_df.rename({candidate: "trialDescription"})
                break
        else:
            report_df = report_df.with_columns(pl.lit(None, dtype=pl.String).alias("trialDescription"))

    if "trialPhase" not in report_df.columns:
        for candidate in ("trialPhase", "trial_phase", "trial_trialPhase"):
            if candidate in report_df.columns:
                report_df = report_df.rename({candidate: "trialPhase"})
                break
        else:
            report_df = report_df.with_columns(pl.lit(None, dtype=pl.String).alias("trialPhase"))

    dataset = (
        report_df
        .explode("diseases")
        .explode("drugs")
        .with_columns(
            pl.col("diseases").struct.field("diseaseFromSource").alias("condition"),
            pl.col("drugs").struct.field("drugFromSource").alias("drug"),
        )
        .drop("diseases", "drugs")
        .filter(pl.col("drug").is_not_null())
        .explode("trialLiterature")
        .rename({"trialLiterature": "pmid"})
        .filter(pl.col("pmid").is_not_null())
        .select(["nct_id", "drug", "condition", "pmid", "reference_type", "trialDescription", "trialPhase"])
    )

    pmids = dataset.select(pl.col("pmid")).unique().to_series().drop_nulls().to_list()
    publications = fetch_publications(pmids)
    abstracts_df = pl.DataFrame([
        {"pmid": pmid, "abstract_text": publications.get(pmid, {}).get("abstractText", "")}
        for pmid in pmids
    ])

    dataset = dataset.join(abstracts_df, on="pmid", how="left")
    return dataset.select(["nct_id", "drug", "condition", "pmid", "abstract_text", "reference_type", "trialDescription", "trialPhase"])





