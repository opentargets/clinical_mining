"""Print a validation report for LLM extraction results.

Usage:
    uv run scripts/validation_report.py \
      --extraction data/outputs/llm_extraction/extraction_2026-04-27.jsonl \
      --input data/outputs/clinical_report/2026-02-02/clinical_report.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_mining.schemas import ClinicalReportExtraction

EXCIPIENT_BLOCKLIST: frozenset[str] = frozenset({
    "placebo", "vehicle", "saline", "normal saline", "sodium chloride",
    "dextrose", "dmso", "dimethyl sulfoxide", "water for injection",
    "sesame oil", "corn oil", "olive oil", "peanut oil", "soybean oil",
    "cremophor", "polysorbate 80", "tween 80", "methylcellulose",
    "carboxymethylcellulose", "cmc", "hydroxypropyl methylcellulose", "hpmc",
    "lactose", "microcrystalline cellulose", "magnesium stearate",
})


def load_extractions(extraction_path: str) -> list[ClinicalReportExtraction]:
    """Parse a JSONL file of ClinicalReportExtraction records."""
    extractions = []
    with Path(extraction_path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                extractions.append(ClinicalReportExtraction.model_validate_json(line))
    return extractions


def load_input_df(input_path: str) -> pl.DataFrame:
    """Read parquet and filter to CLINICAL_TRIAL rows."""
    return pl.read_parquet(input_path).filter(pl.col("type") == "CLINICAL_TRIAL")


def compute_drug_coverage(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> dict:
    """Count records with ≥1 drug in input and ≥1 investigated_drug in output."""
    extracted_ids = {ext.id for ext in extractions}
    input_rows = {
        row["id"]: row
        for row in input_df.iter_rows(named=True)
        if row["id"] in extracted_ids
    }
    total = len(extractions)
    input_with_drug = sum(
        1
        for ext in extractions
        if any(d.get("drugFromSource") for d in (input_rows.get(ext.id, {}).get("drugs") or []))
    )
    output_with_drug = sum(
        1 for ext in extractions if ext.investigated_drugs
    )
    return {
        "input": input_with_drug,
        "output": output_with_drug,
        "total": total,
        "input_pct": input_with_drug / total * 100 if total else 0.0,
        "output_pct": output_with_drug / total * 100 if total else 0.0,
    }


def compute_disease_coverage(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> dict:
    """Count records with ≥1 disease in input and a primary_indication in output."""
    extracted_ids = {ext.id for ext in extractions}
    input_rows = {
        row["id"]: row
        for row in input_df.iter_rows(named=True)
        if row["id"] in extracted_ids
    }
    total = len(extractions)
    input_with_disease = sum(
        1
        for ext in extractions
        if any(d.get("diseaseFromSource") for d in (input_rows.get(ext.id, {}).get("diseases") or []))
    )
    output_with_disease = sum(
        1 for ext in extractions if ext.primary_indications
    )
    return {
        "input": input_with_disease,
        "output": output_with_disease,
        "total": total,
        "input_pct": input_with_disease / total * 100 if total else 0.0,
        "output_pct": output_with_disease / total * 100 if total else 0.0,
    }


def compute_drugs_by_stage(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> pl.DataFrame:
    """Count total investigated_drugs per clinical stage."""
    if not extractions:
        return pl.DataFrame({"clinicalStage": [], "total_investigated_drugs": []})
    counts = pl.DataFrame({
        "id": [ext.id for ext in extractions],
        "num_drugs": [len(ext.investigated_drugs) for ext in extractions],
    })
    return (
        counts
        .join(input_df.select(["id", "clinicalStage"]).unique(subset="id"), on="id", how="left")
        .group_by("clinicalStage")
        .agg(pl.col("num_drugs").sum().alias("total_investigated_drugs"))
        .sort("clinicalStage")
    )


def compute_quote_grounding(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> dict:
    """Check what fraction of evidence_quotes appear verbatim in the source trial text.

    Concatenates all text fields from the input row and checks if each quote
    is a case-insensitive substring. Returns per-extraction and aggregate stats.
    """
    text_fields = [
        "trialOfficialTitle", "trialDescription", "trialDetailedDescription",
    ]
    available = [f for f in text_fields if f in input_df.columns]
    input_lookup = {row["id"]: row for row in input_df.iter_rows(named=True)}

    total_quotes = 0
    grounded_quotes = 0
    ungrounded: list[dict] = []

    for ext in extractions:
        src = input_lookup.get(ext.id, {})
        source_text = " ".join(
            str(src.get(f, "") or "") for f in available
        ).lower()

        all_quotes: list[tuple[str, str | None]] = []
        for pi in ext.primary_indications:
            all_quotes.append(("primary_indication", pi.evidence_quote))
        for bc in (ext.background_conditions or []):
            all_quotes.append(("background_condition", bc.evidence_quote))
        for d in ext.investigated_drugs:
            all_quotes.append(("investigated_drug", d.evidence_quote))
        for d in (ext.comparator_drugs or []):
            all_quotes.append(("comparator_drug", d.evidence_quote))
        for d in (ext.supportive_drugs or []):
            all_quotes.append(("supportive_drug", d.evidence_quote))

        for field, quote in all_quotes:
            if quote is None:
                continue  # background_conditions may omit evidence_quote
            total_quotes += 1
            if quote.lower() in source_text:
                grounded_quotes += 1
            else:
                ungrounded.append({"id": ext.id, "field": field, "quote": quote})

    return {
        "total_quotes": total_quotes,
        "grounded": grounded_quotes,
        "ungrounded": total_quotes - grounded_quotes,
        "grounded_pct": grounded_quotes / total_quotes * 100 if total_quotes else 0.0,
        "ungrounded_examples": ungrounded[:10],
    }


def compute_excipient_hits(
    extractions: list[ClinicalReportExtraction],
) -> list[dict]:
    """Return investigated_drugs whose name matches a known excipient or placebo."""
    hits = []
    for ext in extractions:
        for drug in ext.investigated_drugs:
            if drug.drug.lower().strip() in EXCIPIENT_BLOCKLIST:
                hits.append({"id": ext.id, "drug": drug.drug})
    return hits


def build_comparison_df(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build a per-report DataFrame comparing input and LLM-extracted drugs and diseases."""
    input_lookup = {row["id"]: row for row in input_df.iter_rows(named=True)}

    rows = []
    for ext in extractions:
        src = input_lookup.get(ext.id, {})

        input_drugs = "; ".join(
            d["drugFromSource"]
            for d in (src.get("drugs") or [])
            if d.get("drugFromSource")
        )
        input_diseases = "; ".join(
            d["diseaseFromSource"]
            for d in (src.get("diseases") or [])
            if d.get("diseaseFromSource")
        )
        input_drug_labels = [
            d["drugFromSource"].lower()
            for d in (src.get("drugs") or [])
            if d.get("drugFromSource")
        ]
        output_drug_labels = [d.drug.lower() for d in ext.investigated_drugs]
        has_unmatched_drug = not any(
            a in b or b in a
            for a in input_drug_labels
            for b in output_drug_labels
        ) if input_drug_labels and output_drug_labels else bool(input_drug_labels) != bool(output_drug_labels)

        input_disease_labels = [
            d["diseaseFromSource"].lower()
            for d in (src.get("diseases") or [])
            if d.get("diseaseFromSource")
        ]
        output_disease_labels = [pi.name.lower() for pi in ext.primary_indications]
        has_unmatched_disease = not any(
            a in b or b in a
            for a in input_disease_labels
            for b in output_disease_labels
        ) if input_disease_labels and output_disease_labels else bool(input_disease_labels) != bool(output_disease_labels)

        output_investigated = "; ".join(d.drug for d in ext.investigated_drugs)
        output_drug_modifier = "; ".join(
            ", ".join(filter(None, [d.route, d.formulation]))
            for d in ext.investigated_drugs
            if any([d.route, d.formulation])
        )
        output_comparators = "; ".join(d.drug for d in (ext.comparator_drugs or []))
        output_supportive = "; ".join(d.drug for d in (ext.supportive_drugs or []))
        output_disease = "; ".join(pi.name for pi in ext.primary_indications)
        output_disease_modifier = "; ".join(
            ", ".join(filter(None, [pi.severity, pi.stage, pi.onset, pi.etiology]))
            for pi in ext.primary_indications
            if any([pi.severity, pi.stage, pi.onset, pi.etiology])
        )
        output_background = "; ".join(
            bc.name for bc in (ext.background_conditions or [])
        )
        output_conclusion = ext.conclusion or ""

        rows.append({
            "id": ext.id,
            "drug_intent": ext.drug_intent,
            "drug_intent_confidence": ext.drug_intent_confidence,
            "input_drugs": input_drugs,
            "output_investigated_drugs": output_investigated,
            "output_drug_modifier": output_drug_modifier,
            "output_comparator_drugs": output_comparators,
            "output_supportive_drugs": output_supportive,
            "has_unmatched_drug": has_unmatched_drug,
            "input_diseases": input_diseases,
            "output_primary_indication": output_disease,
            "output_disease_modifier": output_disease_modifier,
            "output_background_conditions": output_background,
            "has_unmatched_disease": has_unmatched_disease,
            "output_conclusion": output_conclusion,
        })

    return pl.DataFrame(rows, schema={
        "id": pl.String,
        "drug_intent": pl.String,
        "drug_intent_confidence": pl.Float64,
        "input_drugs": pl.String,
        "output_investigated_drugs": pl.String,
        "output_drug_modifier": pl.String,
        "output_comparator_drugs": pl.String,
        "output_supportive_drugs": pl.String,
        "has_unmatched_drug": pl.Boolean,
        "input_diseases": pl.String,
        "output_primary_indication": pl.String,
        "output_disease_modifier": pl.String,
        "output_background_conditions": pl.String,
        "has_unmatched_disease": pl.Boolean,
        "output_conclusion": pl.String,
    })


def search_drug_disease_plausibility(drug: str, disease: str) -> bool:
    """Query DuckDuckGo Instant Answer API to check if drug-disease pair is plausible."""
    query = f"{drug} {disease} clinical trial"
    response = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_redirect": "1"},
        timeout=10.0,
    )
    try:
        data = response.json()
    except Exception:
        return False
    return bool(data.get("Abstract")) or bool(data.get("RelatedTopics"))


def print_validation_report(
    extractions: list[ClinicalReportExtraction],
    input_df: pl.DataFrame,
    errors: list[dict],
) -> None:
    """Print the validation report to stdout."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    total = len(extractions)
    print(f"\nTotal extracted records: {total}")
    print(f"Total errors: {len(errors)}")

    from collections import Counter
    intent_counts = Counter(ext.drug_intent for ext in extractions)
    print("\n[0] drug_intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda kv: -kv[1]):
        avg_conf = (
            sum(e.drug_intent_confidence for e in extractions if e.drug_intent == intent)
            / count
        )
        print(f"    {intent}: {count} (mean confidence: {avg_conf:.2f})")

    low_conf = sorted(
        [(e.id, e.drug_intent, e.drug_intent_confidence) for e in extractions],
        key=lambda t: t[2],
    )[:10]
    print("\n    Lowest-confidence extractions (review candidates):")
    for nct, intent, conf in low_conf:
        print(f"      {conf:.2f}  {intent:15s}  {nct}")

    drug_stats = compute_drug_coverage(extractions, input_df)
    print(
        f"\n[1] Drug coverage (investigated_drugs):\n"
        f"    Input  (≥1 drugFromSource): {drug_stats['input']}/{drug_stats['total']} ({drug_stats['input_pct']:.1f}%)\n"
        f"    Output (≥1 investigated):   {drug_stats['output']}/{drug_stats['total']} ({drug_stats['output_pct']:.1f}%)"
    )

    disease_stats = compute_disease_coverage(extractions, input_df)
    print(
        f"\n[2] Disease coverage (primary_indication):\n"
        f"    Input  (≥1 diseaseFromSource): {disease_stats['input']}/{disease_stats['total']} ({disease_stats['input_pct']:.1f}%)\n"
        f"    Output (primary_indication):   {disease_stats['output']}/{disease_stats['total']} ({disease_stats['output_pct']:.1f}%)"
    )

    stage_df = compute_drugs_by_stage(extractions, input_df)
    print("\n[3] Investigated drugs by clinical stage:")
    for row in stage_df.iter_rows(named=True):
        print(f"    {row['clinicalStage']}: {row['total_investigated_drugs']}")

    quote_stats = compute_quote_grounding(extractions, input_df)
    print(
        f"\n[4] Quote grounding:\n"
        f"    Grounded: {quote_stats['grounded']}/{quote_stats['total_quotes']} "
        f"({quote_stats['grounded_pct']:.1f}%)\n"
        f"    Ungrounded (first 10):"
    )
    for ex in quote_stats["ungrounded_examples"]:
        print(f"      [{ex['id']} / {ex['field']}] {ex['quote'][:80]!r}")

    excipient_hits = compute_excipient_hits(extractions)
    print(f"\n[5] Excipient/placebo hits in investigated_drugs: {len(excipient_hits)}")
    for hit in excipient_hits[:10]:
        print(f"    [{hit['id']}] {hit['drug']!r}")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a validation report for LLM extraction results."
    )
    parser.add_argument(
        "--extraction", required=True, help="Path to extraction JSONL file"
    )
    parser.add_argument(
        "--input", required=True, help="Local parquet file or directory (source data)"
    )
    args = parser.parse_args()

    extractions = load_extractions(args.extraction)
    df = load_input_df(args.input)

    print_validation_report(extractions, df, errors=[])

    csv_path = Path(args.extraction).with_suffix(".csv")
    build_comparison_df(extractions, df).write_csv(csv_path)
    print(f"Comparison CSV written to {csv_path}")


if __name__ == "__main__":
    main()
