import polars as pl

# conclusion -> (axis, direction)
CONCLUSION_MAP: dict[str, tuple[str | None, str | None]] = {
    "efficacy":            ("efficacy",      "positive"),
    "lack_of_efficacy":    ("efficacy",      "negative"),
    "effectiveness":       ("effectiveness", "positive"),
    "ineffectiveness":     ("effectiveness", "negative"),
    "safety_tolerability": ("safety",        "positive"),
    "safety_failure":      ("safety",        "negative"),
    "mixed":               (None,            "mixed"),
    "inconclusive":        (None,            "inconclusive"),
    "no_results":          (None,            "not_reported"),
}

AXIS_MAP = {k: v[0] for k, v in CONCLUSION_MAP.items()}
DIRECTION_MAP = {k: v[1] for k, v in CONCLUSION_MAP.items()}

# The columns that should uniquely identify a row:
# "in this trial, drug A showed X when evaluating endpoint Y for disease B".
UNIQUE_KEY = [
    "trialId",
    "drugFromSource",
    "drugId",
    "diseaseFromSource",
    "diseaseId",
    "primary_endpoint",
]

TARGET_SCHEMA = {
    "trialId": pl.String,
    "drugFromSource": pl.String,
    "drugId": pl.String,
    "diseaseFromSource": pl.String,
    "diseaseId": pl.String,
    "primary_endpoint": pl.String,
    "population": pl.String,
    "evidence_quote": pl.String,
    "source_pmid": pl.String,
    "conclusion": pl.String,
    "conclusion_axis": pl.String,
    "conclusion_direction": pl.String,
    "conclusion_confidence": pl.Float64,
    "is_contested": pl.Boolean,
}

RENAMES = {
    "id": "trialId",
    "therapies": "drugFromSource",
    "conditions": "diseaseFromSource",
    "outcome_confidence": "conclusion_confidence",
}


def _names(col: str) -> pl.Expr:
    """List[struct] -> ' + '-joined name string."""
    return (
        pl.col(col)
        .list.eval(pl.element().struct.field("name"))
        .list.join(" + ")
        .alias(col)
    )


def _split_trim(col: str) -> pl.Expr:
    """Split on '+', strip each fragment, drop empties left by stray separators."""
    return (
        pl.col(col)
        .str.split("+")
        .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
    )


def _is_contested() -> pl.Expr:
    """True where one trial/drug/disease/endpoint carries >1 distinct conclusion.

    Nulls are ignored, so a missing conclusion alongside a real one is not
    treated as disagreement. Repeats of the same conclusion are duplicates,
    not contested, so this counts distinct values rather than rows.
    """
    return (
        (pl.col("conclusion").drop_nulls().n_unique().over(UNIQUE_KEY) > 1)
        .fill_null(False)
        .alias("is_contested")
    )


def parse_raw(raw: pl.DataFrame) -> pl.DataFrame:
    """Flatten the nested source: one row per outcome, therapies/conditions as strings."""
    return (
        raw
        .explode("outcomes")
        .unnest("outcomes")
        .with_columns(
            _names("therapies"),
            _names("conditions"),
        )
        .drop("key_reconciliation")
    )


def reshape(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.rename(RENAMES)
        .with_columns(
            _split_trim("drugFromSource"),
            _split_trim("diseaseFromSource"),
        )
        .explode("drugFromSource")
        .explode("diseaseFromSource")
        .with_columns(
            drugId=pl.lit(None, dtype=pl.String),
            diseaseId=pl.lit(None, dtype=pl.String),
            population=pl.lit(None, dtype=pl.String),
            conclusion_axis=pl.col("conclusion").replace_strict(
                AXIS_MAP, default=None, return_dtype=pl.String
            ),
            conclusion_direction=pl.col("conclusion").replace_strict(
                DIRECTION_MAP, default=None, return_dtype=pl.String
            ),
        )
        .with_columns(_is_contested())
        .select(list(TARGET_SCHEMA))
        .cast(TARGET_SCHEMA)
    )


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return reshape(parse_raw(raw))