import pdfplumber
import polars as pl
import re
from dataclasses import dataclass
from loguru import logger

from ontoma.ner.disease import extract_disease_entities
from clinical_mining.utils.polars_helpers import convert_polars_to_spark

from clinical_mining.utils.spark_helpers import spark_session
from clinical_mining.dataset import ClinicalReport
from clinical_mining.schemas import ClinicalReportType
import polars_hash as plh


# ============================================================================
# CONFIGURATION
# ============================================================================

# Header keywords that must appear together to identify a header row
HEADER_KEYWORDS = {
    "approval_partial": ("approval", "partial"),
    "active_ingredient": ("active ingredient",),
    "notes": ("note",),
}

# Minimum keyword groups that must match to confirm a header row
MIN_HEADER_MATCHES = 2


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ColumnStructure:
    """Represents the column indices for a table."""

    approval_date_idx: int | None = None
    brand_name_idx: int | None = None
    approval_idx: int | None = None
    ingredient_idx: int | None = None
    notes_idx: int | None = None

    def is_valid(self) -> bool:
        """Check if structure has at least one useful column."""
        # return self.approval_idx is not None or self.brand_name_idx is not None or self.approval_data_idx is not None or self.ingredient_idx is not None or self.notes_idx is not None
        return any(
            [
                self.approval_date_idx,
                self.brand_name_idx,
                self.approval_idx,
                self.ingredient_idx,
                self.notes_idx,
            ]
        )


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# ACTIVE INGREDIENT CLEANING
# ============================================================================


def clean_active_ingredients(raw_text: str) -> list[str]:
    """Parse and clean active ingredient text.

    Handles:
    - Multiple ingredients (newline-separated)
    - Numbered items: (1), (2), etc.
    - "(genetical recombination)" suffix
    """
    if not raw_text:
        return []

    # Remove "(genetical recombination)"
    text = re.sub(r"\(genetical\s+recombination\)", "", raw_text, flags=re.IGNORECASE)
    # Collapse any newlines/extra whitespace to single spaces to keep terms readable
    text = re.sub(r"\s+", " ", text)

    # Treat "N/A" style markers as missing ingredients entirely
    if re.match(
        r"(?i)^\s*(n\s*/?\s*a|n\.?\s*a\.?|not\s+applicable|not\s+available)\b", text
    ):
        return []

    # Protect dosage/unit patterns like "10 mg/mL" so we don't split inside them
    UNIT_SENTINEL = "__PMDA_UNIT_SLASH__"
    protected_text = re.sub(
        r"(\d[\d,\.\s]*[A-Za-zµ%]+)\s*/\s*([A-Za-zµ%]+)",
        lambda m: f"{m.group(1)}{UNIT_SENTINEL}{m.group(2)}",
        text,
    )

    # Split and clean each ingredient. Treat "/" as a separator only when surrounded
    # by whitespace so unit expressions like "mg/mL" stay intact.
    ingredients = []
    for line in re.split(r"\s+/\s+", protected_text):
        line = line.strip()
        if not line:
            continue

        # Remove leading numbering: (1), (2), etc.
        line = re.sub(r"^\(\d+\)\s*", "", line)
        line = line.replace(UNIT_SENTINEL, "/").strip()

        if line:
            ingredients.append(line)

    return ingredients


# ============================================================================
# HEADER DETECTION
# ============================================================================


def count_matching_keywords(row: list) -> int:
    """Count how many keyword groups from HEADER_KEYWORDS match this row."""
    row_text = " ".join(str(cell or "") for cell in row).lower()
    return sum(
        1
        for keywords in HEADER_KEYWORDS.values()
        if all(kw in row_text for kw in keywords)
    )


def is_header_row(row: list) -> bool:
    """
    Check if row is a header by counting keyword matches.
    Requires MIN_HEADER_MATCHES to avoid false positives from data rows.
    """
    return count_matching_keywords(row) >= MIN_HEADER_MATCHES


def find_column_structure(header_row: list) -> ColumnStructure:
    """Extract column indices from a header row."""
    structure = ColumnStructure()

    for idx, cell in enumerate(header_row):
        if not cell:
            continue

        cell_lower = str(cell).lower()

        if "approval date" in cell_lower:
            structure.approval_date_idx = idx
        elif "brand name" in cell_lower:
            structure.brand_name_idx = idx
        elif "approval" in cell_lower and "partial" in cell_lower:
            structure.approval_idx = idx
        elif "active ingredient" in cell_lower:
            structure.ingredient_idx = idx
        elif "note" in cell_lower:
            structure.notes_idx = idx

    return structure


def find_header_in_table(table: list) -> tuple[int, ColumnStructure] | None:
    """Search for a valid header row in a table.

    Args:
        table: List of rows in the table.

    Returns:
        Tuple of (row_index, ColumnStructure) if a header is found, None otherwise.
    """
    for idx, row in enumerate(table):
        if not row or not any(row):
            continue

        if not is_header_row(row):
            continue

        structure = find_column_structure(row)
        if structure.is_valid():
            return idx, structure

    return None


# ============================================================================
# DATA ROW VALIDATION
# ============================================================================


def is_data_row(row: list) -> bool:
    """Check if row contains meaningful data (at least 2 non-empty cells)."""
    if not row or len(row) < 2:
        return False

    non_empty = sum(1 for cell in row if cell and str(cell).strip())
    return non_empty >= 2


def extract_cell_value(row: list, col_idx: int | None) -> str:
    """Safely extract and normalize a cell value."""
    if col_idx is None or col_idx >= len(row):
        return ""
    return str(row[col_idx] or "").strip()


def is_approval_row(approval_value: str, has_approval_column: bool) -> bool:
    """Check if row represents an approval. If no approval column exists, assume all rows are approvals."""
    if not has_approval_column:
        return True

    if not approval_value:
        return False

    if "approval" in approval_value.lower():
        return True


# ============================================================================
# MAIN PARSER
# ============================================================================


def parse_table_with_structure(
    table: list, structure: ColumnStructure, start_row: int, page_num: int
) -> list[dict]:
    """Parse table rows using a given column structure."""

    records = []

    for row in table[start_row:]:
        if not is_data_row(row):
            continue

        # Extract values
        approval_date = extract_cell_value(row=row, col_idx=structure.approval_date_idx)
        brand_name = extract_cell_value(row=row, col_idx=structure.brand_name_idx)
        approval_type = extract_cell_value(row=row, col_idx=structure.approval_idx)
        ingredient_raw = extract_cell_value(row=row, col_idx=structure.ingredient_idx)
        notes = extract_cell_value(row=row, col_idx=structure.notes_idx)

        # Filter: only include approval rows
        if not is_approval_row(approval_type, structure.approval_idx is not None):
            continue

        # Skip if no meaningful content
        if not ingredient_raw and not notes:
            continue

        records.append(
            {
                "approval_date": approval_date,
                "brand_name": brand_name,
                "active_ingredient": ingredient_raw,
                "notes": notes,
                "approval_type": approval_type,
                "page_number": page_num,
            }
        )

    return records


def parse_pmda_approvals(pmda_path: str) -> pl.DataFrame:
    """Parse PMDA drug approvals PDF into a Polars DataFrame.

    Only includes entries where approval type contains 'Approval'.
    """

    all_records = []
    last_structure: ColumnStructure | None = None

    with pdfplumber.open(pmda_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            logger.info(f"Processing page {page_num}/{len(pdf.pages)}")

            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue

                header_info = find_header_in_table(table)

                if header_info:
                    header_idx, structure = header_info
                    # Found header: use it and update cache
                    last_structure = structure
                    start_row = header_idx + 1

                elif last_structure and last_structure.is_valid():
                    # No header but have cached structure: reuse it
                    logger.info("  → Using cached header structure (no header found)")
                    structure = last_structure
                    start_row = 0

                else:
                    # No header and no cache: skip table
                    logger.info("  → Skipping table (no header or cached structure)")
                    continue

                # Parse the table
                try:
                    records = parse_table_with_structure(
                        table, structure, start_row, page_num
                    )
                    all_records.extend(records)
                except Exception as e:
                    logger.warning(f"Error processing table: {e}")
                    continue

    return (
        pl.DataFrame(all_records)
        .filter((pl.col("active_ingredient") != "") | (pl.col("notes") != ""))
        .unique()
        .sort("page_number")
    )


def extract_clinical_report(
    df: pl.DataFrame,
    spark: spark_session,
) -> ClinicalReport:
    """Extract clinical reports from PMDA approvals."""
    reports = (
        # Extract disease entities (requires Polars - Spark - Polars conversion)
        pl.from_pandas(
            extract_disease_entities(
                spark,
                df=convert_polars_to_spark(
                    polars_df=(
                        df.with_columns(
                            disease_text=pl.col("notes").fill_null("").str.strip_chars()
                        ).drop("notes")
                    ),
                    spark=spark,
                ),
                input_col="disease_text",
                output_col="extracted_diseases",
            ).toPandas()
        )
        # Explode extracted diseases
        .explode(
            "extracted_diseases",
            keep_nulls=False,
        )
        # Add active ingredients column
        .with_columns(
            active_ingredients=pl.col("active_ingredient")
            .fill_null("")
            .map_elements(clean_active_ingredients, return_dtype=pl.List(pl.Utf8)),
        )
        .explode("active_ingredients")
        .filter(
            (pl.col("active_ingredients") != "") & (pl.col("extracted_diseases") != "")
        )
        .select(
            phaseFromSource=pl.lit("approval"),
            type=pl.lit(ClinicalReportType.REGULATORY),
            drugFromSource=pl.col("active_ingredients")
            .str.strip_chars()
            .str.to_lowercase(),
            diseaseFromSource=pl.col("extracted_diseases"),
            hasExpertReview=pl.lit(False),
            url=pl.lit(
                "https://www.pmda.go.jp/english/review-services/reviews/approved-information/drugs/0001.html"
            ),
            source=pl.lit("PMDA"),
        )
        .with_columns(
            id=plh.concat_str("drugFromSource", "diseaseFromSource").chash.sha2_256(),
        )
        .unique()
    )

    mapped_reports = (
        # TODO: call mapping function
        reports.with_columns(
            disease=pl.struct(
                pl.col("diseaseFromSource"), pl.lit("CHEMBL_TO_DO").alias("diseaseId")
            ),
            drug=pl.struct(
                pl.col("drugFromSource"), pl.lit("EFO_TO_DO").alias("drugId")
            ),
        )
        .drop(["diseaseFromSource", "drugFromSource"])
        .unique()
    )
    return ClinicalReport(
        df=(
            mapped_reports.group_by(
                [c for c in mapped_reports.columns if c not in ["disease", "drug"]]
            ).agg(
                pl.col("disease").unique().alias("diseases"),
                pl.col("drug").unique().alias("drugs"),
            )
        )
    )
