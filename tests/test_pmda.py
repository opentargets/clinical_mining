import polars as pl
import pytest

from clinical_mining.provider.pmda import extract_approval_year


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Sep. 24, 2024", 2024),
        ("May 7, 2020", 2020),
        ("20-Apr-06", 2006),
        ("15-Jun-06", 2006),
        ("(1) Aug. 21,\n2018\n(2) Aug. 22,\n2018", 2018),
        ("", None),
        (None, None),
        ("2266--JJuull--0066", None),
    ],
)
def test_extract_approval_year(value, expected):
    assert extract_approval_year(value) == expected


def test_extract_approval_year_in_select():
    df = pl.DataFrame({"approval_date": ["Sep. 24, 2024", "", "20-Apr-06", None]})
    out = df.with_columns(
        year=pl.col("approval_date").map_elements(
            extract_approval_year, return_dtype=pl.Int32
        )
    )
    assert out["year"].to_list() == [2024, None, 2006, None]


def test_find_column_structure_detects_reversed_date_header():
    """Pages 185-196 use 'Date of approval' instead of 'Approval Date'."""
    from clinical_mining.provider.pmda import find_column_structure

    header = [
        "Category",
        "Date of\napproval",
        "Brand name (name of company)",
        "Approval/ Supplemental Change",
        "Names of ingredients (Underlined: New active ingredients)",
        "Note",
    ]
    structure = find_column_structure(header)
    assert structure.approval_date_idx == 1
    assert structure.brand_name_idx == 2
    assert structure.ingredient_idx == 4


def test_extract_approval_year_two_digit_dd_mon_yy():
    """Two-digit year format like '9-Jul-04' maps to the 2000s."""
    assert extract_approval_year("9-Jul-04") == 2004
