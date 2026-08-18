import polars as pl
import pytest

from clinical_mining.data_sources.pmda import extract_approval_year


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
    df = pl.DataFrame(
        {"approval_date": ["Sep. 24, 2024", "", "20-Apr-06", None]}
    )
    out = df.with_columns(
        year=pl.col("approval_date").map_elements(
            extract_approval_year, return_dtype=pl.Int32
        )
    )
    assert out["year"].to_list() == [2024, None, 2006, None]
