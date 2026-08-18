import polars as pl
import pytest

from clinical_mining.data_sources.ema import extract_marketing_year


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("19/06/2015", 2015),
        ("08/03/2002", 2002),
        ("", None),
        (None, None),
        ("not a date", None),
    ],
)
def test_extract_marketing_year(value, expected):
    assert extract_marketing_year(value) == expected


def test_extract_marketing_year_in_select():
    df = pl.DataFrame(
        {"Marketing authorisation date": ["19/06/2015", "", "01/01/2024"]}
    )
    out = df.with_columns(
        year=pl.col("Marketing authorisation date").map_elements(
            extract_marketing_year, return_dtype=pl.Int32
        )
    )
    assert out["year"].to_list() == [2015, None, 2024]
