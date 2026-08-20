import polars as pl

from clinical_mining.provider.chembl.drug_warnings import extract_clinical_report


def _make_drug_warning(**overrides) -> pl.DataFrame:
    data = {
        "warning_id": [1],
        "molregno": [10],
        "warning_type": ["Withdrawn"],
        "warning_year": [2000],
        "warning_country": ["US;UK"],
        "warning_class": ["Hepatotoxicity"],
        "efo_id": ["EFO:123"],
        "efo_term": ["hepatotoxicity"],
        "efo_id_for_warning_class": ["EFO:999"],
    }
    data.update(overrides)
    return pl.DataFrame(
        data,
        schema_overrides={
            "warning_class": pl.String,
            "efo_id": pl.String,
            "efo_term": pl.String,
            "efo_id_for_warning_class": pl.String,
        },
    )


def _make_molecule_dictionary(n: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "molregno": [10 + i for i in range(n)],
            "chembl_id": [f"CHEMBL{i + 1}" for i in range(n)],
            "pref_name": [f"drug{i + 1}" for i in range(n)],
        }
    )


def _make_warning_refs(n: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "warning_id": [1 + i for i in range(n)],
            "ref_type": ["WHO"] * n,
            "ref_id": ["ref1"] * n,
            "ref_url": ["http://example.org/ref1"] * n,
        }
    )


def assert_no_null_entity_structs(df: pl.DataFrame) -> None:
    """Assert list-of-struct columns hold no all-null structs and no empty lists.

    Absence must be represented as null, so that `is_null()` is a reliable test.
    """
    for col, dtype in df.schema.items():
        if not (isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct)):
            continue
        fields = [f.name for f in dtype.inner.fields]
        stats = df.select(
            n_all_null=pl.col(col)
            .list.eval(
                pl.element().filter(
                    pl.all_horizontal(
                        [pl.element().struct.field(f).is_null() for f in fields]
                    )
                )
            )
            .list.len()
            .sum(),
            n_empty=(pl.col(col).list.len() == 0).sum(),
        ).row(0)
        assert stats[0] == 0, f"{col} contains {stats[0]} all-null struct(s)"
        assert stats[1] == 0, f"{col} contains {stats[1]} empty list(s)"


def test_side_effects_null_when_no_annotation():
    """A warning with no EFO annotation keeps its row, with a null sideEffects."""
    warning = _make_drug_warning(
        warning_class=[None],
        efo_id=[None],
        efo_term=[None],
        efo_id_for_warning_class=[None],
    )
    report = extract_clinical_report(
        warning, _make_molecule_dictionary(), _make_warning_refs()
    ).df

    assert report.height == 1
    assert report["sideEffects"].is_null().all()
    assert report["drugs"].to_list() == [
        [{"drugFromSource": "drug1", "drugId": "CHEMBL1"}]
    ]


def test_side_effects_kept_when_annotated():
    """efo_id/efo_term win over the warning class fallback, and `:` becomes `_`."""
    report = extract_clinical_report(
        _make_drug_warning(), _make_molecule_dictionary(), _make_warning_refs()
    ).df

    assert report["sideEffects"].to_list() == [
        [{"diseaseId": "EFO_123", "diseaseFromSource": "hepatotoxicity"}]
    ]
    assert_no_null_entity_structs(report)


def test_side_effects_falls_back_to_warning_class():
    """Without an EFO annotation, the warning class is used."""
    report = extract_clinical_report(
        _make_drug_warning(efo_id=[None], efo_term=[None]),
        _make_molecule_dictionary(),
        _make_warning_refs(),
    ).df

    assert report["sideEffects"].to_list() == [
        [{"diseaseId": "EFO_999", "diseaseFromSource": "Hepatotoxicity"}]
    ]


def test_unannotated_warning_dropped_from_mixed_group():
    """One reference reporting several warnings: unannotated ones don't pad the list."""
    warning = _make_drug_warning(
        warning_id=[1, 2],
        molregno=[10, 10],
        warning_type=["Withdrawn", "Withdrawn"],
        warning_year=[2000, 2000],
        warning_country=["US;UK", "US;UK"],
        warning_class=["Hepatotoxicity", None],
        efo_id=["EFO:123", None],
        efo_term=["hepatotoxicity", None],
        efo_id_for_warning_class=["EFO:999", None],
    )
    report = extract_clinical_report(
        warning, _make_molecule_dictionary(), _make_warning_refs(n=2)
    ).df

    # Both warnings share ref_id + chembl_id, so they collapse into one report
    assert report.height == 1
    assert report["sideEffects"].to_list() == [
        [{"diseaseId": "EFO_123", "diseaseFromSource": "hepatotoxicity"}]
    ]
    assert_no_null_entity_structs(report)
