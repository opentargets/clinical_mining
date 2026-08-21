"""Tests for ClinicalReport deduplication, especially cross-provider tie-breaks."""

import polars as pl

from clinical_mining.dataset.clinical_report import ClinicalReport
from clinical_mining.schemas import ClinicalProvider, ClinicalSource  # noqa: F401


def _report_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a post-map_entities-style DataFrame with all mandatory schema columns."""
    base = {
        "id": [r["id"] for r in rows],
        "source": [r["source"] for r in rows],
        "provider": [r["provider"] for r in rows],
        "phaseFromSource": [r.get("phaseFromSource") for r in rows],
        "clinicalStage": [r.get("clinicalStage", "APPROVAL") for r in rows],
        "origin": [r.get("origin", "REGULATORY_AGENCY") for r in rows],
        "type": [r.get("type", "INDICATION") for r in rows],
        "year": [r.get("year") for r in rows],
        "url": [r.get("url", "https://example.com") for r in rows],
        "diseases": [
            r.get("diseases", [{"diseaseFromSource": "c", "diseaseId": None}])
            for r in rows
        ],
        "drugs": [
            r.get("drugs", [{"drugFromSource": "d", "drugId": None}]) for r in rows
        ],
    }
    return pl.DataFrame(base)


def test_drop_duplicates_same_stage_first_party_wins():
    """When two rows share id+stage, the provider that owns its source wins."""
    # Order deliberately reversed (curated row before owner row) to prove
    # the sort tie-break, not input order, decides the winner.
    df = _report_frame(
        [
            {
                "id": "emea/h/c/000123",
                "source": ClinicalSource.EMA.value,
                "provider": ClinicalProvider.CHEMBL.value,
                "phaseFromSource": "APPROVAL",
            },
            {
                "id": "emea/h/c/000123",
                "source": ClinicalSource.EMA_HUMAN_DRUGS.value,
                "provider": ClinicalProvider.EMA.value,
                "phaseFromSource": "authorised",
            },  # EMA provider row
        ]
    )
    m = ClinicalProvider
    # Sanity: first-party vs third-party according to owned mapping
    assert m.owns_source("EMA", ClinicalSource.EMA_HUMAN_DRUGS.value) is True
    assert m.owns_source("ChEMBL", ClinicalSource.EMA.value) is False

    winner = ClinicalReport.drop_duplicates(df)
    # Only one row, winner must be the EMA-provider row
    assert winner.height == 1
    assert winner["provider"].to_list()[0] == ClinicalProvider.EMA.value
    # Same result when input order is swapped
    df_swapped = df.reverse()
    winner2 = ClinicalReport.drop_duplicates(df_swapped)
    assert winner2["provider"].to_list()[0] == ClinicalProvider.EMA.value


def test_drop_duplicates_best_stage_always_wins():
    """Clinical stage outranks first-party ownership: APPROVAL beats UNKNOWN."""
    df = _report_frame(
        [
            {
                "id": "x1",
                "source": ClinicalSource.EMA_HUMAN_DRUGS.value,
                "provider": ClinicalProvider.EMA.value,
                "phaseFromSource": "authorised",
                "clinicalStage": "UNKNOWN",
            },
            {
                "id": "x1",
                "source": "EMA",
                "provider": ClinicalProvider.CHEMBL.value,
                "phaseFromSource": "approved",
                "clinicalStage": "APPROVAL",
            },
        ]
    )
    winner = ClinicalReport.drop_duplicates(df)
    assert winner.height == 1
    # APPROVAL (rank 1) beats UNKNOWN (rank 11) despite being third-party
    assert winner["provider"].to_list()[0] == ClinicalProvider.CHEMBL.value


def test_drop_duplicates_non_owner_tie_is_deterministic():
    """Two non-owner rows with same id+stage: winner is deterministic (priority 1 tie)."""
    df = _report_frame(
        [
            {
                "id": "dup",
                "source": "EMA",
                "provider": "ChEMBL",
                "phaseFromSource": "APPROVAL",
            },
            {
                "id": "dup",
                "source": "EMA",
                "provider": "ChEMBL",
                "phaseFromSource": "APPROVAL",
            },
        ]
    )
    winner = ClinicalReport.drop_duplicates(df)
    assert winner.height == 1
