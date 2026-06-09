"""Tests for PubMed publication fetching utilities."""

from unittest.mock import MagicMock, patch

from clinical_mining.data_sources.pubmed import (
    build_publications_map,
    fetch_publications,
)

# ---------------------------------------------------------------------------
# Sample Entrez XML parse result (what Entrez.read returns)
# ---------------------------------------------------------------------------

SINGLE_ARTICLE = {
    "PubmedArticle": [
        {
            "MedlineCitation": {
                "PMID": "12345678",
                "Article": {
                    "ArticleTitle": "Paper A",
                    "Abstract": {"AbstractText": ["Abstract", "A."]},
                },
            }
        },
    ],
}

TWO_ARTICLES = {
    "PubmedArticle": [
        {
            "MedlineCitation": {
                "PMID": "12345678",
                "Article": {
                    "ArticleTitle": "Paper A",
                    "Abstract": {"AbstractText": ["Abstract", "A."]},
                },
            }
        },
        {
            "MedlineCitation": {
                "PMID": "99999999",
                "Article": {
                    "ArticleTitle": "Paper B",
                    "Abstract": {"AbstractText": ["Abstract", "B."]},
                },
            }
        },
    ],
}


def _mock_entrez_read(data):
    """Return a fake handle whose .read() returns *data*."""
    handle = MagicMock()
    handle.__enter__.return_value = handle
    handle.__exit__.return_value = None
    return handle


# ---------------------------------------------------------------------------
# fetch_publications
# ---------------------------------------------------------------------------


def test_fetch_publications_empty_list():
    assert fetch_publications([]) == {}


def test_fetch_publications_success():
    handle = _mock_entrez_read(TWO_ARTICLES)
    with (
        patch("clinical_mining.data_sources.pubmed.Entrez.efetch", return_value=handle),
        patch("clinical_mining.data_sources.pubmed.Entrez.read", return_value=TWO_ARTICLES),
    ):
        result = fetch_publications([12345678, 99999999])

    assert result["12345678"]["title"] == "Paper A"
    assert result["12345678"]["abstractText"] == "Abstract A."
    assert result["99999999"]["title"] == "Paper B"
    assert result["99999999"]["abstractText"] == "Abstract B."


def test_fetch_publications_missing_abstract_becomes_empty_string():
    no_abstract = {
        "PubmedArticle": [
            {
                "MedlineCitation": {
                    "PMID": "11111111",
                    "Article": {"ArticleTitle": "Title only"},
                }
            }
        ],
    }
    handle = _mock_entrez_read(no_abstract)
    with (
        patch("clinical_mining.data_sources.pubmed.Entrez.efetch", return_value=handle),
        patch("clinical_mining.data_sources.pubmed.Entrez.read", return_value=no_abstract),
    ):
        result = fetch_publications([11111111])

    assert result["11111111"]["title"] == "Title only"
    assert result["11111111"]["abstractText"] == ""


def test_fetch_publications_retries_on_incomplete_read(capsys):
    from http.client import IncompleteRead

    handle = _mock_entrez_read(SINGLE_ARTICLE)
    efetch = MagicMock(side_effect=[IncompleteRead(b""), handle])
    with (
        patch("clinical_mining.data_sources.pubmed.Entrez.efetch", efetch),
        patch("clinical_mining.data_sources.pubmed.Entrez.read", return_value=SINGLE_ARTICLE),
    ):
        result = fetch_publications([12345678], max_retries=2)

    assert result["12345678"]["title"] == "Paper A"
    assert efetch.call_count == 2


# ---------------------------------------------------------------------------
# build_publications_map
# ---------------------------------------------------------------------------


def test_build_publications_map_basic():
    records = [
        {"id": "NCT00000001", "trialLiterature": [12345678, 99999999]},
        {"id": "NCT00000002", "trialLiterature": [12345678]},
        {"id": "NCT00000003", "trialLiterature": None},
    ]
    pub_data = {
        "12345678": {"title": "Paper A", "abstractText": "Abstract A"},
        "99999999": {"title": "Paper B", "abstractText": "Abstract B"},
    }
    with patch("clinical_mining.data_sources.pubmed.fetch_publications", return_value=pub_data):
        result = build_publications_map(records, max_pubs=2)

    assert len(result["NCT00000001"]) == 2
    assert result["NCT00000001"][0]["title"] == "Paper A"
    assert len(result["NCT00000002"]) == 1
    assert result.get("NCT00000003") is None


def test_build_publications_map_respects_max_pubs():
    records = [{"id": "NCT00000001", "trialLiterature": [1, 2, 3, 4, 5]}]
    pub_data = {str(i): {"title": f"Paper {i}", "abstractText": f"Abs {i}"} for i in range(1, 6)}
    with patch("clinical_mining.data_sources.pubmed.fetch_publications", return_value=pub_data):
        result = build_publications_map(records, max_pubs=3)

    assert len(result["NCT00000001"]) == 3


def test_build_publications_map_deduplicates_pmids():
    records = [
        {"id": "NCT00000001", "trialLiterature": [111, 222]},
        {"id": "NCT00000002", "trialLiterature": [222, 333]},
    ]
    with patch("clinical_mining.data_sources.pubmed.fetch_publications", return_value={}) as mock_fetch:
        build_publications_map(records, max_pubs=3)

    assert mock_fetch.call_count == 1
    assert set(mock_fetch.call_args[0][0]) == {"111", "222", "333"}
