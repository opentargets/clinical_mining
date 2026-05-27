from clinical_mining.data_sources.europepmc import build_epmc_url


def test_build_epmc_url_single_pmid():
    url = build_epmc_url([12345678])
    assert "12345678" in url
    assert "format=json" in url
    assert "pageSize=1" in url


def test_build_epmc_url_multiple_pmids():
    url = build_epmc_url([111, 222, 333])
    assert "pageSize=3" in url
    assert "111" in url
    assert "222" in url
    assert "333" in url


def test_build_epmc_url_string_pmids():
    url = build_epmc_url(["12345678", "99999999"])
    assert "pageSize=2" in url
    assert "12345678" in url


import httpx
from unittest.mock import MagicMock, patch

EPMC_RESPONSE = {
    "resultList": {
        "result": [
            {"pmid": "12345678", "title": "Paper A", "abstractText": "Abstract A."},
            {"pmid": "99999999", "title": "Paper B", "abstractText": "Abstract B."},
        ]
    }
}


def _mock_epmc_response(data):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = data
    return m


def test_fetch_publications_success():
    from clinical_mining.data_sources.europepmc import fetch_publications
    with patch("clinical_mining.data_sources.europepmc.httpx.get", return_value=_mock_epmc_response(EPMC_RESPONSE)):
        result = fetch_publications([12345678, 99999999])
    assert result["12345678"]["title"] == "Paper A"
    assert result["12345678"]["abstractText"] == "Abstract A."
    assert result["99999999"]["title"] == "Paper B"


def test_fetch_publications_missing_abstractText_is_none():
    from clinical_mining.data_sources.europepmc import fetch_publications
    data = {"resultList": {"result": [{"pmid": "11111111", "title": "Title only"}]}}
    with patch("clinical_mining.data_sources.europepmc.httpx.get", return_value=_mock_epmc_response(data)):
        result = fetch_publications([11111111])
    assert result["11111111"]["title"] == "Title only"
    assert result["11111111"]["abstractText"] is None


def test_fetch_publications_http_error_returns_empty_and_warns(capsys):
    from clinical_mining.data_sources.europepmc import fetch_publications
    with patch("clinical_mining.data_sources.europepmc.httpx.get", side_effect=httpx.HTTPError("timeout")):
        result = fetch_publications([12345678])
    assert result == {}
    assert "Warning" in capsys.readouterr().err


def test_fetch_publications_empty_list_returns_empty():
    from clinical_mining.data_sources.europepmc import fetch_publications
    result = fetch_publications([])
    assert result == {}


def test_build_publications_map_basic():
    from clinical_mining.data_sources.europepmc import build_publications_map
    records = [
        {"id": "NCT00000001", "trialLiterature": [12345678, 99999999]},
        {"id": "NCT00000002", "trialLiterature": [12345678]},
        {"id": "NCT00000003", "trialLiterature": None},
    ]
    pub_data = {
        "12345678": {"title": "Paper A", "abstractText": "Abstract A"},
        "99999999": {"title": "Paper B", "abstractText": "Abstract B"},
    }
    with patch("clinical_mining.data_sources.europepmc.fetch_publications", return_value=pub_data):
        result = build_publications_map(records)
    assert len(result["NCT00000001"]) == 2
    assert result["NCT00000001"][0]["title"] == "Paper A"
    assert len(result["NCT00000002"]) == 1
    assert result.get("NCT00000003", []) == []


def test_build_publications_map_respects_max_pubs():
    from clinical_mining.data_sources.europepmc import build_publications_map
    records = [{"id": "NCT00000001", "trialLiterature": [1, 2, 3, 4, 5]}]
    pub_data = {str(i): {"title": f"Paper {i}", "abstractText": f"Abs {i}"} for i in range(1, 6)}
    with patch("clinical_mining.data_sources.europepmc.fetch_publications", return_value=pub_data):
        result = build_publications_map(records, max_pubs=3)
    assert len(result["NCT00000001"]) == 3


def test_build_publications_map_deduplicates_pmids():
    from clinical_mining.data_sources.europepmc import build_publications_map
    records = [
        {"id": "NCT00000001", "trialLiterature": [111, 222]},
        {"id": "NCT00000002", "trialLiterature": [222, 333]},
    ]
    with patch("clinical_mining.data_sources.europepmc.fetch_publications", return_value={}) as mock_fetch:
        build_publications_map(records)
    assert mock_fetch.call_count == 1
    assert set(mock_fetch.call_args[0][0]) == {111, 222, 333}
