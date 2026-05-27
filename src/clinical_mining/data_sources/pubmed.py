"""PubMed publication fetching utilities using Entrez API."""

import time
import sys
from http.client import IncompleteRead

from Bio import Entrez
from loguru import logger


def _iter_batches(items: list[int | str], batch_size: int) -> list[list[int | str]]:
    """Split a list into contiguous batches."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def fetch_publications(pmids: list[int | str], batch_size: int = 1_000, max_retries: int = 2) -> dict[str, dict]:
    """Fetch title and abstractText the Entrez API for a list of PubMed IDs."""
    if not pmids:
        return {}

    Entrez.email = "my_email@example.com"
    publications = {}
    total_batches = (len(pmids) + batch_size - 1) // batch_size
    for idx, batch in enumerate(_iter_batches(pmids, batch_size=batch_size), start=1):
        for attempt in range(1, max_retries + 1):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(str(pmid) for pmid in batch),
                    rettype="medline",
                    retmode="xml",
                )
                records = Entrez.read(handle)
                handle.close()

                for article in records.get("PubmedArticle", []):
                    citation = article["MedlineCitation"]
                    pmid = str(citation["PMID"])
                    article_data = citation.get("Article", {})
                    title = str(article_data.get("ArticleTitle", ""))
                    abstract_chunks = article_data.get("Abstract", {}).get("AbstractText", [])
                    abstract = " ".join(str(chunk) for chunk in abstract_chunks)
                    publications[pmid] = {
                        "title": title,
                        "abstractText": abstract,
                    }

                logger.info(
                    "Fetched PubMed batch {}/{} (size={}, cumulative={})",
                    idx,
                    total_batches,
                    len(batch),
                    len(publications),
                )
                break
            except IncompleteRead as exc:
                logger.warning(
                    "Entrez partial response for batch {}/{} (size={}) attempt {}/{}: {}",
                    idx,
                    total_batches,
                    len(batch),
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(attempt)
                else:
                    print(
                        f"Warning: Entrez fetch failed for batch {idx}/{total_batches} "
                        f"(size={len(batch)}): {exc}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(
                    f"Warning: Entrez fetch failed for batch {idx}/{total_batches} "
                    f"(size={len(batch)}): {exc}",
                    file=sys.stderr,
                )
                break

    return publications


def build_publications_map(
    records: list[dict],
    max_pubs: int = 1,
) -> dict[str, list[dict]]:
    """Pre-fetch selected publications for a list of records.

    Selects up to max_pubs PMIDs per trial, batch-fetches only those
    unique PMIDs, then returns {nct_id: [pub_dict, ...]}.
    """
    if max_pubs <= 0:
        return {}

    selected_pmids_by_trial: dict[str, list[str]] = {}
    selected_pmids: set[str] = set()
    for record in records:
        nct_id = record["id"]
        trial_pmids = [str(pmid) for pmid in (record.get("trialLiterature") or [])[:max_pubs]]
        if trial_pmids:
            selected_pmids_by_trial[nct_id] = trial_pmids
            selected_pmids.update(trial_pmids)

    if not selected_pmids:
        return {}

    logger.info(
        "Preparing publication fetch for {} selected unique PMIDs across {} records (max_pubs={})",
        len(selected_pmids),
        len(records),
        max_pubs,
    )

    pub_lookup = fetch_publications(list(selected_pmids))

    result: dict[str, list[dict]] = {}
    for nct_id, pmids in selected_pmids_by_trial.items():
        pubs = [pub_lookup[pmid] for pmid in pmids if pmid in pub_lookup]
        if pubs:
            result[nct_id] = pubs
    return result
