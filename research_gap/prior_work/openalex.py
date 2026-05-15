"""
research_gap/prior_work/openalex.py – OpenAlex API client.

Searches https://api.openalex.org/works for related papers and returns
a normalised list of candidate dicts.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_OPENALEX_BASE = "https://api.openalex.org"
_WORKS_ENDPOINT = "/works"
_DEFAULT_FIELDS = (
    "id,title,authorships,publication_year,host_venue,abstract_inverted_index,"
    "cited_by_count,doi,open_access"
)


def _invert_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted:
        return ""
    max_pos = max(pos for positions in inverted.values() for pos in positions)
    tokens: list[str] = [""] * (max_pos + 1)
    for word, positions in inverted.items():
        for pos in positions:
            tokens[pos] = word
    return " ".join(t for t in tokens if t)


def _normalise_work(work: dict) -> dict[str, Any]:
    """Convert an OpenAlex work object to the pipeline's canonical format."""
    # Authors
    authors: list[str] = []
    for authorship in work.get("authorships", []):
        name = (authorship.get("author") or {}).get("display_name", "")
        if name:
            authors.append(name)

    # Venue
    venue = ""
    hv = work.get("host_venue") or {}
    venue = hv.get("display_name") or hv.get("publisher") or ""

    # DOI / URL
    doi = work.get("doi") or ""
    oa_url = (work.get("open_access") or {}).get("oa_url") or ""
    url = doi or oa_url

    return {
        "title": work.get("title") or "",
        "authors": authors,
        "year": work.get("publication_year"),
        "venue": venue,
        "abstract": _invert_abstract(work.get("abstract_inverted_index")),
        "url": url,
        "citation_count": work.get("cited_by_count"),
        "source": "openalex",
    }


def search_openalex(
    query: str,
    email: str = "",
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search OpenAlex for papers matching *query*.

    Parameters
    ----------
    query:
        Free-text search query.
    email:
        E-mail address for OpenAlex polite pool (optional but recommended).
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Normalised candidate paper dicts.
    """
    params: dict[str, Any] = {
        "search": query,
        "per-page": min(top_k, 50),
        "select": _DEFAULT_FIELDS,
        "mailto": email,
    }
    if not email:
        params.pop("mailto")

    headers: dict[str, str] = {}
    if email:
        headers["User-Agent"] = f"research-gap/0.1 (mailto:{email})"

    try:
        resp = requests.get(
            _OPENALEX_BASE + _WORKS_ENDPOINT,
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("OpenAlex search failed for %r: %s", query, exc)
        return []

    data = resp.json()
    results = data.get("results", [])
    logger.debug("OpenAlex returned %d results for %r", len(results), query)
    return [_normalise_work(w) for w in results[:top_k]]
