"""
research_gap/prior_work/semantic_scholar.py – Semantic Scholar API client.

Searches the Semantic Scholar Academic Graph (S2AG) API and returns a
normalised list of candidate paper dicts.

API key is optional; unauthenticated requests are rate-limited to ~100/5min.
Set SEMANTIC_SCHOLAR_API_KEY for higher limits.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_PAPER_SEARCH_ENDPOINT = "/paper/search"
_DEFAULT_FIELDS = "title,authors,year,venue,abstract,citationCount,externalIds,url"


def _normalise_paper(paper: dict) -> dict[str, Any]:
    """Convert a Semantic Scholar paper object to the canonical format."""
    authors: list[str] = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]

    # URL: prefer S2 page, fall back to DOI
    url = paper.get("url") or ""
    doi = (paper.get("externalIds") or {}).get("DOI", "")
    if not url and doi:
        url = f"https://doi.org/{doi}"

    return {
        "title": paper.get("title") or "",
        "authors": authors,
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
        "abstract": paper.get("abstract") or "",
        "url": url,
        "citation_count": paper.get("citationCount"),
        "source": "semantic_scholar",
    }


def search_semantic_scholar(
    query: str,
    api_key: str = "",
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for papers matching *query*.

    Parameters
    ----------
    query:
        Free-text search query.
    api_key:
        Optional Semantic Scholar API key.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Normalised candidate paper dicts.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    params: dict[str, Any] = {
        "query": query,
        "limit": min(top_k, 100),
        "fields": _DEFAULT_FIELDS,
    }

    try:
        resp = requests.get(
            _S2_BASE + _PAPER_SEARCH_ENDPOINT,
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Semantic Scholar search failed for %r: %s", query, exc)
        return []

    data = resp.json()
    papers = data.get("data", [])
    logger.debug("Semantic Scholar returned %d results for %r", len(papers), query)
    return [_normalise_paper(p) for p in papers[:top_k]]
