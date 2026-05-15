"""Tests for prior_work API clients – uses ``responses`` to mock HTTP."""

from __future__ import annotations

import json

import pytest
import responses as rsps_lib

from research_gap.prior_work.openalex import search_openalex, _invert_abstract
from research_gap.prior_work.semantic_scholar import search_semantic_scholar


# ---------------------------------------------------------------------------
# OpenAlex tests
# ---------------------------------------------------------------------------

class TestInvertAbstract:
    def test_simple_sentence(self):
        inverted = {"Hello": [0], "world": [1]}
        result = _invert_abstract(inverted)
        assert result == "Hello world"

    def test_none_returns_empty(self):
        assert _invert_abstract(None) == ""

    def test_multiple_positions(self):
        inverted = {"the": [0, 3], "cat": [1], "sat": [2]}
        result = _invert_abstract(inverted)
        tokens = result.split()
        assert tokens[0] == "the"
        assert tokens[1] == "cat"
        assert tokens[2] == "sat"
        assert tokens[3] == "the"


class TestSearchOpenAlex:
    _OPENALEX_URL = "https://api.openalex.org/works"

    def _make_work(
        self,
        title: str = "Test Paper",
        year: int = 2023,
        cited_by: int = 42,
        abstract_inv: dict | None = None,
        doi: str = "https://doi.org/10.1234/test",
    ) -> dict:
        return {
            "id": "https://openalex.org/W1234",
            "title": title,
            "authorships": [
                {"author": {"display_name": "Alice Smith"}},
                {"author": {"display_name": "Bob Jones"}},
            ],
            "publication_year": year,
            "host_venue": {"display_name": "NeurIPS", "publisher": "ACM"},
            "abstract_inverted_index": abstract_inv or {"test": [0], "abstract": [1]},
            "cited_by_count": cited_by,
            "doi": doi,
            "open_access": {"oa_url": ""},
        }

    @rsps_lib.activate
    def test_basic_query_construction(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            json={"results": [self._make_work()], "meta": {"count": 1}},
            status=200,
        )
        results = search_openalex("rotary positional encoding", top_k=5)
        assert len(results) == 1
        # Verify the request was made with correct params
        req = rsps_lib.calls[0].request
        assert "search=rotary+positional+encoding" in req.url or "search=rotary" in req.url

    @rsps_lib.activate
    def test_normalised_fields_present(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            json={"results": [self._make_work(title="My Paper", year=2022, cited_by=10)]},
            status=200,
        )
        results = search_openalex("test query")
        assert results[0]["title"] == "My Paper"
        assert results[0]["year"] == 2022
        assert results[0]["citation_count"] == 10
        assert results[0]["source"] == "openalex"
        assert "Alice Smith" in results[0]["authors"]

    @rsps_lib.activate
    def test_empty_results(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            json={"results": []},
            status=200,
        )
        results = search_openalex("very obscure query xyz")
        assert results == []

    @rsps_lib.activate
    def test_http_error_returns_empty(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            status=500,
        )
        results = search_openalex("test")
        assert results == []

    @rsps_lib.activate
    def test_top_k_respected(self):
        works = [self._make_work(title=f"Paper {i}") for i in range(10)]
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            json={"results": works},
            status=200,
        )
        results = search_openalex("test", top_k=3)
        assert len(results) == 3

    @rsps_lib.activate
    def test_email_added_to_user_agent(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            json={"results": []},
            status=200,
        )
        search_openalex("test", email="researcher@example.com")
        req = rsps_lib.calls[0].request
        assert "researcher@example.com" in req.headers.get("User-Agent", "")

    @rsps_lib.activate
    def test_abstract_reconstructed_from_inverted_index(self):
        inv = {"neural": [0], "network": [1], "training": [2]}
        rsps_lib.add(
            rsps_lib.GET,
            self._OPENALEX_URL,
            json={"results": [self._make_work(abstract_inv=inv)]},
            status=200,
        )
        results = search_openalex("test")
        assert results[0]["abstract"] == "neural network training"


# ---------------------------------------------------------------------------
# Semantic Scholar tests
# ---------------------------------------------------------------------------

class TestSearchSemanticScholar:
    _S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def _make_paper(
        self,
        title: str = "S2 Test Paper",
        year: int = 2023,
        abstract: str = "A test abstract.",
        citation_count: int = 7,
        url: str = "https://www.semanticscholar.org/paper/abc",
    ) -> dict:
        return {
            "paperId": "abc123",
            "title": title,
            "authors": [{"name": "Carol White"}, {"name": "Dave Brown"}],
            "year": year,
            "venue": "ICML",
            "abstract": abstract,
            "citationCount": citation_count,
            "externalIds": {"DOI": "10.5678/s2test"},
            "url": url,
        }

    @rsps_lib.activate
    def test_basic_query_construction(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": [self._make_paper()], "total": 1},
            status=200,
        )
        results = search_semantic_scholar("attention mechanism")
        assert len(results) == 1
        req = rsps_lib.calls[0].request
        assert "attention" in req.url

    @rsps_lib.activate
    def test_normalised_fields_present(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": [self._make_paper(title="S2 Paper", year=2021, citation_count=99)]},
            status=200,
        )
        results = search_semantic_scholar("test")
        assert results[0]["title"] == "S2 Paper"
        assert results[0]["year"] == 2021
        assert results[0]["citation_count"] == 99
        assert results[0]["source"] == "semantic_scholar"
        assert "Carol White" in results[0]["authors"]

    @rsps_lib.activate
    def test_api_key_added_to_header(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": []},
            status=200,
        )
        search_semantic_scholar("test", api_key="my-secret-key")
        req = rsps_lib.calls[0].request
        assert req.headers.get("x-api-key") == "my-secret-key"

    @rsps_lib.activate
    def test_no_api_key_no_header(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": []},
            status=200,
        )
        search_semantic_scholar("test")
        req = rsps_lib.calls[0].request
        assert "x-api-key" not in req.headers

    @rsps_lib.activate
    def test_empty_results(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": []},
            status=200,
        )
        results = search_semantic_scholar("very obscure query xyz")
        assert results == []

    @rsps_lib.activate
    def test_http_error_returns_empty(self):
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            status=429,
        )
        results = search_semantic_scholar("test")
        assert results == []

    @rsps_lib.activate
    def test_top_k_respected(self):
        papers = [self._make_paper(title=f"S2 Paper {i}") for i in range(10)]
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": papers},
            status=200,
        )
        results = search_semantic_scholar("test", top_k=4)
        assert len(results) == 4

    @rsps_lib.activate
    def test_url_falls_back_to_doi(self):
        paper = self._make_paper(url="")
        paper["externalIds"] = {"DOI": "10.1234/fallback"}
        rsps_lib.add(
            rsps_lib.GET,
            self._S2_URL,
            json={"data": [paper]},
            status=200,
        )
        results = search_semantic_scholar("test")
        assert results[0]["url"] == "https://doi.org/10.1234/fallback"
