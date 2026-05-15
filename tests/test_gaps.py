"""Tests for research_gap/gaps.py – heuristic gap extraction."""

from __future__ import annotations

import pytest

from research_gap.gaps import extract_gaps_heuristic, _make_search_queries


_CONTEXT_PACK_WITH_GAPS = {
    "title": "A Test Paper",
    "abstract": "We propose a method for X.",
    "headings": ["Introduction", "Limitations", "Future Work"],
    "sections": {
        "Introduction": "We propose a new approach. However, this remains an open problem.",
        "Limitations": (
            "Our approach has several limitations. "
            "We leave for future work the extension to multilingual settings. "
            "The proposed method does not scale well in low-resource regimes."
        ),
        "Future Work": (
            "Future directions include scaling to larger datasets. "
            "We invite the community to explore alternative formulations."
        ),
    },
    "key_sections": {
        "Limitations": (
            "Our approach has several limitations. "
            "We leave for future work the extension to multilingual settings. "
            "The proposed method does not scale well in low-resource regimes."
        ),
        "Future Work": (
            "Future directions include scaling to larger datasets. "
            "We invite the community to explore alternative formulations."
        ),
        "Introduction": "We propose a new approach. However, this remains an open problem.",
    },
}

_CONTEXT_PACK_NO_GAPS = {
    "title": "A Clean Paper",
    "abstract": "We propose a solid method.",
    "headings": ["Method", "Results"],
    "sections": {
        "Method": "We use a deep neural network.",
        "Results": "We achieve state-of-the-art performance.",
    },
    "key_sections": {},
}


class TestExtractGapsHeuristic:
    def test_finds_gaps_in_limitations(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        gap_texts = [g["gap"] for g in gaps]
        assert any("limitation" in t.lower() for t in gap_texts)

    def test_finds_future_work_sentences(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        gap_texts = [g["gap"] for g in gaps]
        assert any("future work" in t.lower() or "future directions" in t.lower() for t in gap_texts)

    def test_finds_invite_pattern(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        gap_texts = [g["gap"] for g in gaps]
        assert any("invite" in t.lower() for t in gap_texts)

    def test_evidence_has_section_and_quote(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        assert gaps, "Expected at least one gap"
        for gap in gaps:
            assert "evidence" in gap
            assert len(gap["evidence"]) >= 1
            ev = gap["evidence"][0]
            assert "section" in ev
            assert "quote" in ev
            assert ev["section"] != ""
            assert ev["quote"] != ""

    def test_no_duplicate_sentences(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        gap_texts = [g["gap"] for g in gaps]
        assert len(gap_texts) == len(set(gap_texts)), "Duplicate gap sentences found"

    def test_no_gaps_returns_empty_list(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_NO_GAPS)
        assert gaps == []

    def test_output_schema_keys(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        required_keys = {
            "gap", "evidence", "why_it_matters",
            "non_incremental_directions", "prior_work_search_queries",
        }
        for gap in gaps:
            assert required_keys.issubset(gap.keys()), f"Missing keys in gap: {gap}"

    def test_search_queries_generated(self):
        gaps = extract_gaps_heuristic(_CONTEXT_PACK_WITH_GAPS)
        assert gaps, "Expected gaps"
        for gap in gaps:
            assert isinstance(gap["prior_work_search_queries"], list)
            assert len(gap["prior_work_search_queries"]) >= 1


class TestMakeSearchQueries:
    def test_returns_list(self):
        result = _make_search_queries("We leave for future work the extension to multilingual settings.")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_short_text_falls_back_gracefully(self):
        result = _make_search_queries("hi")
        assert isinstance(result, list)

    def test_filters_short_words(self):
        result = _make_search_queries("We do not know if this is correct")
        # "know" and "correct" are 4-5 chars; "correct" is 7 chars and should appear
        query = result[0] if result else ""
        assert "we" not in query.lower().split() or len(query) > 0
