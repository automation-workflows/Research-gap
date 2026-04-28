"""Tests for research_gap/reporting.py – report generation."""

from __future__ import annotations

from research_gap.reporting import generate_report


_CONTEXT_PACK = {
    "title": "Learning to Rotate",
    "abstract": "This paper proposes a novel rotary encoding method.",
    "headings": ["Introduction", "Method", "Experiments", "Limitations", "Conclusion"],
    "sections": {
        "Introduction": "We propose a new approach.",
        "Limitations": "We leave for future work the extension to low-resource settings.",
    },
    "key_sections": {
        "Limitations": "We leave for future work the extension to low-resource settings.",
    },
}

_GAPS = [
    {
        "gap": "Extension to low-resource settings remains unexplored.",
        "evidence": [
            {
                "section": "Limitations",
                "quote": "We leave for future work the extension to low-resource settings.",
            }
        ],
        "why_it_matters": "Low-resource settings are important for real-world deployment.",
        "non_incremental_directions": [
            {
                "direction": "Develop a zero-shot variant",
                "axis_of_difference": "assumptions: no labelled data",
            }
        ],
        "prior_work_search_queries": ["low resource rotary encoding zero-shot"],
    }
]

_NOVELTY = [
    {
        "gap": "Extension to low-resource settings remains unexplored.",
        "idea": "Develop a zero-shot variant",
        "nearest_prior_work": [
            {
                "title": "Zero-Shot Transfer Learning",
                "authors": ["Alice"],
                "year": 2022,
                "venue": "ICML",
                "abstract": "We study zero-shot settings.",
                "url": "https://example.com/paper",
                "citation_count": 55,
                "source": "openalex",
                "similarity": 0.82,
                "risk": "high",
            }
        ],
    }
]


class TestGenerateReport:
    def test_returns_string(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_title_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "Learning to Rotate" in report

    def test_headings_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "Introduction" in report
        assert "Limitations" in report

    def test_gap_text_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "low-resource settings" in report

    def test_evidence_quote_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "future work" in report.lower()

    def test_novelty_table_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "Zero-Shot Transfer Learning" in report
        assert "high" in report

    def test_empty_gaps_still_generates(self):
        report = generate_report(_CONTEXT_PACK, [], [])
        assert "Learning to Rotate" in report
        assert "No gaps identified" in report

    def test_abstract_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "rotary encoding" in report

    def test_search_queries_in_report(self):
        report = generate_report(_CONTEXT_PACK, _GAPS, _NOVELTY)
        assert "low resource rotary encoding" in report
