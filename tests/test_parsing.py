"""Tests for research_gap/parsing.py – TEI XML parsing."""

from __future__ import annotations

import textwrap

import pytest

from research_gap.parsing import parse_tei


# ---------------------------------------------------------------------------
# Minimal TEI XML fixtures
# ---------------------------------------------------------------------------

_TEI_MINIMAL = textwrap.dedent(
    """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title type="main">Learning to Rotate: Temporal and Semantic Rotary Encoding</title>
          </titleStmt>
        </fileDesc>
      </teiHeader>
      <text>
        <front>
          <div type="abstract">
            <p>This is the abstract text.</p>
          </div>
        </front>
        <body>
          <div>
            <head>Introduction</head>
            <p>We introduce a new method. This work explores open problems in the field.</p>
          </div>
          <div>
            <head>Related Work</head>
            <p>Prior work includes many approaches. However, limitations remain unexplored.</p>
          </div>
          <div>
            <head>Conclusion</head>
            <p>We conclude that future work should address these challenges.</p>
          </div>
        </body>
      </text>
    </TEI>
    """
)

_TEI_WITH_LIMITATIONS = textwrap.dedent(
    """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title type="main">Test Paper With Limitations Section</title>
          </titleStmt>
        </fileDesc>
      </teiHeader>
      <text>
        <front>
          <div type="abstract">
            <p>Abstract of the paper.</p>
          </div>
        </front>
        <body>
          <div>
            <head>1. Introduction</head>
            <p>Introduction text here.</p>
          </div>
          <div>
            <head>Limitations</head>
            <p>Our approach has several limitations. We leave for future work the extension to multilingual settings.</p>
          </div>
          <div>
            <head>Future Work</head>
            <p>Future directions include scaling to larger datasets and exploring new modalities.</p>
          </div>
        </body>
      </text>
    </TEI>
    """
)

_TEI_EMPTY_BODY = textwrap.dedent(
    """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title type="main">Minimal Paper</title>
          </titleStmt>
        </fileDesc>
      </teiHeader>
      <text>
        <body></body>
      </text>
    </TEI>
    """
)


# ---------------------------------------------------------------------------
# Tests for parse_tei
# ---------------------------------------------------------------------------

class TestParseTei:
    def test_title_extraction(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert cp["title"] == "Learning to Rotate: Temporal and Semantic Rotary Encoding"

    def test_abstract_extraction(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert "abstract text" in cp["abstract"]

    def test_headings_list(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert "Introduction" in cp["headings"]
        assert "Related Work" in cp["headings"]
        assert "Conclusion" in cp["headings"]

    def test_sections_dict(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert "Introduction" in cp["sections"]
        assert "open problems" in cp["sections"]["Introduction"]

    def test_key_sections_introduction(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert "Introduction" in cp["key_sections"]

    def test_key_sections_related_work(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert "Related Work" in cp["key_sections"]

    def test_key_sections_conclusion(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert "Conclusion" in cp["key_sections"]

    def test_limitations_section_matched(self):
        cp = parse_tei(_TEI_WITH_LIMITATIONS)
        assert "Limitations" in cp["key_sections"]
        assert "future work" in cp["key_sections"]["Limitations"].lower()

    def test_future_work_section_matched(self):
        cp = parse_tei(_TEI_WITH_LIMITATIONS)
        assert "Future Work" in cp["key_sections"]
        assert "scaling" in cp["key_sections"]["Future Work"]

    def test_empty_body_returns_structure(self):
        cp = parse_tei(_TEI_EMPTY_BODY)
        assert cp["title"] == "Minimal Paper"
        assert cp["headings"] == []
        assert cp["sections"] == {}
        assert cp["key_sections"] == {}

    def test_accepts_bytes_input(self):
        cp = parse_tei(_TEI_MINIMAL.encode("utf-8"))
        assert cp["title"] != ""

    def test_accepts_string_input(self):
        cp = parse_tei(_TEI_MINIMAL)
        assert cp["title"] != ""

    def test_output_keys_present(self):
        cp = parse_tei(_TEI_MINIMAL)
        for key in ("title", "abstract", "headings", "sections", "key_sections"):
            assert key in cp, f"Missing key: {key}"

    def test_abstract_in_key_sections_when_no_section_match(self):
        """Abstract should be populated in key_sections via TEI abstract element."""
        cp = parse_tei(_TEI_MINIMAL)
        assert "Abstract" in cp["key_sections"]
        assert "abstract text" in cp["key_sections"]["Abstract"]

    def test_duplicate_headings_handled(self):
        """Sections with duplicate headings should not overwrite each other."""
        tei = textwrap.dedent(
            """<?xml version="1.0" encoding="UTF-8"?>
            <TEI xmlns="http://www.tei-c.org/ns/1.0">
              <teiHeader><fileDesc><titleStmt>
                <title type="main">Dup Test</title>
              </titleStmt></fileDesc></teiHeader>
              <text><body>
                <div><head>Experiments</head><p>First experiments block.</p></div>
                <div><head>Experiments</head><p>Second experiments block.</p></div>
              </body></text>
            </TEI>
            """
        )
        cp = parse_tei(tei)
        # Should not raise and should store both
        assert "Experiments" in cp["sections"]
        assert "Experiments (2)" in cp["sections"]
