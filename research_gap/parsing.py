"""
research_gap/parsing.py – PDF ingestion and TEI XML parsing.

Functions
---------
download_arxiv_pdf(arxiv_id, out_dir) -> Path
    Download a PDF from arXiv and save to out_dir.

run_grobid(pdf_path, grobid_url) -> bytes
    Submit a PDF to a running GROBID instance and return raw TEI XML bytes.

parse_tei(tei_xml) -> dict
    Parse GROBID TEI XML and return a context_pack dict with:
        - title: str
        - abstract: str
        - headings: list[str]
        - sections: dict[str, str]   (heading -> full text)
        - key_sections: dict[str, str]  (canonical name -> text)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import requests
from lxml import etree

logger = logging.getLogger(__name__)

# TEI namespace
_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Canonical section names and their matching patterns
_KEY_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "Abstract": re.compile(r"abstract", re.I),
    "Introduction": re.compile(r"intro(duction)?", re.I),
    "Related Work": re.compile(r"related\s+work|background|prior\s+work|literature", re.I),
    "Experiments": re.compile(r"experiment(s|al)?|evaluation|results", re.I),
    "Discussion": re.compile(r"discussion", re.I),
    "Limitations": re.compile(r"limitation(s)?", re.I),
    "Conclusion": re.compile(r"conclusion(s)?", re.I),
    "Future Work": re.compile(r"future\s+work|future\s+direction", re.I),
}

# ---------------------------------------------------------------------------
# arXiv download
# ---------------------------------------------------------------------------

_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


def download_arxiv_pdf(arxiv_id: str, out_dir: Path) -> Path:
    """Download a PDF from arXiv and save to *out_dir/paper.pdf*.

    Parameters
    ----------
    arxiv_id:
        arXiv identifier, e.g. ``"2604.24717v1"`` or ``"2604.24717"``.
    out_dir:
        Directory to save the PDF in; created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the downloaded PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "paper.pdf"

    if pdf_path.exists():
        logger.info("PDF already exists at %s, skipping download", pdf_path)
        return pdf_path

    url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    logger.info("Downloading %s → %s", url, pdf_path)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    with pdf_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)

    logger.info("Downloaded %.1f KB", pdf_path.stat().st_size / 1024)
    return pdf_path


# ---------------------------------------------------------------------------
# GROBID
# ---------------------------------------------------------------------------

_GROBID_ENDPOINT = "/api/processFulltextDocument"


def run_grobid(pdf_path: Path, grobid_url: str = "http://localhost:8070") -> bytes:
    """Convert a PDF to TEI XML using a running GROBID instance.

    Parameters
    ----------
    pdf_path:
        Local path to the PDF file.
    grobid_url:
        Base URL of the GROBID service (default: ``http://localhost:8070``).

    Returns
    -------
    bytes
        Raw TEI XML response from GROBID.

    Raises
    ------
    requests.HTTPError
        If GROBID returns a non-2xx response.
    """
    url = grobid_url.rstrip("/") + _GROBID_ENDPOINT
    logger.info("Sending %s to GROBID at %s", pdf_path, url)

    with Path(pdf_path).open("rb") as fh:
        resp = requests.post(
            url,
            files={"input": (Path(pdf_path).name, fh, "application/pdf")},
            data={"consolidateHeader": "1", "consolidateCitations": "0"},
            timeout=300,
        )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# TEI parsing helpers
# ---------------------------------------------------------------------------

def _text_of(node: etree._Element) -> str:
    """Return normalised whitespace-joined text of a node."""
    parts = node.itertext()
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _div_sections(root: etree._Element) -> list[tuple[str, str]]:
    """Return ``(heading, body_text)`` pairs for every <div> that has a <head>."""
    result: list[tuple[str, str]] = []
    for div in root.xpath("//tei:div[tei:head]", namespaces=_NS):
        heading_nodes = div.xpath("tei:head", namespaces=_NS)
        heading = _text_of(heading_nodes[0]) if heading_nodes else ""
        paragraphs = div.xpath(".//tei:p", namespaces=_NS)
        body = "\n\n".join(_text_of(p) for p in paragraphs).strip()
        if heading or body:
            result.append((heading, body))
    return result


def _match_key_sections(
    sections: list[tuple[str, str]],
) -> dict[str, str]:
    """Map canonical section names to their text using pattern matching."""
    key: dict[str, str] = {}
    for canonical, pattern in _KEY_SECTION_PATTERNS.items():
        for heading, body in sections:
            if pattern.search(heading):
                # Append if multiple matching sections exist
                if canonical in key:
                    key[canonical] = key[canonical] + "\n\n" + body
                else:
                    key[canonical] = body
    return key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_tei(tei_xml: bytes | str) -> dict[str, Any]:
    """Parse GROBID TEI XML into a context pack dictionary.

    Parameters
    ----------
    tei_xml:
        Raw TEI XML as bytes or str.

    Returns
    -------
    dict with keys:
        ``title`` (str),
        ``abstract`` (str),
        ``headings`` (list[str]),
        ``sections`` (dict[str, str]),
        ``key_sections`` (dict[str, str])
    """
    if isinstance(tei_xml, str):
        tei_xml = tei_xml.encode()

    try:
        root = etree.fromstring(tei_xml)
    except etree.XMLSyntaxError as exc:
        logger.error("Failed to parse TEI XML: %s", exc)
        raise

    # Title
    title_nodes = root.xpath(
        "//tei:titleStmt/tei:title[@type='main']", namespaces=_NS
    )
    title = _text_of(title_nodes[0]) if title_nodes else ""

    # Abstract – GROBID puts it in <abstract> under profileDesc;
    # some TEI variants use <div type="abstract"> inside <front>.
    abstract_nodes = root.xpath(
        "//tei:abstract | //tei:div[@type='abstract']", namespaces=_NS
    )
    abstract = _text_of(abstract_nodes[0]) if abstract_nodes else ""

    # All body sections
    sections_list = _div_sections(root)
    headings = [h for h, _ in sections_list if h]
    sections_dict: dict[str, str] = {}
    for heading, body in sections_list:
        if heading:
            # Deduplicate headings by appending numeric suffix
            if heading in sections_dict:
                idx = 2
                while f"{heading} ({idx})" in sections_dict:
                    idx += 1
                sections_dict[f"{heading} ({idx})"] = body
            else:
                sections_dict[heading] = body

    key_sections = _match_key_sections(sections_list)

    # If abstract not found via section matching, try TEI abstract element
    if "Abstract" not in key_sections and abstract:
        key_sections["Abstract"] = abstract

    return {
        "title": title,
        "abstract": abstract,
        "headings": headings,
        "sections": sections_dict,
        "key_sections": key_sections,
    }
