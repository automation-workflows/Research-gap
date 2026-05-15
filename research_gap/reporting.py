"""
research_gap/reporting.py – Markdown report generation.

Produces a human-readable ``report.md`` summarising:
    - Paper title and extracted headings
    - Identified gaps with evidence quotes
    - Proposed non-incremental research directions
    - Novelty check results (nearest prior work with similarity + risk label)
"""

from __future__ import annotations

import textwrap
from typing import Any


def _md_escape(text: str) -> str:
    """Minimally escape Markdown special characters in plain text."""
    for ch in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", "!", "|"):
        text = text.replace(ch, "\\" + ch)
    return text


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " …"


def _render_headings(headings: list[str]) -> str:
    if not headings:
        return "_No headings extracted._\n"
    lines = [f"- {h}" for h in headings]
    return "\n".join(lines) + "\n"


def _render_gaps(gaps: list[dict[str, Any]]) -> str:
    if not gaps:
        return "_No gaps identified._\n"

    parts: list[str] = []
    for idx, gap in enumerate(gaps, start=1):
        gap_text = gap.get("gap", "")
        evidence = gap.get("evidence", [])
        why = gap.get("why_it_matters", "")
        directions = gap.get("non_incremental_directions", [])
        queries = gap.get("prior_work_search_queries", [])

        block: list[str] = [f"### Gap {idx}\n\n{gap_text}\n"]

        if evidence:
            block.append("**Evidence**\n")
            for ev in evidence:
                sec = ev.get("section", "")
                quote = ev.get("quote", "")
                block.append(f'> *[{sec}]* "{_truncate(quote)}"\n')

        if why:
            block.append(f"\n**Why it matters**\n\n{why}\n")

        if directions:
            block.append("\n**Non-incremental directions**\n")
            for d in directions:
                direction = d.get("direction", "")
                axis = d.get("axis_of_difference", "")
                block.append(f"- **{direction}**")
                if axis:
                    block.append(f"  *(Axis of difference: {axis})*")
                block.append("")

        if queries:
            block.append("\n**Prior-work search queries**\n")
            for q in queries:
                block.append(f"- `{q}`")
            block.append("")

        parts.append("\n".join(block))

    return "\n---\n\n".join(parts)


def _render_novelty(novelty_entries: list[dict[str, Any]]) -> str:
    if not novelty_entries:
        return "_No prior-work verification performed._\n"

    parts: list[str] = []
    for entry in novelty_entries:
        idea = entry.get("idea", "")
        gap = entry.get("gap", "")
        nearest = entry.get("nearest_prior_work", [])

        header = f"**Idea:** {_truncate(idea, 200)}"
        if gap and gap != idea:
            header = f"**Gap:** {_truncate(gap, 150)}\n\n{header}"

        rows: list[str] = []
        if nearest:
            rows.append("| # | Title | Year | Similarity | Risk | Source |")
            rows.append("|---|-------|------|-----------|------|--------|")
            for rank, paper in enumerate(nearest[:10], start=1):
                title = paper.get("title", "")
                url = paper.get("url", "")
                year = paper.get("year") or "?"
                sim = paper.get("similarity", 0.0)
                risk = paper.get("risk", "")
                source = paper.get("source", "")
                title_cell = f"[{title}]({url})" if url else title
                risk_badge = {"high": "🔴 high", "medium": "🟡 medium", "low": "🟢 low"}.get(
                    risk, risk
                )
                rows.append(
                    f"| {rank} | {title_cell} | {year} | {sim:.3f} | {risk_badge} | {source} |"
                )
        else:
            rows.append("_No candidate papers found._")

        parts.append(header + "\n\n" + "\n".join(rows) + "\n")

    return "\n---\n\n".join(parts)


def generate_report(
    context_pack: dict[str, Any],
    gaps: list[dict[str, Any]],
    novelty_entries: list[dict[str, Any]],
) -> str:
    """Generate a Markdown report string.

    Parameters
    ----------
    context_pack:
        Output of ``parse_tei``.
    gaps:
        Output of ``extract_gaps_heuristic`` or ``extract_gaps_llm``.
    novelty_entries:
        Output of the novelty-ranking step in ``__main__``.

    Returns
    -------
    str
        Full Markdown report text.
    """
    title = context_pack.get("title") or "Unknown Paper"
    headings = context_pack.get("headings", [])
    abstract = context_pack.get("abstract", "")

    sections: list[str] = []

    # Title
    sections.append(f"# Research Gap Analysis Report\n\n## {title}\n")

    # Abstract
    if abstract:
        sections.append(f"### Abstract\n\n{_truncate(abstract, 600)}\n")

    # Section headings
    sections.append("---\n\n## Document Structure (Extracted Headings)\n")
    sections.append(_render_headings(headings))

    # Gaps
    sections.append(f"\n---\n\n## Identified Research Gaps ({len(gaps)} total)\n")
    sections.append(_render_gaps(gaps))

    # Novelty check
    sections.append("\n---\n\n## Prior-Work Novelty Check\n")
    sections.append(_render_novelty(novelty_entries))

    sections.append(
        "\n---\n\n*Generated by [research-gap](https://github.com/automation-workflows/Research-gap)*\n"
    )

    return "\n".join(sections)
