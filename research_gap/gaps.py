"""
research_gap/gaps.py – Gap extraction from a context pack.

Two modes
---------
Heuristic (--no-llm):
    Regex-based sentence extraction looking for patterns like "limitation",
    "future work", "remains", "unexplored", etc.

LLM (default):
    Uses OpenAI chat completions to produce a structured JSON response
    matching the gaps.json schema.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

_GAP_PATTERNS = re.compile(
    r"""
    \b(
        limitation[s]?
        | future\s+work
        | future\s+direction[s]?
        | remains?\s+(?:an?\s+)?open
        | unexplored
        | we\s+leave\s+for
        | open\s+problem[s]?
        | open\s+question[s]?
        | not\s+yet\s+addressed
        | beyond\s+the\s+scope
        | we\s+do\s+not\s+(?:address|consider|explore)
        | invite[s]?
        | left\s+(?:as\s+)?(?:an?\s+)?open
        | unresolved
        | warrant[s]?\s+further
        | requires?\s+further
        | one\s+(?:potential\s+)?avenue
        | an?\s+interesting\s+direction
        | has\s+yet\s+to\s+be
        | remain[s]?\s+(?:to\s+be\s+)?(?:explored|studied|investigated|unclear|unknown|an?\s+open)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def extract_gaps_heuristic(context_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract candidate gap sentences using regex patterns.

    Returns a list of dicts with keys:
        ``gap`` (str): the matched sentence
        ``evidence`` (list[dict]): each with ``section`` and ``quote``
        ``why_it_matters`` (str): empty string (heuristic mode)
        ``non_incremental_directions`` (list): empty (heuristic mode)
        ``prior_work_search_queries`` (list[str]): auto-generated from gap text
    """
    results: list[dict[str, Any]] = []
    seen_gaps: set[str] = set()

    # Prioritise key sections, fall back to all sections
    sections_to_scan: dict[str, str] = {}

    for sec_name in [
        "Limitations",
        "Future Work",
        "Discussion",
        "Conclusion",
        "Introduction",
    ]:
        text = context_pack.get("key_sections", {}).get(sec_name, "")
        if text:
            sections_to_scan[sec_name] = text

    # Also scan all other sections for completeness
    for heading, text in context_pack.get("sections", {}).items():
        if heading not in sections_to_scan:
            sections_to_scan[heading] = text

    for section_name, text in sections_to_scan.items():
        for sentence in _split_sentences(text):
            if not _GAP_PATTERNS.search(sentence):
                continue
            norm = re.sub(r"\s+", " ", sentence).strip()
            if norm in seen_gaps:
                continue
            seen_gaps.add(norm)
            results.append(
                {
                    "gap": norm,
                    "evidence": [{"section": section_name, "quote": norm}],
                    "why_it_matters": "",
                    "non_incremental_directions": [],
                    "prior_work_search_queries": _make_search_queries(norm),
                }
            )

    logger.info("Heuristic extraction found %d candidate gaps", len(results))
    return results


def _make_search_queries(gap_text: str) -> list[str]:
    """Generate simple keyword queries from a gap sentence."""
    # Strip stop words roughly by keeping words > 4 chars
    words = re.findall(r"\b[a-zA-Z]{5,}\b", gap_text)
    words = [w.lower() for w in words if w.lower() not in _STOP_WORDS]
    if not words:
        return [gap_text[:80]]
    return [" ".join(words[:6])]


_STOP_WORDS = {
    "about", "above", "after", "again", "although", "among", "around",
    "because", "before", "being", "between", "could", "during", "either",
    "every", "first", "found", "further", "however", "large", "might",
    "never", "other", "their", "there", "these", "those", "though",
    "through", "under", "where", "which", "while", "within", "would",
    "often", "since", "still", "such", "than", "that", "them", "then",
    "they", "this", "thus", "very", "when", "with",
}


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a research-gap analyst.

Given a context pack from a scientific paper (sections and their text),
extract research gaps with full evidence.

Output ONLY valid JSON – an array (no surrounding object) of gap objects.
Each object MUST have these exact keys:
  "gap": string – a concise 1–2 sentence description of the research gap
  "evidence": array of >=2 objects, each with:
      "section": string – section name where the quote appears
      "quote": string – VERBATIM substring from the provided text (must be exact)
  "why_it_matters": string – 1–2 sentences on scientific/practical importance
  "non_incremental_directions": array of objects, each with:
      "direction": string – a concrete proposed research direction
      "axis_of_difference": string – how it fundamentally differs from existing work
        (e.g., problem formulation, assumptions, evaluation target, modality)
  "prior_work_search_queries": array of 3–5 strings – keyword queries for Semantic Scholar/OpenAlex

Rules:
- Each quote MUST be a verbatim substring of the corresponding section text.
- Do NOT paraphrase or invent quotes.
- Non-incremental directions must be fundamentally different, not minor tweaks.
- Output ONLY the JSON array, no prose.
"""

_USER_PROMPT_TEMPLATE = """Paper title: {title}

Context sections:
{sections_text}

Extract all research gaps with evidence from the text above."""


def _build_sections_text(context_pack: dict[str, Any], max_chars: int = 12000) -> str:
    """Build a compact text representation of key sections."""
    parts: list[str] = []
    total = 0

    # Key sections first (most relevant)
    for sec_name, text in context_pack.get("key_sections", {}).items():
        if not text.strip():
            continue
        snippet = text[:3000] if len(text) > 3000 else text
        chunk = f"## {sec_name}\n{snippet}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    # Fill remaining budget from other sections
    for heading, text in context_pack.get("sections", {}).items():
        if heading in context_pack.get("key_sections", {}):
            continue
        if not text.strip():
            continue
        snippet = text[:1500] if len(text) > 1500 else text
        chunk = f"## {heading}\n{snippet}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


def _validate_quotes(gaps: list[dict], context_pack: dict[str, Any]) -> list[dict]:
    """Drop or repair evidence items whose quote is not a verbatim substring."""
    all_text: dict[str, str] = {}
    all_text.update(context_pack.get("key_sections", {}))
    all_text.update(context_pack.get("sections", {}))
    # Concatenated blob for loose matching
    full_text = "\n".join(all_text.values())

    repaired: list[dict] = []
    for gap_item in gaps:
        valid_evidence: list[dict] = []
        for ev in gap_item.get("evidence", []):
            quote = ev.get("quote", "")
            section = ev.get("section", "")
            # Check verbatim in the specific section first, then full text
            target = all_text.get(section, full_text)
            if quote and quote in target:
                valid_evidence.append(ev)
            else:
                logger.debug("Dropping non-verbatim quote: %.80s…", quote)
        gap_item["evidence"] = valid_evidence
        repaired.append(gap_item)
    return repaired


def extract_gaps_llm(
    context_pack: dict[str, Any],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    """Extract research gaps using an OpenAI LLM.

    Parameters
    ----------
    context_pack:
        Output of ``parse_tei``.
    api_key:
        OpenAI API key.
    model:
        Chat model to use (default: ``gpt-4o-mini``).

    Returns
    -------
    list[dict]
        Parsed and validated gap objects.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for LLM mode: pip install openai") from exc

    client = OpenAI(api_key=api_key)

    sections_text = _build_sections_text(context_pack)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        title=context_pack.get("title", "(unknown)"),
        sections_text=sections_text,
    )

    logger.debug("LLM prompt length: %d chars", len(user_prompt))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "[]"
    logger.debug("LLM raw response (first 500 chars): %s", raw[:500])

    # The model returns a JSON object wrapping the array when response_format=json_object
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        # Unwrap common wrapper keys
        for key in ("gaps", "results", "items", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Try to find any list value
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                parsed = []

    if not isinstance(parsed, list):
        logger.warning("Unexpected LLM response structure; returning empty list")
        return []

    # Validate and repair quotes
    gaps = _validate_quotes(parsed, context_pack)
    logger.info("LLM extraction produced %d gaps (after quote validation)", len(gaps))
    return gaps
