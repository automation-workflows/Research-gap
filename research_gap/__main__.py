"""
research_gap/__main__.py – CLI entrypoint.

Run as:
    python -m research_gap [options]
    research-gap [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from research_gap.parsing import (
    download_arxiv_pdf,
    run_grobid,
    parse_tei,
)
from research_gap.gaps import extract_gaps_heuristic, extract_gaps_llm
from research_gap.prior_work.openalex import search_openalex
from research_gap.prior_work.semantic_scholar import search_semantic_scholar
from research_gap.prior_work.embeddings import rank_by_similarity
from research_gap.reporting import generate_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _resolve_pdfs(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    """Return list of PDF paths to process based on CLI arguments."""
    pdfs: list[Path] = []
    if args.arxiv_id:
        for aid in args.arxiv_id:
            pdf_path = download_arxiv_pdf(aid, out_dir / aid)
            pdfs.append(pdf_path)
    if args.pdf:
        for p in args.pdf:
            pdfs.append(Path(p))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        pdfs.extend(sorted(input_dir.glob("*.pdf")))
    return pdfs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """Run the full pipeline for a single PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %s → %s", pdf_path, out_dir)

    # 1. GROBID → TEI XML
    tei_path = out_dir / "paper.tei.xml"
    if tei_path.exists() and not args.force:
        logger.info("TEI XML already exists, skipping GROBID (use --force to rerun)")
        tei_xml = tei_path.read_bytes()
    else:
        tei_xml = run_grobid(pdf_path, grobid_url=args.grobid_url)
        tei_path.write_bytes(tei_xml)
        logger.info("Saved TEI XML → %s", tei_path)

    # 2. Parse TEI → context_pack.json
    context_pack = parse_tei(tei_xml)
    cp_path = out_dir / "context_pack.json"
    cp_path.write_text(json.dumps(context_pack, indent=2, ensure_ascii=False))
    logger.info("Saved context pack → %s", cp_path)

    # 3. Gap extraction
    if args.no_llm:
        logger.info("Running heuristic gap extraction (--no-llm)")
        gaps = extract_gaps_heuristic(context_pack)
    else:
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning(
                "No OPENAI_API_KEY found – falling back to heuristic mode. "
                "Set OPENAI_API_KEY or pass --openai-api-key, or use --no-llm."
            )
            gaps = extract_gaps_heuristic(context_pack)
        else:
            logger.info("Running LLM gap extraction")
            gaps = extract_gaps_llm(
                context_pack,
                api_key=api_key,
                model=args.llm_model,
            )

    gaps_path = out_dir / "gaps.json"
    gaps_path.write_text(json.dumps(gaps, indent=2, ensure_ascii=False))
    logger.info("Saved gaps → %s (%d gaps)", gaps_path, len(gaps))

    # 4. Prior-work verification
    s2_key = args.s2_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    oa_email = args.openalex_email or os.environ.get("OPENALEX_EMAIL", "")

    novelty_entries: list[dict] = []
    for gap_item in gaps:
        queries: list[str] = gap_item.get("prior_work_search_queries", [])
        if not queries:
            # Fall back to gap text itself
            queries = [gap_item.get("gap", "")[:120]]

        idea_texts: list[str] = [gap_item.get("gap", "")]
        for d in gap_item.get("non_incremental_directions", []):
            idea_texts.append(d.get("direction", ""))

        candidates: list[dict] = []
        for q in queries[:3]:  # limit to 3 queries per gap
            if q.strip():
                candidates += search_openalex(q, email=oa_email, top_k=args.top_k)
                candidates += search_semantic_scholar(q, api_key=s2_key, top_k=args.top_k)

        # Deduplicate by title
        seen: set[str] = set()
        unique_candidates: list[dict] = []
        for c in candidates:
            key = c.get("title", "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Rank by embedding similarity
        for idea in idea_texts:
            if idea.strip() and unique_candidates:
                ranked = rank_by_similarity(idea, unique_candidates)
                novelty_entries.append(
                    {
                        "gap": gap_item.get("gap", ""),
                        "idea": idea,
                        "nearest_prior_work": ranked[: args.top_k],
                    }
                )

    novelty_path = out_dir / "novelty_report.json"
    novelty_path.write_text(json.dumps(novelty_entries, indent=2, ensure_ascii=False))
    logger.info("Saved novelty report → %s", novelty_path)

    # 5. Generate report.md
    report_path = out_dir / "report.md"
    report_text = generate_report(context_pack, gaps, novelty_entries)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved report → %s", report_path)

    print(f"\n✓ Pipeline complete. Outputs in: {out_dir}")
    print(f"  context_pack.json  → {cp_path}")
    print(f"  gaps.json          → {gaps_path}")
    print(f"  novelty_report.json→ {novelty_path}")
    print(f"  report.md          → {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-gap",
        description="Research-gap analysis pipeline for arXiv papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  research-gap --arxiv-id 2604.24717v1 --out-dir out/2604.24717v1
  research-gap --pdf paper.pdf --out-dir out/paper --no-llm
  research-gap --input-dir papers/ --out-dir out/batch
        """,
    )

    # Input sources (at least one required)
    inp = parser.add_argument_group("Input sources (at least one required)")
    inp.add_argument(
        "--arxiv-id",
        metavar="ID",
        nargs="+",
        help="arXiv paper ID(s), e.g. 2604.24717v1",
    )
    inp.add_argument(
        "--pdf",
        metavar="PATH",
        nargs="+",
        help="Path(s) to local PDF file(s)",
    )
    inp.add_argument(
        "--input-dir",
        metavar="DIR",
        help="Directory containing PDF files for batch processing",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default="out",
        help="Output directory (default: out/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun GROBID even if TEI XML already exists",
    )

    # GROBID
    parser.add_argument(
        "--grobid-url",
        metavar="URL",
        default="http://localhost:8070",
        help="GROBID service URL (default: http://localhost:8070)",
    )

    # Gap extraction
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use heuristic gap extraction instead of LLM",
    )
    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        default="gpt-4o-mini",
        help="OpenAI model to use for gap extraction (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--openai-api-key",
        metavar="KEY",
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )

    # Prior-work
    parser.add_argument(
        "--top-k",
        metavar="N",
        type=int,
        default=10,
        help="Top-K candidate papers per query (default: 10)",
    )
    parser.add_argument(
        "--s2-api-key",
        metavar="KEY",
        help="Semantic Scholar API key (overrides SEMANTIC_SCHOLAR_API_KEY env var)",
    )
    parser.add_argument(
        "--openalex-email",
        metavar="EMAIL",
        help="Email for OpenAlex polite pool (overrides OPENALEX_EMAIL env var)",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if not any([args.arxiv_id, args.pdf, args.input_dir]):
        parser.error("Provide at least one of --arxiv-id, --pdf, or --input-dir.")

    out_dir = Path(args.out_dir)
    pdfs = _resolve_pdfs(args, out_dir)

    if not pdfs:
        logger.error("No PDF files found to process.")
        sys.exit(1)

    for pdf_path in pdfs:
        # For batch mode, create per-paper subdirectory
        if len(pdfs) > 1:
            paper_out = out_dir / pdf_path.stem
        else:
            paper_out = out_dir

        try:
            run_pipeline(pdf_path, paper_out, args)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", pdf_path, exc, exc_info=args.verbose)
            if len(pdfs) == 1:
                sys.exit(1)


if __name__ == "__main__":
    main()
