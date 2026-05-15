# research-gap

A **local, scriptable research-gap analysis pipeline** for arXiv papers.

Given one or more arXiv papers (by ID or PDF path), the pipeline:

1. Downloads the PDF (or accepts a local file / batch directory)
2. Converts the PDF to structured TEI XML via **GROBID**
3. Extracts all section headings and selected key sections into a `context_pack.json`
4. Identifies research gaps either via **heuristic patterns** (`--no-llm`) or an **LLM** (OpenAI)
5. Verifies novelty by searching **OpenAlex** and **Semantic Scholar**, ranking candidates with a local embedding model (`all-MiniLM-L6-v2`)
6. Generates a `report.md` summarising everything

All processing runs **locally** – no data is sent anywhere except to APIs you explicitly configure.

---

## Table of contents

- [Setup](#setup)
- [Start GROBID (Docker)](#start-grobid-docker)
- [Configuration (.env)](#configuration-env)
- [Usage](#usage)
  - [Download from arXiv and analyse](#download-from-arxiv-and-analyse)
  - [Analyse a local PDF](#analyse-a-local-pdf)
  - [Batch mode](#batch-mode)
  - [Heuristic mode (no LLM)](#heuristic-mode-no-llm)
- [Output files](#output-files)
- [Module structure](#module-structure)
- [Running tests](#running-tests)
- [FAQ](#faq)

---

## Setup

### Prerequisites

- Python 3.9+
- Docker (for GROBID)

### Install

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package (includes all runtime dependencies)
pip install -e ".[dev]"
```

> **Dependencies installed:** `requests`, `lxml`, `openai`, `python-dotenv`,
> `sentence-transformers`, `numpy`  
> **Dev extras:** `pytest`, `pytest-mock`, `responses`

---

## Start GROBID (Docker)

GROBID converts PDF files to structured TEI XML. Start it with Docker before running the pipeline:

```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1
```

The service will be available at `http://localhost:8070`. The pipeline uses this URL by default; override with `--grobid-url` if needed.

You can test that GROBID is running:

```bash
curl http://localhost:8070/api/isalive
```

---

## Configuration (.env)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For LLM mode | OpenAI API key for gap extraction |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Higher rate limits on S2 API |
| `OPENALEX_EMAIL` | Recommended | Joins OpenAlex polite pool for better throughput |

If `OPENAI_API_KEY` is not set, the pipeline automatically falls back to heuristic mode.

---

## Usage

### Download from arXiv and analyse

```bash
research-gap --arxiv-id 2604.24717v1 --out-dir out/2604.24717v1
```

This downloads the PDF, runs GROBID, extracts gaps with LLM (if `OPENAI_API_KEY` is set), verifies novelty, and writes all outputs to `out/2604.24717v1/`.

### Analyse a local PDF

```bash
research-gap --pdf paper.pdf --out-dir out/paper
```

### Batch mode

Process all PDFs in a directory:

```bash
research-gap --input-dir papers/ --out-dir out/batch
```

Each PDF gets its own subdirectory inside `out/batch/`.

### Heuristic mode (no LLM)

Skip the OpenAI call entirely – use fast regex-based gap extraction:

```bash
research-gap --arxiv-id 2604.24717v1 --out-dir out/2604.24717v1 --no-llm
```

### All options

```
usage: research-gap [-h] [--arxiv-id ID [ID ...]] [--pdf PATH [PATH ...]]
                    [--input-dir DIR] [--out-dir DIR] [--force]
                    [--grobid-url URL] [--no-llm] [--llm-model MODEL]
                    [--openai-api-key KEY] [--top-k N] [--s2-api-key KEY]
                    [--openalex-email EMAIL] [-v]

Input sources:
  --arxiv-id ID         arXiv paper ID(s), e.g. 2604.24717v1
  --pdf PATH            Path(s) to local PDF file(s)
  --input-dir DIR       Directory containing PDF files

Output:
  --out-dir DIR         Output directory (default: out/)
  --force               Rerun GROBID even if TEI XML already exists

GROBID:
  --grobid-url URL      GROBID service URL (default: http://localhost:8070)

Gap extraction:
  --no-llm              Use heuristic extraction instead of LLM
  --llm-model MODEL     OpenAI model (default: gpt-4o-mini)
  --openai-api-key KEY  Overrides OPENAI_API_KEY env var

Prior-work:
  --top-k N             Top-K candidate papers per query (default: 10)
  --s2-api-key KEY      Overrides SEMANTIC_SCHOLAR_API_KEY env var
  --openalex-email EMAIL Overrides OPENALEX_EMAIL env var

Misc:
  -v, --verbose         Enable debug logging
```

---

## Output files

All files are written to `--out-dir`:

| File | Description |
|---|---|
| `paper.pdf` | Downloaded PDF (when using `--arxiv-id`) |
| `paper.tei.xml` | GROBID TEI XML output |
| `context_pack.json` | Parsed sections: title, abstract, headings, key sections |
| `gaps.json` | Extracted research gaps with evidence, directions, queries |
| `novelty_report.json` | Ranked prior work per gap/direction with similarity + risk |
| `report.md` | Human-readable Markdown summary of all findings |

### `context_pack.json` schema

```json
{
  "title": "string",
  "abstract": "string",
  "headings": ["Introduction", "Related Work", ...],
  "sections": { "heading": "full section text", ... },
  "key_sections": {
    "Abstract": "...",
    "Introduction": "...",
    "Related Work": "...",
    "Experiments": "...",
    "Discussion": "...",
    "Limitations": "...",
    "Conclusion": "...",
    "Future Work": "..."
  }
}
```

### `gaps.json` schema

```json
[
  {
    "gap": "Concise 1-2 sentence description of the research gap",
    "evidence": [
      { "section": "Limitations", "quote": "Verbatim substring from paper" },
      { "section": "Conclusion",  "quote": "Another verbatim substring" }
    ],
    "why_it_matters": "Explanation of scientific/practical significance",
    "non_incremental_directions": [
      {
        "direction": "Concrete proposed research direction",
        "axis_of_difference": "problem formulation | assumptions | evaluation target | modality | ..."
      }
    ],
    "prior_work_search_queries": ["keyword query 1", "keyword query 2"]
  }
]
```

### `novelty_report.json` schema

```json
[
  {
    "gap": "...",
    "idea": "...",
    "nearest_prior_work": [
      {
        "title": "Paper Title",
        "authors": ["Author A", "Author B"],
        "year": 2023,
        "venue": "NeurIPS",
        "abstract": "...",
        "url": "https://...",
        "citation_count": 42,
        "source": "openalex | semantic_scholar",
        "similarity": 0.812,
        "risk": "low | medium | high"
      }
    ]
  }
]
```

Risk labels:
- 🟢 **low** (similarity < 0.55) – idea appears novel
- 🟡 **medium** (0.55–0.74) – related work exists, review carefully
- 🔴 **high** (≥ 0.75) – very similar work found, idea may not be novel

---

## Module structure

```
research_gap/
├── __init__.py
├── __main__.py          # CLI entrypoint (argparse, pipeline orchestration)
├── parsing.py           # arXiv download, GROBID call, TEI → context_pack
├── gaps.py              # Heuristic + LLM gap extraction
├── reporting.py         # report.md generation
└── prior_work/
    ├── __init__.py
    ├── openalex.py      # OpenAlex API client
    ├── semantic_scholar.py  # Semantic Scholar API client
    └── embeddings.py    # sentence-transformers similarity ranking

tests/
├── test_parsing.py      # TEI parsing unit tests
├── test_prior_work.py   # API clients with mocked HTTP
├── test_gaps.py         # Heuristic extraction tests
└── test_reporting.py    # Report generation tests
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests use `responses` to mock all HTTP calls – no network access required.

---

## FAQ

**Q: GROBID crashes / returns an error for my PDF.**  
A: Some PDFs are malformed or encrypted. Try a different version of the paper (e.g., `v2` instead of `v1`). You can also pass the raw text by pre-converting with `pdftotext` and submitting as plain text.

**Q: LLM mode returns empty gaps.**  
A: The LLM may not find explicit gap sentences. The heuristic mode (`--no-llm`) is more reliable for papers that mention gaps indirectly. Also check that `OPENAI_API_KEY` is set correctly.

**Q: How do I use a different OpenAI model?**  
A: Pass `--llm-model gpt-4o` (or any supported model). The default is `gpt-4o-mini` for cost efficiency.

**Q: Can I run without Semantic Scholar / OpenAlex?**  
A: Yes. If both APIs fail (e.g., due to rate limiting or no network), `novelty_report.json` will have empty `nearest_prior_work` arrays. The rest of the pipeline completes normally.

**Q: Are my papers / API keys sent anywhere?**  
A: PDFs are sent to your **local** GROBID instance only. Gap extraction text is sent to OpenAI (if using LLM mode). Search queries (not paper content) are sent to OpenAlex and Semantic Scholar.
