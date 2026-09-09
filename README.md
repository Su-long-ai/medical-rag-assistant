# Medical RAG Assistant API

A **FastAPI + RAG portfolio backend** for document-aware medical information retrieval. It combines bounded file ingestion, Chroma retrieval, deterministic query routing, optional current-information search, optional read-only Neo4j lookup and SSE response delivery.

> This is informational software and a portfolio/demo project. It is **not a medical device** and should not be used for diagnosis or treatment decisions.

## What changed in the v2 refactor

The original prototype demonstrated a broad stack, but several implementation details weakened its credibility as a public portfolio project. v2 fixes the important ones:

- removed hard-coded `medical_analysis` confidence/score values;
- stopped returning internal file paths and extracted document text from upload responses;
- stores uploads under opaque document IDs instead of user filenames;
- replaced a deprecated free-form LangChain agent loop with deterministic route-and-synthesize orchestration;
- fixed a routing bug where ordinary questions could ignore uploaded documents unless the prompt explicitly mentioned “the document”;
- replaced LLM-generated Cypher with a parameterized read-only Neo4j lookup;
- removed wildcard CORS headers from the SSE route and centralized CORS policy;
- made health checks side-effect-free instead of calling external services;
- disabled destructive/debug endpoints by default;
- added upload-size/PDF-page bounds and offline tests.

## Architecture

```text
PDF / DOCX / TXT
      |
      v
upload policy -> opaque local document ID -> text extraction -> OCR fallback
      |                                                |
      +----------------------------------------------> chunking -> Chroma

user question
      |
      v
deterministic query policy
   | document       | time-sensitive       | graph intent      | direct
   v                v                      v                   v
Chroma RAG       web adapter       read-only Neo4j lookup     none
   \_____________________|______________________|_______________/
                         |
                         v
                 evidence-aware LLM
                         |
                         v
               SSE status/meta/chunks
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale.

## Core features

- FastAPI backend and OpenAPI docs
- PDF, DOCX and TXT ingestion
- pypdf extraction with bounded Tesseract/PyMuPDF OCR fallback
- Chroma vector retrieval with metadata that excludes local paths
- deterministic routing between document RAG, web context, graph context and direct response
- DeepSeek or OpenAI-compatible chat model configuration
- optional Tavily current-information search
- optional parameterized read-only Neo4j entity/relationship lookup
- bounded in-memory multi-session history
- Server-Sent Events for incremental status, source metadata and answer chunks
- environment-only credentials and privacy-focused public response schemas
- unit tests for policy and safety boundaries

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then configure at minimum:

```text
DEEPSEEK_API_KEY=...
ZHIPU_API_KEY=...
```

Start the development server:

```bash
python run.py
```

Then open `http://127.0.0.1:8000/docs`.

## Query routing

Routing is intentionally deterministic and covered by tests:

1. explicit document intent + indexed documents -> document RAG;
2. explicitly time-sensitive/latest question -> configured web-search route;
3. otherwise, if uploaded documents exist -> document RAG;
4. explicit graph/relationship intent with no indexed documents -> read-only graph route;
5. otherwise -> direct informational response.

This makes it possible to explain and test why a query used a given evidence source.

## API routes

```text
POST /api/upload/file
GET  /api/upload/supported-formats
GET  /api/upload/uploads-status
POST /api/upload/clear-uploads      # disabled by default

POST /api/chat/stream
GET  /api/chat/history/{session_id}
POST /api/chat/clear
GET  /api/chat/sessions             # disabled by default
POST /api/chat/test-kg              # disabled by default

GET  /
GET  /health
```

The `/api/chat/stream` endpoint uses SSE. The stream contains `status`, `meta`, `chunk`, `done` or `error` events. This is incremental HTTP delivery, not a claim of provider-level token streaming.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check app tests
python -m compileall -q app run.py
```

The repository includes tests for:

- document-aware query routing;
- time-sensitive route precedence;
- path-traversal-resistant filename handling;
- upload size/type limits;
- bounded session history;
- graph/web tool isolation;
- public upload-schema privacy boundaries.

A GitHub Actions workflow template is included at `docs/github-actions-ci.example.yml`. It can be moved to `.github/workflows/ci.yml` when the publishing token has GitHub `workflow` scope.

## Security and data boundaries

By default, graph search, web search and destructive/debug endpoints are disabled. Public API responses do not return internal upload paths or extracted document text. See [`docs/SECURITY_PRIVACY.md`](docs/SECURITY_PRIVACY.md).

Do not upload real personal medical records to a shared/public instance. An internet-facing deployment needs authentication, per-user isolation, rate limiting, encrypted storage, retention/deletion controls and a deployment-specific threat model.

## Research / product limitations

This repository does **not** contain evidence of clinical accuracy, diagnostic performance, retrieval benchmark superiority or human-subject validation. RAG can improve grounding but cannot guarantee correctness, freshness or completeness. OCR and retrieval can fail, and external search results can be low quality.

## License

MIT License. See `LICENSE`.
