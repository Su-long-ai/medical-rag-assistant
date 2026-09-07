# Medical RAG Assistant API

A FastAPI backend for document-aware medical question answering. The project combines file ingestion, retrieval-augmented generation (RAG), optional OCR, conversational sessions and Neo4j-backed knowledge-graph tools behind a streaming API.

This repository is a technical portfolio project. It is **not a medical device** and must not be used as a source of diagnosis or treatment decisions.

## Highlights

- FastAPI backend with automatic OpenAPI documentation
- PDF, DOCX and TXT uploads
- text extraction with OCR fallback for scanned PDFs
- document chunking and Chroma-based retrieval
- configurable LLM providers
- Neo4j knowledge-graph integration
- streaming chat responses
- multi-session conversation history
- upload/status management endpoints
- local secret, upload and vector-store hygiene through `.gitignore`

## Architecture

```text
uploaded document
      │
      ▼
text extraction ── OCR fallback
      │
      ▼
chunking + embeddings
      │
      ▼
  vector retrieval ─────────────┐
                               │
user question ─► chat service ─┼─► agent / LLM ─► streaming response
                               │
                 Neo4j graph ──┘
```

## Repository layout

```text
app/
├── agents/       agent prompts, callbacks and medical-agent logic
├── routers/      chat and upload API routes
├── schemas/      Pydantic request/response models
├── services/     chat, file and session services
├── config.py     environment-based configuration
├── deps.py       shared integrations and dependency helpers
└── main.py       FastAPI application
run.py            development launcher
requirements.txt  Python dependencies
.env.example      configuration template
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Create local configuration:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/macOS
cp .env.example .env
```

Fill in only the services you intend to use. The template includes:

```text
DEEPSEEK_API_KEY
MOONSHOT_API_KEY
ZHIPU_API_KEY
TAVILY_API_KEY
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
EMBEDDING_MODEL
```

Start the API:

```bash
python run.py
```

or directly:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Main API routes

### Chat

```text
POST /api/chat/stream
GET  /api/chat/history/{session_id}
POST /api/chat/clear
GET  /api/chat/sessions
POST /api/chat/test-kg
```

### Uploads

```text
POST /api/upload/file
GET  /api/upload/supported-formats
GET  /api/upload/uploads-status
POST /api/upload/clear-uploads
```

### Service

```text
GET /
GET /health
```

## External services

Depending on the features you enable, the application can use:

- an LLM provider configured through environment variables
- Neo4j for graph-backed tools
- Chroma for local vector retrieval
- Tesseract for OCR of scanned documents

Tesseract is a host dependency and is not installed by `pip`.

## Data and security

- API keys and database credentials belong in `.env`, never in source code.
- `.env`, uploads, vector-store data, caches and local runtime artifacts are excluded from Git.
- Do not upload personal medical records to a shared/public deployment.
- Add authentication, authorization, rate limiting and deployment hardening before exposing the API outside a trusted development environment.

## Limitations

RAG and knowledge graphs can improve grounding, but they do not eliminate hallucinations, stale information, extraction errors or retrieval failures. OCR quality also depends on scan quality and language support.

Any health-related output should be treated as informational software output, not professional medical advice.

## License

MIT License. See `LICENSE`.
