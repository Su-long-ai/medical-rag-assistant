# Medical RAG Assistant API

A FastAPI backend for document-aware medical question answering. It accepts PDF, DOCX, and text uploads, stores document chunks in Chroma, and can use an LLM plus a Neo4j graph as supporting context.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

The API documentation is then available at `http://127.0.0.1:8000/docs`.

## Required services

- At least one configured LLM provider, depending on the route you use
- Neo4j for graph-backed functionality
- Tesseract installed on the host if OCR is required

## Safety and privacy

This is a technical demonstration, not a medical device or a source of diagnosis or treatment. Do not upload personal health information to a shared deployment. API keys, database credentials, uploads, and vector-store data are excluded from version control.

