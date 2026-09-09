import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import config
from .routers import chat, upload

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical RAG Assistant API",
    description=(
        "Portfolio/demo backend for document-aware medical information retrieval. "
        "Not a medical device and not a source of diagnosis or treatment decisions."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(chat.router)
app.include_router(upload.router)


@app.get("/")
async def root() -> dict:
    return {
        "name": "Medical RAG Assistant API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "disclaimer": "Informational software only; not a medical device.",
    }


@app.get("/health")
async def health_check() -> dict:
    """Side-effect-free readiness summary; does not call external services."""

    return {
        "status": "ok",
        "configuration": {
            "llm_configured": config.llm_configured(),
            "embeddings_configured": config.embeddings_configured(),
            "web_search_enabled": config.ENABLE_WEB_SEARCH,
            "graph_tool_enabled": config.ENABLE_GRAPH_TOOL,
            "admin_endpoints_enabled": config.ENABLE_ADMIN_ENDPOINTS,
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled request error: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "The request could not be completed.",
        },
    )
