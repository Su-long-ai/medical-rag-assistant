from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from tavily import TavilyClient
from zhipuai import ZhipuAI

from .config import config

logger = logging.getLogger(__name__)


def _required(value: str | None, label: str) -> str:
    if not value:
        raise RuntimeError(f"{label} is not configured")
    return value


@lru_cache()
def get_llm():
    """Create the configured chat model without embedding credentials in code."""

    if config.LLM_PROVIDER == "deepseek":
        key = _required(config.DEEPSEEK_API_KEY or config.LLM_API_KEY, "DeepSeek API key")
        return ChatDeepSeek(api_key=key, model=config.LLM_MODEL, temperature=0.1)

    if config.LLM_PROVIDER == "openai_compatible":
        key = _required(config.LLM_API_KEY, "LLM_API_KEY")
        base_url = _required(config.LLM_BASE_URL, "LLM_BASE_URL")
        return ChatOpenAI(
            base_url=base_url,
            api_key=key,
            model=config.LLM_MODEL,
            temperature=0.1,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")


@lru_cache()
def get_neo4j_driver():
    if not config.ENABLE_GRAPH_TOOL:
        raise RuntimeError("Knowledge-graph tool is disabled")
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, _required(config.NEO4J_PASSWORD, "NEO4J_PASSWORD")),
    )


_READ_ONLY_ENTITY_LOOKUP = """
MATCH (n)
WITH n, coalesce(n.name, n.title, n.label, '') AS display_name
WHERE toLower(toString(display_name)) CONTAINS toLower($needle)
OPTIONAL MATCH (n)-[r]-(m)
RETURN labels(n) AS labels,
       properties(n) AS entity,
       type(r) AS relationship,
       labels(m) AS neighbor_labels,
       properties(m) AS neighbor
LIMIT $limit
"""


def safe_graph_query(query_text: str) -> str:
    """Run a parameterized read-only entity lookup through the official driver."""

    if not config.ENABLE_GRAPH_TOOL:
        return json.dumps({"status": "disabled", "results": []}, ensure_ascii=False)

    needle = " ".join(str(query_text).split())[:160]

    def _lookup(transaction):
        records = transaction.run(_READ_ONLY_ENTITY_LOOKUP, needle=needle, limit=8)
        return [record.data() for record in records]

    try:
        with get_neo4j_driver().session() as session:
            rows = session.execute_read(_lookup)
        return json.dumps({"status": "ok", "results": rows}, ensure_ascii=False, default=str)
    except Exception:
        logger.exception("Knowledge-graph lookup failed")
        return json.dumps({"status": "unavailable", "results": []}, ensure_ascii=False)


@lru_cache()
def get_tavily_client() -> TavilyClient:
    if not config.ENABLE_WEB_SEARCH:
        raise RuntimeError("Web search is disabled")
    return TavilyClient(api_key=_required(config.TAVILY_API_KEY, "TAVILY_API_KEY"))


def search_web(query: str) -> dict[str, Any]:
    if not config.ENABLE_WEB_SEARCH:
        return {"status": "disabled", "results": []}
    try:
        result = get_tavily_client().search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
        return {"status": "ok", **result}
    except Exception:
        logger.exception("Web search failed")
        return {"status": "unavailable", "results": []}


class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


@lru_cache()
def get_embedding_model() -> Embeddings:
    return ZhipuEmbeddings(
        api_key=_required(config.ZHIPU_API_KEY, "ZHIPU_API_KEY"),
        model=config.EMBEDDING_MODEL,
    )


@lru_cache()
def get_vector_store() -> Chroma:
    os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
    return Chroma(
        collection_name="medical-rag-assistant",
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=get_embedding_model(),
    )


def has_indexed_documents() -> bool:
    if not config.embeddings_configured():
        return False
    try:
        payload = get_vector_store().get(limit=1)
        return bool(payload.get("ids"))
    except Exception:
        logger.exception("Unable to inspect vector-store document count")
        return False


def clear_vector_store() -> None:
    if not config.embeddings_configured():
        return
    try:
        get_vector_store().delete_collection()
    finally:
        get_vector_store.cache_clear()


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def add_documents_to_vector_store(
    *,
    text_content: str,
    filename: str,
    document_id: str,
) -> dict[str, Any]:
    """Chunk and index one uploaded document without storing internal paths."""

    try:
        document = Document(
            page_content=text_content,
            metadata={
                "filename": filename,
                "document_id": document_id,
                "source": "uploaded_document",
            },
        )
        chunks = get_text_splitter().split_documents([document])
        if not chunks:
            return {"success": False, "error_code": "no_chunks", "chunks_count": 0}
        get_vector_store().add_documents(chunks)
        return {"success": True, "chunks_count": len(chunks)}
    except Exception:
        logger.exception("Document indexing failed")
        return {"success": False, "error_code": "indexing_failed", "chunks_count": 0}


def search_relevant_documents(query: str, k: int | None = None) -> dict[str, Any]:
    try:
        results = get_vector_store().similarity_search_with_score(
            query,
            k=k or config.MAX_RETRIEVED_DOCS,
        )
        return {"success": True, "documents": results, "count": len(results)}
    except Exception:
        logger.exception("Document retrieval failed")
        return {"success": False, "documents": [], "count": 0, "error_code": "retrieval_failed"}
