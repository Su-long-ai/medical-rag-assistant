from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryRoute(str, Enum):
    DOCUMENT = "document"
    WEB = "web"
    GRAPH = "graph"
    DIRECT = "direct"


@dataclass(frozen=True, slots=True)
class QueryDecision:
    route: QueryRoute
    reason: str


_DOCUMENT_TERMS = (
    "根据文档", "根据文件", "文档中", "文件中", "报告中", "上传的", "刚才的文档",
    "document", "uploaded file", "according to the file", "according to the document",
)
_WEB_TERMS = (
    "最新", "近期", "今天", "现在", "当前指南", "新药", "临床试验", "最新研究",
    "latest", "recent", "today", "current guideline", "clinical trial", "new drug",
)
_GRAPH_TERMS = (
    "知识图谱", "实体关系", "疾病关系", "药物关系", "graph", "relationship between",
)


def choose_query_route(message: str, *, has_documents: bool) -> QueryDecision:
    """Choose a transparent, deterministic retrieval route.

    Explicit document intent wins. Time-sensitive questions prefer web search.
    When an index already contains uploaded documents, ordinary informational
    questions default to document retrieval rather than silently ignoring the
    user's corpus. Graph lookup is reserved for explicit relationship intent.
    """

    text = " ".join(message.lower().split())
    if any(term.lower() in text for term in _DOCUMENT_TERMS):
        if has_documents:
            return QueryDecision(QueryRoute.DOCUMENT, "explicit_document_intent")
        return QueryDecision(QueryRoute.DIRECT, "document_requested_but_index_empty")

    if any(term.lower() in text for term in _WEB_TERMS):
        return QueryDecision(QueryRoute.WEB, "time_sensitive_query")

    if has_documents:
        return QueryDecision(QueryRoute.DOCUMENT, "indexed_documents_available")

    if any(term.lower() in text for term in _GRAPH_TERMS):
        return QueryDecision(QueryRoute.GRAPH, "explicit_graph_intent")

    return QueryDecision(QueryRoute.DIRECT, "no_external_context_required")


def route_status_message(route: QueryRoute) -> str:
    return {
        QueryRoute.DOCUMENT: "Retrieving relevant uploaded-document context...",
        QueryRoute.WEB: "Checking configured web-search context...",
        QueryRoute.GRAPH: "Checking configured knowledge-graph context...",
        QueryRoute.DIRECT: "Preparing an informational response...",
    }[route]
