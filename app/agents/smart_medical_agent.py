from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from ..deps import get_llm, safe_graph_query, search_web
from ..services.query_policy import QueryRoute
from .prompts import MEDICAL_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _render_tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


class SmartMedicalAgent:
    """Deterministic route-and-synthesize assistant.

    Routing, external retrieval and final synthesis are deliberately separated
    so each decision can be tested without a free-form agent loop.
    """

    def __init__(
        self,
        *,
        llm: Any | None = None,
        graph_query: Callable[[str], Any] | None = None,
        web_search: Callable[[str], Any] | None = None,
    ):
        self._llm = llm
        self._graph_query = graph_query or safe_graph_query
        self._web_search = web_search or search_web

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _graph_context(self, message: str) -> str:
        try:
            return _render_tool_output(self._graph_query(message))
        except Exception:
            logger.exception("Graph tool failed")
            return "Knowledge-graph context is unavailable."

    def _web_context(self, message: str) -> str:
        try:
            return _render_tool_output(self._web_search(message))
        except Exception:
            logger.exception("Web-search tool failed")
            return "Web-search context is unavailable."

    def answer(
        self,
        *,
        message: str,
        route: QueryRoute,
        chat_history: str = "",
        document_context: str = "",
    ) -> str:
        evidence_label = "No external context was used."
        evidence = ""

        if route is QueryRoute.DOCUMENT:
            evidence_label = "Retrieved uploaded-document context:"
            evidence = document_context or "No relevant document chunks were retrieved."
        elif route is QueryRoute.WEB:
            evidence_label = "Configured web-search context:"
            evidence = self._web_context(message)
        elif route is QueryRoute.GRAPH:
            evidence_label = "Configured knowledge-graph context:"
            evidence = self._graph_context(message)

        prompt = f"""
{MEDICAL_AGENT_SYSTEM_PROMPT}

Conversation history (may be empty):
{chat_history or '(none)'}

{evidence_label}
{evidence or '(none)'}

User question:
{message}

Answer rules:
- Prefer supplied evidence when it is relevant.
- Do not imply that a retrieved snippet, graph row or web result was verified if it was not.
- If evidence is missing or insufficient, clearly state the limitation.
- Keep the answer informational and avoid personalized diagnosis/prescription claims.
""".strip()

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def extract_response_text(response: Any) -> str:
        if hasattr(response, "content"):
            return str(response.content)
        if isinstance(response, dict):
            for key in ("output", "result", "answer", "content"):
                if key in response:
                    return str(response[key])
        return str(response)
