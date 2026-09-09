from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, AsyncGenerator

from ..agents.smart_medical_agent import SmartMedicalAgent
from ..config import config
from ..deps import has_indexed_documents, search_relevant_documents
from .query_policy import QueryDecision, QueryRoute, choose_query_route, route_status_message
from .session_service import SessionManager

logger = logging.getLogger(__name__)


class MedicalChatService:
    def __init__(
        self,
        *,
        agent: SmartMedicalAgent | None = None,
        session_manager: SessionManager | None = None,
        has_documents: Callable[[], bool] = has_indexed_documents,
        search_documents: Callable[[str], dict[str, Any]] = search_relevant_documents,
    ):
        self.agent = agent or SmartMedicalAgent()
        self.session_manager = session_manager or SessionManager()
        self._has_documents = has_documents
        self._search_documents = search_documents

    @property
    def sessions(self):
        return self.session_manager.store

    def _document_context(self, message: str) -> tuple[str, list[str]]:
        result = self._search_documents(message)
        if not result.get("success"):
            return "", []

        snippets: list[str] = []
        sources: list[str] = []
        for document, score in result.get("documents", []):
            metadata = getattr(document, "metadata", {}) or {}
            filename = str(metadata.get("filename") or "uploaded document")
            content = str(getattr(document, "page_content", ""))
            if filename not in sources:
                sources.append(filename)
            snippets.append(
                f"[source={filename}; retrieval_score={score}]\n"
                f"{content[:config.MAX_CONTEXT_CHARS_PER_CHUNK]}"
            )
        return "\n\n".join(snippets), sources

    def prepare(self, message: str, session_id: str) -> tuple[QueryDecision, str, str, list[str]]:
        has_documents = self._has_documents()
        decision = choose_query_route(message, has_documents=has_documents)
        history = self.session_manager.format_chat_history(session_id)
        context = ""
        sources: list[str] = []
        if decision.route is QueryRoute.DOCUMENT:
            context, sources = self._document_context(message)
        return decision, history, context, sources

    async def chat_stream(self, message: str, session_id: str) -> AsyncGenerator[dict[str, Any], None]:
        try:
            decision, history, context, sources = self.prepare(message, session_id)
            yield {
                "type": "status",
                "content": route_status_message(decision.route),
                "route": decision.route.value,
            }

            response_text = await asyncio.to_thread(
                self.agent.answer,
                message=message,
                route=decision.route,
                chat_history=history,
                document_context=context,
            )
            self.session_manager.add_exchange(session_id, message, response_text)

            yield {
                "type": "meta",
                "content": "",
                "route": decision.route.value,
                "sources": sources,
            }

            chunk_size = 160
            for index in range(0, len(response_text), chunk_size):
                yield {"type": "chunk", "content": response_text[index:index + chunk_size]}
                await asyncio.sleep(0)

            yield {"type": "done", "content": ""}
        except Exception:
            logger.exception("Chat request failed")
            yield {
                "type": "error",
                "content": "The request could not be completed. Check server logs for details.",
            }

    def get_chat_history(self, session_id: str) -> list[dict[str, str]]:
        return self.session_manager.get_history_list(session_id)

    def clear_session(self, session_id: str) -> bool:
        return self.session_manager.clear_session(session_id)
