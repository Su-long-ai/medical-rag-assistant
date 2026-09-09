from types import SimpleNamespace

from app.services.chat_service import MedicalChatService
from app.services.query_policy import QueryRoute


class NoopAgent:
    def answer(self, **kwargs):
        return "unused"


def test_ordinary_question_retrieves_when_documents_exist() -> None:
    calls = []

    def search_documents(query: str):
        calls.append(query)
        document = SimpleNamespace(
            metadata={"filename": "report.pdf"},
            page_content="Treatment evidence from the uploaded report.",
        )
        return {"success": True, "documents": [(document, 0.12)]}

    service = MedicalChatService(
        agent=NoopAgent(),
        has_documents=lambda: True,
        search_documents=search_documents,
    )
    decision, _, context, sources = service.prepare("Summarize treatment options", "s")

    assert decision.route is QueryRoute.DOCUMENT
    assert calls == ["Summarize treatment options"]
    assert "Treatment evidence" in context
    assert sources == ["report.pdf"]


def test_time_sensitive_route_does_not_query_local_documents() -> None:
    calls = []
    service = MedicalChatService(
        agent=NoopAgent(),
        has_documents=lambda: True,
        search_documents=lambda query: calls.append(query) or {"success": True, "documents": []},
    )

    decision, _, context, sources = service.prepare("What is the latest clinical trial update?", "s")

    assert decision.route is QueryRoute.WEB
    assert calls == []
    assert context == ""
    assert sources == []
