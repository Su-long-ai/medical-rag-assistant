from dataclasses import dataclass

from app.agents.smart_medical_agent import SmartMedicalAgent
from app.services.query_policy import QueryRoute


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return FakeResponse("safe answer")


def test_document_context_is_injected_without_tool_call() -> None:
    llm = FakeLLM()
    calls = []
    agent = SmartMedicalAgent(
        llm=llm,
        graph_query=lambda q: calls.append(("graph", q)),
        web_search=lambda q: calls.append(("web", q)),
    )
    answer = agent.answer(
        message="summarize",
        route=QueryRoute.DOCUMENT,
        document_context="retrieved evidence",
    )
    assert answer == "safe answer"
    assert "retrieved evidence" in llm.prompts[-1]
    assert calls == []


def test_web_route_calls_only_web_tool() -> None:
    llm = FakeLLM()
    calls = []
    agent = SmartMedicalAgent(
        llm=llm,
        graph_query=lambda q: calls.append(("graph", q)) or "graph",
        web_search=lambda q: calls.append(("web", q)) or "web result",
    )
    agent.answer(message="latest research", route=QueryRoute.WEB)
    assert calls == [("web", "latest research")]
    assert "web result" in llm.prompts[-1]
