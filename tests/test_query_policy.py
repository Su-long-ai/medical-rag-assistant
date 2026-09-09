from app.services.query_policy import QueryRoute, choose_query_route


def test_uploaded_documents_are_not_silently_ignored() -> None:
    decision = choose_query_route("Summarize the treatment options", has_documents=True)
    assert decision.route is QueryRoute.DOCUMENT


def test_time_sensitive_query_prefers_web_without_explicit_document_intent() -> None:
    decision = choose_query_route("What is the latest clinical trial update?", has_documents=True)
    assert decision.route is QueryRoute.WEB


def test_explicit_document_intent_wins() -> None:
    decision = choose_query_route("根据文档总结最新治疗方案", has_documents=True)
    assert decision.route is QueryRoute.DOCUMENT


def test_graph_requires_explicit_intent_when_no_docs() -> None:
    decision = choose_query_route("查询知识图谱里的疾病关系", has_documents=False)
    assert decision.route is QueryRoute.GRAPH
