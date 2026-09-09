from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_is_side_effect_free_and_reports_configuration() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert set(payload["configuration"]) == {
        "llm_configured",
        "embeddings_configured",
        "web_search_enabled",
        "graph_tool_enabled",
        "admin_endpoints_enabled",
    }


def test_admin_session_listing_is_disabled_by_default() -> None:
    response = client.get("/api/chat/sessions")
    assert response.status_code == 403


def test_invalid_upload_type_is_rejected_before_provider_access() -> None:
    response = client.post(
        "/api/upload/file",
        files={"file": ("payload.exe", b"not executable", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_file_type"
