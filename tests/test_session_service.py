from app.services.session_service import SessionManager


def test_session_history_is_bounded() -> None:
    manager = SessionManager(max_messages_per_session=4)
    manager.add_exchange("s", "u1", "a1")
    manager.add_exchange("s", "u2", "a2")
    manager.add_exchange("s", "u3", "a3")
    history = manager.get_history_list("s")
    assert [item["content"] for item in history] == ["u2", "a2", "u3", "a3"]


def test_clear_session_reports_presence() -> None:
    manager = SessionManager()
    manager.add_exchange("s", "hello", "hi")
    assert manager.clear_session("s") is True
    assert manager.clear_session("s") is False
