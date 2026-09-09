from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import RLock


class SessionManager:
    """Small in-memory session store with bounded history and no hidden globals."""

    def __init__(self, max_messages_per_session: int = 40):
        self.max_messages_per_session = max_messages_per_session
        self.store: dict[str, deque[dict[str, str]]] = {}
        self._lock = RLock()

    def _session(self, session_id: str) -> deque[dict[str, str]]:
        with self._lock:
            if session_id not in self.store:
                self.store[session_id] = deque(maxlen=self.max_messages_per_session)
            return self.store[session_id]

    def format_chat_history(self, session_id: str, max_messages: int = 8) -> str:
        with self._lock:
            messages = list(self.store.get(session_id, ())) [-max_messages:]
        lines = []
        for message in messages:
            label = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{label}: {message['content']}")
        return "\n".join(lines)

    def add_exchange(self, session_id: str, user_message: str, assistant_message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        session = self._session(session_id)
        with self._lock:
            session.append({"role": "user", "content": user_message, "timestamp": timestamp})
            session.append({"role": "assistant", "content": assistant_message, "timestamp": timestamp})

    def get_history_list(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self.store.get(session_id, ())]

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            return self.store.pop(session_id, None) is not None

    def get_active_sessions(self) -> list[str]:
        with self._lock:
            return sorted(self.store)
