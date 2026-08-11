"""
会话管理服务
"""
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import HumanMessage, AIMessage


class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建指定会话ID的聊天历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def format_chat_history(self, session_id: str, max_messages: int = 8) -> str:
        """
        将聊天历史格式化为字符串

        Args:
            session_id: 会话ID
            max_messages: 最多返回的消息数

        Returns:
            格式化的历史记录字符串
        """
        chat_history = self.get_session_history(session_id)
        if not chat_history.messages:
            return ""

        formatted_history = []
        recent_messages = chat_history.messages[-max_messages:]

        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Assistant: {msg.content}")

        return "\n".join(formatted_history)

    def add_exchange(self, session_id: str, user_message: str, ai_message: str):
        """添加一轮对话到历史记录"""
        chat_history = self.get_session_history(session_id)
        chat_history.add_user_message(user_message)
        chat_history.add_ai_message(ai_message)

    def get_history_list(self, session_id: str) -> list:
        """获取聊天历史列表"""
        if session_id not in self.store:
            return []

        chat_history = self.store[session_id]
        messages = chat_history.messages

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        return history

    def clear_session(self, session_id: str) -> bool:
        """清除会话记忆"""
        if session_id in self.store:
            del self.store[session_id]
            return True
        return False

    def get_active_sessions(self) -> list:
        """获取所有活跃会话ID"""
        return list(self.store.keys())