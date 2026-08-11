from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    session_id: str = Field(..., min_length=1, description="会话ID")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "message": "什么是高血压？有哪些症状？",
                "session_id": "user_123_session_1"
            }
        })


class StreamChunk(BaseModel):
    type: str = Field(..., description="数据类型: token, chunk, status, done, error")
    content: str = Field(..., description="内容")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "type": "chunk",
                "content": "高血压是一种常见的心血管疾病"
            }
        })


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "session_id": "user_123_session_1",
                "messages": [
                    {
                        "role": "user",
                        "content": "什么是高血压？",
                        "timestamp": "2025-01-15T10:30:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "高血压是指血液在血管中流动时对血管壁产生的压力持续升高的疾病...",
                        "timestamp": "2025-01-15T10:30:15Z"
                    }
                ]
            }
        })


class SessionClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="要清除的会话ID")


class SessionClearResponse(BaseModel):
    success: bool
    message: str
    session_id: str