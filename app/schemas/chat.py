from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str = Field(..., min_length=1, max_length=128, description="Client-provided session ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Summarize the treatment options described in my uploaded report.",
                "session_id": "demo-session-1",
            }
        }
    )


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class SessionClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


class SessionClearResponse(BaseModel):
    success: bool
    message: str
    session_id: str
