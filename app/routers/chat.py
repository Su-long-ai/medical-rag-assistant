import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import config
from ..deps import safe_graph_query
from ..schemas.chat import ChatHistoryResponse, ChatRequest, SessionClearRequest, SessionClearResponse
from ..services.chat_service import MedicalChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])
chat_service = MedicalChatService()


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    async def generate_response() -> AsyncGenerator[str, None]:
        async for event in chat_service.chat_stream(request.message, request.session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    return ChatHistoryResponse(session_id=session_id, messages=chat_service.get_chat_history(session_id))


@router.post("/clear", response_model=SessionClearResponse)
async def clear_session(request: SessionClearRequest) -> SessionClearResponse:
    success = chat_service.clear_session(request.session_id)
    return SessionClearResponse(
        success=success,
        message="Session cleared." if success else "Session did not exist.",
        session_id=request.session_id,
    )


@router.get("/sessions")
async def get_active_sessions() -> dict:
    if not config.ENABLE_ADMIN_ENDPOINTS:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled.")
    active_sessions = chat_service.session_manager.get_active_sessions()
    return {"active_sessions": active_sessions, "count": len(active_sessions)}


@router.post("/test-kg")
async def test_knowledge_graph(query: dict) -> dict:
    if not config.ENABLE_ADMIN_ENDPOINTS:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled.")
    question = str(query.get("question", ""))[:500]
    return {"success": True, "result": safe_graph_query(question)}
