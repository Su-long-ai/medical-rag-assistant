from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
from ..schemas.chat import ChatRequest, ChatHistoryResponse, SessionClearRequest, SessionClearResponse
from ..services.chat_service import MedicalChatService

router = APIRouter(prefix="/api/chat", tags=["聊天"])

# 全局聊天服务实例
chat_service = MedicalChatService()


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    try:
        async def generate_response() -> AsyncGenerator[str, None]:
            async for chunk in chat_service.chat_stream(request.message, request.session_id):
                # 转换为SSE格式
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天服务错误: {str(e)}")


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """获取聊天历史"""
    try:
        history = chat_service.get_chat_history(session_id)
        return ChatHistoryResponse(session_id=session_id, messages=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天历史失败: {str(e)}")


@router.post("/clear")
async def clear_session(request: SessionClearRequest) -> SessionClearResponse:
    """清除会话记忆"""
    try:
        success = chat_service.clear_session(request.session_id)
        if success:
            return SessionClearResponse(
                success=True,
                message="会话记忆已清除",
                session_id=request.session_id
            )
        else:
            return SessionClearResponse(
                success=False,
                message="会话不存在或已经被清除",
                session_id=request.session_id
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除会话失败: {str(e)}")


@router.get("/sessions")
async def get_active_sessions():
    """获取活跃会话列表"""
    try:
        active_sessions = list(chat_service.sessions.keys())
        return {
            "active_sessions": active_sessions,
            "count": len(active_sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@router.post("/test-kg")
async def test_knowledge_graph(query: dict):
    """测试知识图谱查询"""
    try:
        from ..deps import get_graph_tool
        graph_tool = get_graph_tool()

        result = graph_tool.func(query.get("question", ""))
        return {
            "success": True,
            "query": query.get("question", ""),
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query.get("question", "")
        }