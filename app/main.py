from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from .routers import chat, upload
from .config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时的初始化
    print("🚀 医疗AI助手启动中...")

    # 检查必要的配置
    if not config.DEEPSEEK_API_KEY:
        print("⚠️  警告: 未设置 DEEPSEEK_API_KEY")

    try:
        # 测试Neo4j连接
        from .deps import get_neo4j_graph
        graph = get_neo4j_graph()
        graph.refresh_schema()
        print("✅ Neo4j 连接成功")
    except Exception as e:
        print(f"❌ Neo4j 连接失败: {e}")


    print("🎉 医疗AI助手启动完成!")
    yield
    # 关闭时的清理
    print("👋 医疗AI助手正在关闭...")


# 创建FastAPI应用
app = FastAPI(
    title="医疗AI助手",
    description="基于DeepSeek和Neo4j的智能医疗问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vue开发服务器地址
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(upload.router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "医疗AI助手API服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "upload": "/api/upload",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查各个组件状态
        status = {
            "api": "healthy",
            "deepseek": "unknown",
            "neo4j": "unknown",
            "embedding": "unknown"
        }

        # 检查DeepSeek API
        try:
            from .deps import get_llm
            llm = get_llm()
            status["deepseek"] = "healthy"
        except Exception as e:
            status["deepseek"] = f"error: {str(e)}"

        # 检查Neo4j
        try:
            from .deps import get_neo4j_graph
            graph = get_neo4j_graph()
            graph.query("RETURN 1 as test")
            status["neo4j"] = "healthy"
        except Exception as e:
            status["neo4j"] = f"error: {str(e)}"

        # 检查嵌入模型
        try:
            from .deps import get_embedding_model
            embedding_model = get_embedding_model()
            status["embedding"] = "healthy"
        except Exception as e:
            status["embedding"] = f"error: {str(e)}"

        return {"status": "healthy", "components": status}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc),
            "path": str(request.url)
        }
    )