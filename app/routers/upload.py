"""
文件上传和文件管理路由
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import logging

from ..schemas.file import FileUploadResponse
from ..services import FileService
from ..config import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["文件上传"])

# 全局服务实例
file_service = FileService()


@router.post("/file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """文件上传接口"""
    try:
        logger.info(f"🚀 开始处理文件上传: {file.filename}")

        # 验证文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        logger.info(f"📁 文件扩展名: {file_ext}")

        if file_ext not in config.ALLOWED_EXTENSIONS:
            logger.error(f"❌ 不支持的文件类型: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )

        # 读取文件内容
        logger.info("📖 开始读取文件内容...")
        file_content = await file.read()
        logger.info(f"📊 文件大小: {len(file_content)} bytes ({len(file_content)/1024:.2f} KB)")

        # 验证文件大小
        if len(file_content) > config.MAX_FILE_SIZE:
            logger.error(f"❌ 文件大小超过限制: {len(file_content)} > {config.MAX_FILE_SIZE}")
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制 ({config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB)"
            )

        # 处理文件
        logger.info("🔄 开始处理文件（提取文本 + RAG向量化）...")
        result = await file_service.process_file(file_content, file.filename)

        logger.info(f"✅ 文件处理完成，结果: {result.get('success', False)}")

        if result["success"]:
            text_length = result.get('text_length', 0)
            rag_info = result.get('rag_info', {})
            file_path = result.get('file_path', '')

            logger.info(f"📝 文本提取成功: {text_length} 字符")
            logger.info(f"🧠 RAG处理状态: {rag_info.get('success', False)}")
            logger.info(f"📦 文档分块数量: {rag_info.get('chunks_count', 0)}")
            logger.info(f"💬 RAG处理消息: {rag_info.get('message', '')}")

            # 构建详细的响应消息
            if rag_info.get('success'):
                chunks_count = rag_info.get('chunks_count', 0)
                message = f"文件上传成功并完成RAG处理，已分割为 {chunks_count} 个文档块可供检索"
            else:
                message = "文件上传成功，但RAG处理失败"

            return FileUploadResponse(
                success=True,
                filename=file.filename,
                message=message,
                text_length=result["text_length"],
                file_path=file_path,
                rag_info=rag_info,
                medical_analysis=result.get("medical_analysis", {})
            )
        else:
            error_msg = result.get('error', '未知错误')
            logger.error(f"❌ 文件处理失败: {error_msg}")
            return FileUploadResponse(
                success=False,
                filename=file.filename,
                message="文件处理失败",
                error=result["error"]
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 服务器内部错误: {str(e)}")
        import traceback
        logger.error(f"📋 错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")




@router.get("/supported-formats")
async def get_supported_formats():
    """获取支持的文件格式"""
    return {
        "supported_extensions": list(config.ALLOWED_EXTENSIONS),
        "max_file_size_mb": config.MAX_FILE_SIZE / 1024 / 1024
    }


@router.get("/uploads-status")
async def get_uploads_status():
    """获取uploads目录状态"""
    try:
        logger.info("📊 获取uploads目录状态请求")

        status = file_service.get_uploads_status()

        return {
            "file_count": status.get("file_count", 0),
            "files": status.get("files", []),
            "total_size": status.get("total_size", "0B"),
            "directory_path": config.UPLOAD_DIR
        }

    except Exception as e:
        logger.error(f"❌ 获取uploads目录状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取uploads目录状态失败: {str(e)}")


@router.post("/clear-uploads")
async def clear_uploads():
    """清空uploads目录中的所有文件"""
    try:
        logger.info("🗑️ 收到清空uploads目录请求")

        result = file_service.clear_uploads_directory()

        return {
            "success": result.get("success", False),
            "message": result.get("message", "清空uploads目录操作完成"),
            "deleted_count": result.get("deleted_count", 0),
            "deleted_files": result.get("deleted_files", [])
        }

    except Exception as e:
        logger.error(f"❌ 清空uploads目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空uploads目录失败: {str(e)}")


