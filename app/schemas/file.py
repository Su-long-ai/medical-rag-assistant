from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any


class FileUploadResponse(BaseModel):
    success: bool = Field(..., description="是否上传成功")
    filename: str = Field(..., description="文件名")
    message: str = Field(..., description="响应消息")
    text_length: Optional[int] = Field(None, description="提取的文本长度")
    file_path: Optional[str] = Field(None, description="文件保存路径")
    text_content: Optional[str] = Field(None, description="提取的文本内容")
    rag_info: Optional[Dict[str, Any]] = Field(None, description="RAG处理结果")
    medical_analysis: Optional[Dict[str, Any]] = Field(None, description="医疗分析结果")
    error: Optional[str] = Field(None, description="错误信息")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "medical_report.pdf",
                "message": "文件上传并完成RAG处理",
                "text_length": 1245,
                "file_path": "uploads/abc123_medical_report.pdf",
                "rag_info": {
                    "success": True,
                    "chunks_count": 5,
                    "message": "成功添加 5 个文档块到向量数据库"
                }
            }
        })