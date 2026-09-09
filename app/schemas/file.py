from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RAGIndexInfo(BaseModel):
    chunks_count: int = Field(..., ge=0)


class FileUploadResponse(BaseModel):
    success: bool
    filename: str
    message: str
    document_id: Optional[str] = None
    text_length: Optional[int] = None
    rag_info: Optional[RAGIndexInfo] = None
    error_code: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "filename": "report.pdf",
                "message": "Document indexed successfully.",
                "document_id": "9f71c9e1f6d44fba8ef2e918730f710e",
                "text_length": 1245,
                "rag_info": {"chunks_count": 5},
            }
        }
    )


class UploadStatusResponse(BaseModel):
    file_count: int = Field(..., ge=0)
    total_size_bytes: int = Field(..., ge=0)
