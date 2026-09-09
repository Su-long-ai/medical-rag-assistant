import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import config
from ..deps import clear_vector_store
from ..schemas.file import FileUploadResponse, UploadStatusResponse
from ..services.file_service import FileService
from ..services.upload_policy import UploadPolicyError, validate_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["uploads"])
file_service = FileService()


@router.post("/file", response_model=FileUploadResponse)
async def upload_file(file: Annotated[UploadFile, File(...)]) -> FileUploadResponse:
    filename = file.filename or ""
    content = await file.read(config.MAX_FILE_SIZE + 1)
    try:
        validate_upload(
            filename,
            len(content),
            allowed_extensions=config.ALLOWED_EXTENSIONS,
            max_size_bytes=config.MAX_FILE_SIZE,
        )
    except UploadPolicyError as exc:
        http_status = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if exc.code == "file_too_large"
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(
            status_code=http_status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    if not config.embeddings_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "embeddings_not_configured",
                "message": "Configure the embedding provider before indexing documents.",
            },
        )

    result = await file_service.process_file(content, filename)
    if not result.get("success"):
        return FileUploadResponse(
            success=False,
            filename=result.get("filename", filename),
            message=result.get("message", "The document could not be processed."),
            error_code=result.get("error_code", "processing_failed"),
        )

    return FileUploadResponse(
        success=True,
        filename=result["filename"],
        message="Document indexed successfully.",
        document_id=result["document_id"],
        text_length=result["text_length"],
        rag_info=result["rag_info"],
    )


@router.get("/supported-formats")
async def get_supported_formats() -> dict:
    return {
        "supported_extensions": sorted(config.ALLOWED_EXTENSIONS),
        "max_file_size_mb": round(config.MAX_FILE_SIZE / 1024 / 1024, 2),
        "max_pdf_pages": config.MAX_PDF_PAGES,
    }


@router.get("/uploads-status", response_model=UploadStatusResponse)
async def get_uploads_status() -> UploadStatusResponse:
    return UploadStatusResponse(**file_service.get_uploads_status())


@router.post("/clear-uploads")
async def clear_uploads() -> dict:
    if not config.ENABLE_ADMIN_ENDPOINTS:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled.")
    result = file_service.clear_uploads_directory()
    clear_vector_store()
    return {
        "success": True,
        "deleted_count": result["deleted_count"],
        "message": "Uploads and the local vector index were cleared.",
    }
