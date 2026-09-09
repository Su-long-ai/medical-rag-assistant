from __future__ import annotations

import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import docx
import pytesseract
from PIL import Image
from pypdf import PdfReader

from ..config import config
from ..deps import add_documents_to_vector_store
from .upload_policy import UploadPolicyError, validate_upload

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self):
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    async def save_file(self, file_content: bytes, safe_filename: str, document_id: str) -> str:
        extension = Path(safe_filename).suffix.lower()
        stored_name = f"{document_id}{extension}"
        file_path = os.path.join(config.UPLOAD_DIR, stored_name)
        async with aiofiles.open(file_path, "wb") as handle:
            await handle.write(file_content)
        return file_path

    def extract_text_from_file(self, file_path: str, filename: str) -> str:
        extension = Path(filename).suffix.lower()
        if extension == ".pdf":
            return self._extract_from_pdf(file_path)
        if extension == ".docx":
            return self._extract_from_docx(file_path)
        if extension == ".txt":
            return self._extract_from_txt(file_path)
        raise ValueError("Unsupported file type")

    def _extract_from_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        if len(reader.pages) > config.MAX_PDF_PAGES:
            raise ValueError(f"PDF exceeds the {config.MAX_PDF_PAGES}-page processing limit")

        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text
        return self._extract_with_ocr(file_path)

    def _extract_with_ocr(self, file_path: str) -> str:
        try:
            pytesseract.get_tesseract_version()
            import fitz
        except Exception as exc:
            raise ValueError("OCR fallback is unavailable on this host") from exc

        output: list[str] = []
        document = fitz.open(file_path)
        try:
            if len(document) > config.MAX_PDF_PAGES:
                raise ValueError(f"PDF exceeds the {config.MAX_PDF_PAGES}-page processing limit")
            for page in document:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = pytesseract.image_to_string(image, lang=config.OCR_LANGUAGES)
                if page_text.strip():
                    output.append(page_text)
        finally:
            document.close()
        return "\n".join(output).strip()

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        document = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()

    async def process_file(self, file_content: bytes, filename: str) -> dict[str, Any]:
        file_path: str | None = None
        try:
            safe_filename = validate_upload(
                filename,
                len(file_content),
                allowed_extensions=config.ALLOWED_EXTENSIONS,
                max_size_bytes=config.MAX_FILE_SIZE,
            )
            document_id = uuid.uuid4().hex
            file_path = await self.save_file(file_content, safe_filename, document_id)
            text_content = self.extract_text_from_file(file_path, safe_filename)
            if not text_content:
                raise ValueError("No text could be extracted from the file")

            index_result = add_documents_to_vector_store(
                text_content=text_content,
                filename=safe_filename,
                document_id=document_id,
            )
            if not index_result.get("success"):
                self.cleanup_file(file_path)
                return {
                    "success": False,
                    "filename": safe_filename,
                    "error_code": index_result.get("error_code", "indexing_failed"),
                    "message": "The document could not be indexed.",
                }

            return {
                "success": True,
                "filename": safe_filename,
                "document_id": document_id,
                "text_length": len(text_content),
                "rag_info": {"chunks_count": int(index_result["chunks_count"])},
            }
        except UploadPolicyError as exc:
            return {
                "success": False,
                "filename": str(filename or ""),
                "error_code": exc.code,
                "message": str(exc),
            }
        except Exception:
            logger.exception("File processing failed")
            if file_path:
                self.cleanup_file(file_path)
            return {
                "success": False,
                "filename": str(filename or ""),
                "error_code": "processing_failed",
                "message": "The file could not be processed.",
            }

    def get_uploads_status(self) -> dict[str, Any]:
        root = Path(config.UPLOAD_DIR)
        if not root.exists():
            return {"file_count": 0, "total_size_bytes": 0}
        files = [path for path in root.iterdir() if path.is_file()]
        return {
            "file_count": len(files),
            "total_size_bytes": sum(path.stat().st_size for path in files),
        }

    def clear_uploads_directory(self) -> dict[str, Any]:
        root = Path(config.UPLOAD_DIR)
        deleted = 0
        if root.exists():
            for path in root.iterdir():
                if path.is_file():
                    path.unlink()
                    deleted += 1
        return {"success": True, "deleted_count": deleted}

    @staticmethod
    def cleanup_file(file_path: str) -> None:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            logger.exception("Unable to clean up temporary upload")
