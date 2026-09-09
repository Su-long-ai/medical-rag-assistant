from __future__ import annotations

from pathlib import Path
from typing import Collection


class UploadPolicyError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sanitize_filename(filename: str) -> str:
    """Return a display-only basename safe from path traversal."""

    normalized = str(filename or "").replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        raise UploadPolicyError("invalid_filename", "A valid filename is required.")
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        raise UploadPolicyError("invalid_filename", "The filename contains invalid characters.")
    if len(name) > 180:
        raise UploadPolicyError("filename_too_long", "The filename is too long.")
    return name


def validate_upload(
    filename: str,
    size_bytes: int,
    *,
    allowed_extensions: Collection[str],
    max_size_bytes: int,
) -> str:
    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in allowed_extensions:
        raise UploadPolicyError("unsupported_file_type", f"Unsupported file type: {extension or '(none)'}." )
    if size_bytes <= 0:
        raise UploadPolicyError("empty_file", "The uploaded file is empty.")
    if size_bytes > max_size_bytes:
        raise UploadPolicyError("file_too_large", "The uploaded file exceeds the configured size limit.")
    return safe_name
