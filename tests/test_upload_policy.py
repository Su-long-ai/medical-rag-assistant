import pytest

from app.services.upload_policy import UploadPolicyError, sanitize_filename, validate_upload


def test_path_components_are_removed_from_display_filename() -> None:
    assert sanitize_filename(r"..\..\private\report.pdf") == "report.pdf"
    assert sanitize_filename("../../private/report.pdf") == "report.pdf"


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UploadPolicyError) as error:
        validate_upload("payload.exe", 10, allowed_extensions={".pdf"}, max_size_bytes=100)
    assert error.value.code == "unsupported_file_type"


def test_file_size_limit_is_enforced() -> None:
    with pytest.raises(UploadPolicyError) as error:
        validate_upload("report.pdf", 101, allowed_extensions={".pdf"}, max_size_bytes=100)
    assert error.value.code == "file_too_large"
