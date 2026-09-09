from app.schemas.file import FileUploadResponse


def test_upload_response_does_not_expose_internal_path_or_fake_medical_score() -> None:
    fields = set(FileUploadResponse.model_fields)
    assert "file_path" not in fields
    assert "text_content" not in fields
    assert "medical_analysis" not in fields
