from datetime import datetime

from pydantic import BaseModel


class OcrDocumentResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    format: str
    size_bytes: int
    page_count: int
    status: str
    error_message: str | None
    markdown_content: str | None
    thumbnail_b64: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class OcrDocumentListResponse(BaseModel):
    id: int
    filename: str
    format: str
    size_bytes: int
    page_count: int
    status: str
    error_message: str | None
    thumbnail_b64: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
