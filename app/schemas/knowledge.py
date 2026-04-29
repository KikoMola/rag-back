from datetime import datetime
from pydantic import BaseModel


class CollectionCreate(BaseModel):
    name: str
    description: str | None = None


class CollectionResponse(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime
    document_count: int = 0

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: int
    collection_id: int
    filename: str
    filepath: str
    format: str
    size_bytes: int
    chunk_count: int
    status: str
    error_message: str | None
    summary: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    id: int
    filename: str
    status: str
    chunk_count: int
    error_message: str | None
    summary: str | None = None

    model_config = {"from_attributes": True}


class KnowledgeQuery(BaseModel):
    question: str
    collection_ids: list[int] | None = None
    top_k: int = 10


class SuggestedQuestionsResponse(BaseModel):
    collection_id: int
    questions: list[str]
    cached: bool = False


class CompareDocumentsRequest(BaseModel):
    document_id_a: int
    document_id_b: int
    question: str | None = None  # Pregunta de comparación opcional

