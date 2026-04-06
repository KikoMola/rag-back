from datetime import datetime
from pydantic import BaseModel


class StatsResponse(BaseModel):
    total_documents: int
    total_collections: int
    models: list[str]


class RecentDocumentActivity(BaseModel):
    id: int
    filename: str
    collection_name: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class RecentCollectionActivity(BaseModel):
    id: int
    name: str
    document_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SearchCollectionResult(BaseModel):
    id: int
    name: str
    description: str | None
    document_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DashboardResponse(BaseModel):
    stats: StatsResponse
    recent_documents: list[RecentDocumentActivity]
    recent_collections: list[RecentCollectionActivity]
