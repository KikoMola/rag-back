from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.conversation import Conversation, Message
from app.models.document import Collection, Document
from app.schemas.dashboard import (
    DashboardResponse,
    RecentCollectionActivity,
    RecentDocumentActivity,
    SearchCollectionResult,
    StatsResponse,
)
from app.services import ollama_service

router = APIRouter()


@router.get("", response_model=DashboardResponse)
async def get_dashboard(session: AsyncSession = Depends(get_session)):
    # Total documents
    total_docs_result = await session.execute(select(func.count(Document.id)))
    total_documents = total_docs_result.scalar() or 0

    # Total collections
    total_cols_result = await session.execute(select(func.count(Collection.id)))
    total_collections = total_cols_result.scalar() or 0

    # Ollama models
    ollama_status = await ollama_service.check_ollama_status()
    models = ollama_status.get("models", []) if ollama_status["status"] == "ok" else []

    # Recent documents (last 10)
    recent_docs_result = await session.execute(
        select(Document, Collection.name.label("collection_name"))
        .join(Collection, Document.collection_id == Collection.id)
        .order_by(Document.created_at.desc())
        .limit(10)
    )
    recent_documents = [
        RecentDocumentActivity(
            id=doc.id,
            filename=doc.filename,
            collection_name=col_name,
            status=doc.status,
            created_at=doc.created_at,
        )
        for doc, col_name in recent_docs_result.all()
    ]

    # Recent collections (last 10)
    recent_cols_result = await session.execute(
        select(Collection, func.count(Document.id).label("doc_count"))
        .outerjoin(Document)
        .group_by(Collection.id)
        .order_by(Collection.created_at.desc())
        .limit(10)
    )
    recent_collections = [
        RecentCollectionActivity(
            id=col.id,
            name=col.name,
            document_count=doc_count,
            created_at=col.created_at,
        )
        for col, doc_count in recent_cols_result.all()
    ]

    return DashboardResponse(
        stats=StatsResponse(
            total_documents=total_documents,
            total_collections=total_collections,
            models=models,
        ),
        recent_documents=recent_documents,
        recent_collections=recent_collections,
    )


@router.get("/search", response_model=list[SearchCollectionResult])
async def search_collections(
    q: str = "",
    session: AsyncSession = Depends(get_session),
):
    pattern = f"%{q}%"
    result = await session.execute(
        select(Collection, func.count(Document.id).label("doc_count"))
        .outerjoin(Document)
        .where(
            Collection.name.ilike(pattern)
            | Collection.description.ilike(pattern)
        )
        .group_by(Collection.id)
        .order_by(Collection.created_at.desc())
    )
    return [
        SearchCollectionResult(
            id=col.id,
            name=col.name,
            description=col.description,
            document_count=doc_count,
            created_at=col.created_at,
        )
        for col, doc_count in result.all()
    ]
