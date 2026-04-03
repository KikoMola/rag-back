import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.database import get_session
from app.models.document import Collection, Document
from app.schemas.knowledge import (
    CollectionCreate,
    CollectionResponse,
    DocumentResponse,
    DocumentStatusResponse,
    KnowledgeQuery,
)
from app.services import rag_service
from app.services.document_processor import get_format_from_filename
from app.vectorstore import chroma_client

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".epub", ".html", ".htm", ".txt", ".md"}


# ─── Colecciones ─────────────────────────────────────────────────────

@router.post("/collections", response_model=CollectionResponse, status_code=201)
async def create_collection(
    body: CollectionCreate,
    session: AsyncSession = Depends(get_session),
):
    collection = Collection(name=body.name, description=body.description)
    session.add(collection)
    await session.commit()
    await session.refresh(collection)
    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        created_at=collection.created_at,
        document_count=0,
    )


@router.get("/collections", response_model=list[CollectionResponse])
async def list_collections(session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(
            Collection,
            func.count(Document.id).label("doc_count"),
        )
        .outerjoin(Document)
        .group_by(Collection.id)
        .order_by(Collection.created_at.desc())
    )
    rows = result.all()
    return [
        CollectionResponse(
            id=col.id,
            name=col.name,
            description=col.description,
            created_at=col.created_at,
            document_count=doc_count,
        )
        for col, doc_count in rows
    ]


@router.delete("/collections/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Collection)
        .where(Collection.id == collection_id)
        .options(selectinload(Collection.documents))
    )
    collection = result.scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Colección no encontrada")

    chroma_client.delete_collection(f"knowledge_col_{collection_id}")

    await session.delete(collection)
    await session.commit()


# ─── Documentos ──────────────────────────────────────────────────────

@router.post(
    "/collections/{collection_id}/documents",
    response_model=list[DocumentResponse],
    status_code=201,
)
async def upload_documents(
    collection_id: int,
    files: list[UploadFile],
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Colección no encontrada")

    documents: list[Document] = []

    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado: {ext}. "
                       f"Válidos: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        collection_dir = settings.uploads_dir / str(collection_id)
        collection_dir.mkdir(parents=True, exist_ok=True)
        filepath = collection_dir / file.filename
        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = filepath.stat().st_size

        doc = Document(
            collection_id=collection_id,
            filename=file.filename,
            filepath=str(filepath),
            format=get_format_from_filename(file.filename),
            size_bytes=file_size,
            status="pending",
        )
        session.add(doc)
        documents.append(doc)

    await session.commit()

    for doc in documents:
        await session.refresh(doc)

    for doc in documents:
        asyncio.create_task(rag_service.ingest_document(doc.id))

    return documents


@router.get(
    "/collections/{collection_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    collection_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Document)
        .where(Document.collection_id == collection_id)
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.get(
    "/collections/{collection_id}/documents/{document_id}",
    response_model=DocumentStatusResponse,
)
async def get_document_status(
    collection_id: int,
    document_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.collection_id == collection_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return doc


@router.delete(
    "/collections/{collection_id}/documents/{document_id}",
    status_code=204,
)
async def delete_document(
    collection_id: int,
    document_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.collection_id == collection_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    rag_service.remove_document_from_index(doc.id, doc.collection_id)

    filepath = Path(doc.filepath)
    if filepath.exists():
        filepath.unlink()

    await session.delete(doc)
    await session.commit()


# ─── Query RAG ───────────────────────────────────────────────────────

@router.post("/collections/{collection_id}/query")
async def query_knowledge(collection_id: int, body: KnowledgeQuery):
    """Hacer una pregunta sobre los documentos de una colección (SSE streaming)."""
    async def event_generator():
        try:
            async for stream_token in rag_service.query_knowledge(
                question=body.question,
                collection_ids=[collection_id],
                top_k=body.top_k,
            ):
                event_type = "thinking" if stream_token["type"] == "thinking" else "token"
                yield {"event": event_type, "data": json.dumps({"token": stream_token["token"]})}
            yield {"event": "done", "data": json.dumps({"status": "complete"})}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())
