from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.document import Document
from app.models.conversation import Conversation
from app.models.tags import ConversationTag, DocumentTag, Tag
from app.schemas.tags import TagAssignRequest, TagCreate, TagResponse

router = APIRouter()


# ─── CRUD Tags ───────────────────────────────────────────────────────

@router.post("/tags", response_model=TagResponse, status_code=201)
async def create_tag(body: TagCreate, session: AsyncSession = Depends(get_session)):
    existing = await session.execute(select(Tag).where(Tag.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Ya existe una etiqueta con ese nombre")
    tag = Tag(name=body.name, color=body.color)
    session.add(tag)
    await session.commit()
    await session.refresh(tag)
    return tag


@router.get("/tags", response_model=list[TagResponse])
async def list_tags(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Tag).order_by(Tag.name))
    return result.scalars().all()


@router.delete("/tags/{tag_id}", status_code=204)
async def delete_tag(tag_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Tag).where(Tag.id == tag_id))
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Etiqueta no encontrada")
    await session.delete(tag)
    await session.commit()


# ─── Tags en conversaciones ───────────────────────────────────────────

@router.get("/conversations/{conversation_id}/tags", response_model=list[TagResponse])
async def get_conversation_tags(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Tag)
        .join(ConversationTag, ConversationTag.tag_id == Tag.id)
        .where(ConversationTag.conversation_id == conversation_id)
        .order_by(Tag.name)
    )
    return result.scalars().all()


@router.post("/conversations/{conversation_id}/tags", response_model=TagResponse, status_code=201)
async def assign_tag_to_conversation(
    conversation_id: int,
    body: TagAssignRequest,
    session: AsyncSession = Depends(get_session),
):
    conv = await session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    tag = await session.get(Tag, body.tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Etiqueta no encontrada")

    existing = await session.execute(
        select(ConversationTag).where(
            ConversationTag.conversation_id == conversation_id,
            ConversationTag.tag_id == body.tag_id,
        )
    )
    if not existing.scalar_one_or_none():
        session.add(ConversationTag(conversation_id=conversation_id, tag_id=body.tag_id))
        await session.commit()
    return tag


@router.delete("/conversations/{conversation_id}/tags/{tag_id}", status_code=204)
async def remove_tag_from_conversation(
    conversation_id: int,
    tag_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(ConversationTag).where(
            ConversationTag.conversation_id == conversation_id,
            ConversationTag.tag_id == tag_id,
        )
    )
    ct = result.scalar_one_or_none()
    if not ct:
        raise HTTPException(status_code=404, detail="Asignación no encontrada")
    await session.delete(ct)
    await session.commit()


# ─── Tags en documentos ───────────────────────────────────────────────

@router.get("/documents/{document_id}/tags", response_model=list[TagResponse])
async def get_document_tags(
    document_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Tag)
        .join(DocumentTag, DocumentTag.tag_id == Tag.id)
        .where(DocumentTag.document_id == document_id)
        .order_by(Tag.name)
    )
    return result.scalars().all()


@router.post("/documents/{document_id}/tags", response_model=TagResponse, status_code=201)
async def assign_tag_to_document(
    document_id: int,
    body: TagAssignRequest,
    session: AsyncSession = Depends(get_session),
):
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    tag = await session.get(Tag, body.tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Etiqueta no encontrada")

    existing = await session.execute(
        select(DocumentTag).where(
            DocumentTag.document_id == document_id,
            DocumentTag.tag_id == body.tag_id,
        )
    )
    if not existing.scalar_one_or_none():
        session.add(DocumentTag(document_id=document_id, tag_id=body.tag_id))
        await session.commit()
    return tag


@router.delete("/documents/{document_id}/tags/{tag_id}", status_code=204)
async def remove_tag_from_document(
    document_id: int,
    tag_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(DocumentTag).where(
            DocumentTag.document_id == document_id,
            DocumentTag.tag_id == tag_id,
        )
    )
    dt = result.scalar_one_or_none()
    if not dt:
        raise HTTPException(status_code=404, detail="Asignación no encontrada")
    await session.delete(dt)
    await session.commit()
