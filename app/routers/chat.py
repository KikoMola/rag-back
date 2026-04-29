import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.database import get_session
from app.models.conversation import Conversation, Message
from app.schemas.chat import (
    ConversationCreate,
    ConversationForkRequest,
    ConversationForkResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageCreate,
)
from app.services import ollama_service
from app.services import rag_service

router = APIRouter()


# ─── Conversaciones ───────────────────────────────────────────────────

@router.post("/conversations", response_model=ConversationListResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    session: AsyncSession = Depends(get_session),
):
    conversation = Conversation(
        title=body.title,
        mode=body.mode,
        collection_ids_json=json.dumps(body.collection_ids) if body.collection_ids else None,
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.get("/conversations", response_model=list[ConversationListResponse])
async def list_conversations(session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Conversation).order_by(Conversation.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/conversations/search", response_model=list[ConversationListResponse])
async def search_conversations(
    q: str = Query(..., min_length=1),
    session: AsyncSession = Depends(get_session),
):
    """Buscar texto en el historial de conversaciones."""
    result = await session.execute(
        select(Conversation)
        .where(
            Conversation.id.in_(
                select(Message.conversation_id).where(
                    Message.content.ilike(f"%{q}%")
                )
            )
        )
        .order_by(Conversation.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    return ConversationResponse.from_orm_with_collections(conversation)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    await session.delete(conversation)
    await session.commit()


# ─── Mensajes ────────────────────────────────────────────────────────

@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    body: MessageCreate,
    session: AsyncSession = Depends(get_session),
):
    # 1. Verificar que la conversación existe y cargar mensajes previos
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    is_first_message = len(conversation.messages) == 0

    # 2. Guardar mensaje del usuario
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=body.content,
    )
    session.add(user_message)
    await session.commit()

    # 3. Disparar título automático tras el primer mensaje (background)
    if is_first_message:
        asyncio.create_task(
            _auto_title(conversation_id, body.content)
        )

    mode = conversation.mode
    collection_ids = (
        json.loads(conversation.collection_ids_json)
        if conversation.collection_ids_json
        else None
    )

    # 4. Streaming via SSE
    if mode == "rag":
        # Historial previo (sin el mensaje recién añadido)
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation.messages
        ]
        return EventSourceResponse(_rag_generator(
            conversation_id, body.content, collection_ids, history, model=body.model
        ))
    else:
        # Modo general: chat directo con Ollama
        ollama_messages = [
            {
                "role": "system",
                "content": "Eres C.O.R.E., un asistente de inteligencia artificial "
                           "local. Responde de forma útil, precisa y concisa en español.",
            },
        ]
        for msg in conversation.messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})
        ollama_messages.append({"role": "user", "content": body.content})
        return EventSourceResponse(_general_generator(conversation_id, ollama_messages, model=body.model))


async def _general_generator(conversation_id: int, ollama_messages: list[dict], model: str | None = None):
    """SSE generator para modo general."""
    full_response = ""
    try:
        async for stream_token in ollama_service.chat_stream(ollama_messages, model=model):
            if stream_token["type"] == "thinking":
                yield {"event": "thinking", "data": json.dumps({"token": stream_token["token"]})}
            else:
                full_response += stream_token["token"]
                yield {"event": "token", "data": json.dumps({"token": stream_token["token"]})}

        async with _bg_session() as bg_session:
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
            )
            bg_session.add(assistant_message)
            await bg_session.commit()
            yield {
                "event": "done",
                "data": json.dumps({"message_id": assistant_message.id}),
            }
    except Exception as exc:
        yield {"event": "error", "data": json.dumps({"error": str(exc)})}


async def _rag_generator(
    conversation_id: int,
    question: str,
    collection_ids: list[int] | None,
    history: list[dict],
    model: str | None = None,
):
    """SSE generator para modo RAG."""
    full_response = ""
    try:
        async for stream_token in rag_service.query_knowledge(
            question=question,
            collection_ids=collection_ids,
            top_k=settings.rag_top_k,
            history=history,
            model=model,
        ):
            if stream_token["type"] == "thinking":
                yield {"event": "thinking", "data": json.dumps({"token": stream_token["token"]})}
            elif stream_token["type"] == "sources":
                yield {"event": "sources", "data": stream_token["token"]}
            else:
                full_response += stream_token["token"]
                yield {"event": "token", "data": json.dumps({"token": stream_token["token"]})}

        async with _bg_session() as bg_session:
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
            )
            bg_session.add(assistant_message)
            await bg_session.commit()
            yield {
                "event": "done",
                "data": json.dumps({"message_id": assistant_message.id}),
            }
    except Exception as exc:
        yield {"event": "error", "data": json.dumps({"error": str(exc)})}


async def _auto_title(conversation_id: int, first_message: str) -> None:
    """Genera y guarda un título automático para la conversación."""
    title = await ollama_service.generate_title(first_message)
    async with _bg_session() as session:
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation and conversation.title in ("Nueva conversación", ""):
            conversation.title = title
            await session.commit()


# ─── Fork ────────────────────────────────────────────────────────────

@router.post(
    "/conversations/{conversation_id}/fork",
    response_model=ConversationForkResponse,
    status_code=201,
)
async def fork_conversation(
    conversation_id: int,
    body: ConversationForkRequest,
    session: AsyncSession = Depends(get_session),
):
    # 1. Cargar conversación con mensajes
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    # 2. Verificar que el mensaje pertenece a esta conversación
    pivot_message = next(
        (m for m in conversation.messages if m.id == body.message_id), None
    )
    if not pivot_message:
        raise HTTPException(
            status_code=404,
            detail="Mensaje no encontrado en esta conversación",
        )

    # 3. Filtrar mensajes hasta el pivot (inclusive, por created_at)
    messages_to_copy = [
        m for m in conversation.messages
        if m.created_at <= pivot_message.created_at
    ]
    messages_to_copy.sort(key=lambda m: m.created_at)

    # 4. Crear nueva conversación fork
    fork = Conversation(
        title=f"Fork de {conversation.title}",
        mode=conversation.mode,
        collection_ids_json=conversation.collection_ids_json,
        forked_from_id=conversation_id,
        forked_at_message_id=body.message_id,
    )
    session.add(fork)
    await session.flush()  # obtener fork.id antes del commit

    # 5. Copiar mensajes
    for msg in messages_to_copy:
        session.add(Message(
            conversation_id=fork.id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
        ))

    await session.commit()
    await session.refresh(fork)

    return ConversationForkResponse(
        id=fork.id,
        title=fork.title,
        forked_from_id=conversation_id,
        forked_at_message_id=body.message_id,
        message_count=len(messages_to_copy),
        created_at=fork.created_at,
    )


@router.get(
    "/conversations/{conversation_id}/forks",
    response_model=list[ConversationListResponse],
)
async def list_forks(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Lista todas las bifurcaciones creadas a partir de una conversación."""
    result = await session.execute(
        select(Conversation)
        .where(Conversation.forked_from_id == conversation_id)
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


# ─── Exportar ────────────────────────────────────────────────────────

@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: int,
    format: str = Query("md", pattern="^(md|pdf)$"),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    if format == "md":
        lines = [f"# {conversation.title}\n"]
        lines.append(f"*Exportado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")
        lines.append(f"*Modo: {conversation.mode}*\n\n---\n")
        for msg in conversation.messages:
            role_label = "**Usuario**" if msg.role == "user" else "**C.O.R.E.**"
            lines.append(f"\n{role_label}:\n\n{msg.content}\n")
        content = "\n".join(lines)
        filename = f"conversation_{conversation_id}.md"
        return PlainTextResponse(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # PDF con fpdf2
    try:
        from fpdf import FPDF
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Exportación PDF no disponible. Instala fpdf2: pip install fpdf2",
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, conversation.title, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Exportado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Modo: {conversation.mode}", ln=True)
    pdf.ln(4)

    for msg in conversation.messages:
        role_label = "Usuario" if msg.role == "user" else "C.O.R.E."
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, role_label + ":", ln=True)
        pdf.set_font("Helvetica", "", 10)
        # multi_cell gestiona el salto de línea automático
        safe_content = msg.content.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(3)

    pdf_bytes = pdf.output()
    filename = f"conversation_{conversation_id}.pdf"
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─── Helpers ────────────────────────────────────────────────────────

def _bg_session():
    """Sesión independiente para guardar datos dentro del generator SSE."""
    from app.database import async_session
    return async_session()

