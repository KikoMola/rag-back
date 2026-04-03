import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.database import get_session
from app.models.conversation import Conversation, Message
from app.schemas.chat import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    MessageCreate,
)
from app.services import ollama_service

router = APIRouter()


@router.post("/conversations", response_model=ConversationListResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    session: AsyncSession = Depends(get_session),
):
    conversation = Conversation(title=body.title)
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
    return conversation


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

    # 2. Guardar mensaje del usuario
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=body.content,
    )
    session.add(user_message)
    await session.commit()

    # 3. Construir historial para Ollama
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

    # 4. Streaming via SSE
    async def event_generator():
        full_response = ""
        full_thinking = ""
        try:
            async for stream_token in ollama_service.chat_stream(ollama_messages):
                if stream_token["type"] == "thinking":
                    full_thinking += stream_token["token"]
                    yield {"event": "thinking", "data": json.dumps({"token": stream_token["token"]})}
                else:
                    full_response += stream_token["token"]
                    yield {"event": "token", "data": json.dumps({"token": stream_token["token"]})}

            # 5. Guardar respuesta del asistente
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

    return EventSourceResponse(event_generator())


def _bg_session():
    """Sesión independiente para guardar datos dentro del generator SSE."""
    from app.database import async_session
    return async_session()
