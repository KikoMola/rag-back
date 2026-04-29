from datetime import datetime
from pydantic import BaseModel, field_validator
import json


class MessageCreate(BaseModel):
    content: str
    model: str | None = None  # None → usa el modelo por defecto del settings


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationCreate(BaseModel):
    title: str = "Nueva conversación"
    mode: str = "general"  # "general" | "rag"
    collection_ids: list[int] | None = None


class ConversationListResponse(BaseModel):
    id: int
    title: str
    mode: str
    forked_from_id: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: int
    title: str
    mode: str
    collection_ids: list[int] | None = None
    forked_from_id: int | None = None
    forked_at_message_id: int | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_with_collections(cls, obj: object) -> "ConversationResponse":
        """Deserializa collection_ids_json → collection_ids."""
        data = {
            "id": obj.id,  # type: ignore[attr-defined]
            "title": obj.title,  # type: ignore[attr-defined]
            "mode": obj.mode,  # type: ignore[attr-defined]
            "collection_ids": json.loads(obj.collection_ids_json)  # type: ignore[attr-defined]
            if obj.collection_ids_json  # type: ignore[attr-defined]
            else None,
            "forked_from_id": obj.forked_from_id,  # type: ignore[attr-defined]
            "forked_at_message_id": obj.forked_at_message_id,  # type: ignore[attr-defined]
            "created_at": obj.created_at,  # type: ignore[attr-defined]
            "updated_at": obj.updated_at,  # type: ignore[attr-defined]
            "messages": obj.messages,  # type: ignore[attr-defined]
        }
        return cls.model_validate(data)


# ─── Fork ────────────────────────────────────────────────────────────

class ConversationForkRequest(BaseModel):
    message_id: int  # ID del mensaje desde el que bifurcar (inclusive)


class ConversationForkResponse(BaseModel):
    id: int
    title: str
    forked_from_id: int
    forked_at_message_id: int
    message_count: int
    created_at: datetime

    model_config = {"from_attributes": True}

