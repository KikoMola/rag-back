from datetime import datetime

from pydantic import BaseModel


class TtsVoiceResponse(BaseModel):
    id: str
    name: str
    language: str
    gender: str


class TtsVoicesResponse(BaseModel):
    default_voice: str
    voices: list[TtsVoiceResponse]


class TtsAudioListResponse(BaseModel):
    id: int
    title: str
    source_type: str
    source_name: str | None
    voice: str
    speed: float
    status: str
    progress: int
    char_count: int
    duration_seconds: float | None
    size_bytes: int | None
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TtsAudioResponse(TtsAudioListResponse):
    text_content: str
