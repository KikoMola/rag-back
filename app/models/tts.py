from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.conversation import Base


class TtsAudio(Base):
    __tablename__ = "tts_audios"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    # Origen del texto: "file" (archivo subido), "document" (documento de una colección) o "text"
    source_type: Mapped[str] = mapped_column(String(20))
    source_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    voice: Mapped[str] = mapped_column(String(50))
    speed: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    filepath: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Texto ya limpio que se lee en voz alta
    text_content: Mapped[str] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
