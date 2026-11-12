import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logger = logging.getLogger(__name__)
from app.database import engine
from app.models.conversation import Base
from app.models.document import Collection, Document  # noqa: F401
from app.models.ocr import OcrDocument  # noqa: F401
from app.models.tags import Tag, ConversationTag, DocumentTag  # noqa: F401
from app.models.tts import TtsAudio  # noqa: F401
from sqlalchemy import text


_MIGRATIONS: list[str] = [
    # conversations
    "ALTER TABLE conversations ADD COLUMN mode VARCHAR(20) NOT NULL DEFAULT 'general'",
    "ALTER TABLE conversations ADD COLUMN collection_ids_json TEXT",
    "ALTER TABLE conversations ADD COLUMN forked_from_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL",
    "ALTER TABLE conversations ADD COLUMN forked_at_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL",
    # documents
    "ALTER TABLE documents ADD COLUMN summary TEXT",
    # ocr_documents
    "ALTER TABLE ocr_documents ADD COLUMN thumbnail_b64 TEXT",

    # collections
    "ALTER TABLE collections ADD COLUMN suggested_questions_json TEXT",
]


async def _run_migrations(conn) -> None:
    """Aplica ALTER TABLE para columnas nuevas. Ignora si ya existen."""
    for sql in _MIGRATIONS:
        try:
            await conn.execute(text(sql))
        except Exception:
            pass  # columna ya existe → OperationalError ignorado


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings.ensure_dirs()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_migrations(conn)
        # Los trabajos TTS viven en memoria: los que quedaron a medias no se reanudan
        await conn.execute(
            text(
                "UPDATE tts_audios SET status = 'error', "
                "error_message = 'La generación se interrumpió al reiniciar el servidor' "
                "WHERE status IN ('pending', 'processing')"
            )
        )
    yield
    await engine.dispose()


app = FastAPI(
    title="RAG Local",
    description="Sistema RAG local con Ollama y ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import chat, dashboard, knowledge, ocr, tags, tts  # noqa: E402

app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["OCR"])
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
