import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.document import Document
from app.models.tts import TtsAudio
from app.schemas.tts import TtsAudioListResponse, TtsAudioResponse, TtsVoicesResponse
from app.services import tts_service
from app.services.tts_text import TextExtractionError, extract_for_speech, normalize_for_speech

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".epub", ".html", ".htm", ".txt", ".md"}


@router.get("/voices", response_model=TtsVoicesResponse)
async def list_voices():
    """Voces disponibles en el motor TTS."""
    voices = await tts_service.list_voices()
    return {"default_voice": settings.tts_default_voice, "voices": voices}


@router.post("/audios", response_model=TtsAudioListResponse, status_code=201)
async def create_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    document_id: int | None = Form(None),
    title: str | None = Form(None),
    voice: str | None = Form(None),
    speed: float = Form(1.0, ge=0.5, le=2.0),
    page_start: int | None = Form(None, ge=1),
    page_end: int | None = Form(None, ge=1),
    session: AsyncSession = Depends(get_session),
):
    """Crea un audio a partir de un archivo, de un documento ya indexado o de texto libre.

    Hay que indicar exactamente un origen: `file`, `document_id` o `text`.
    El rango `page_start`/`page_end` (ambos inclusive) solo aplica a PDFs.
    """
    text = text if text and text.strip() else None
    if sum(source is not None for source in (file, document_id, text)) != 1:
        raise HTTPException(
            status_code=422,
            detail="Indica exactamente un origen: file, document_id o text.",
        )

    voice = voice or settings.tts_default_voice
    if voice not in {v.id for v in await tts_service.list_voices()}:
        raise HTTPException(status_code=422, detail=f"Voz no disponible: {voice}")

    try:
        if file is not None:
            source_type, source_name = "file", file.filename or "archivo"
            suffix = Path(source_name).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Formato no soportado. Permitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(file.file, tmp)
            try:
                content = await asyncio.to_thread(extract_for_speech, tmp.name, page_start, page_end)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        elif document_id is not None:
            document = await session.get(Document, document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Documento no encontrado")
            if not Path(document.filepath).exists():
                raise HTTPException(status_code=404, detail="El archivo del documento ya no existe")
            source_type, source_name = "document", document.filename
            content = await asyncio.to_thread(extract_for_speech, document.filepath, page_start, page_end)
        else:
            source_type, source_name = "text", None
            content = normalize_for_speech(text)
    except TextExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("No se pudo extraer texto para TTS")
        raise HTTPException(status_code=422, detail=f"No se pudo leer el documento: {exc}") from exc

    if not content:
        raise HTTPException(
            status_code=422,
            detail="No se ha encontrado texto que leer. Si es un PDF escaneado, pásalo antes por OCR.",
        )
    if len(content) > settings.tts_max_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El texto tiene {len(content):,} caracteres y el máximo es {settings.tts_max_chars:,}. "
                "Usa un rango de páginas para generar el audio por partes."
            ).replace(",", "."),
        )

    if not title or not title.strip():
        if source_name:
            title = Path(source_name).stem
        else:
            first_line = content.split("\n", 1)[0]
            title = first_line if len(first_line) <= 60 else first_line[:60].rsplit(" ", 1)[0] + "…"
            title = title.rstrip(". ") or "Audio"
        if page_start or page_end:
            title += f" (pág. {page_start or 1}-{page_end or 'fin'})"

    audio = TtsAudio(
        title=title.strip()[:500],
        source_type=source_type,
        source_name=source_name,
        voice=voice,
        speed=speed,
        status="pending",
        char_count=len(content),
        text_content=content,
    )
    session.add(audio)
    await session.commit()
    await session.refresh(audio)

    background_tasks.add_task(tts_service.run_job, audio.id)
    return audio


@router.get("/audios", response_model=list[TtsAudioListResponse])
async def list_audios(session: AsyncSession = Depends(get_session)):
    """Lista todos los audios, del más reciente al más antiguo."""
    result = await session.execute(select(TtsAudio).order_by(TtsAudio.created_at.desc(), TtsAudio.id.desc()))
    return result.scalars().all()


@router.get("/audios/{audio_id}", response_model=TtsAudioResponse)
async def get_audio(audio_id: int, session: AsyncSession = Depends(get_session)):
    """Devuelve el audio con su estado, progreso y el texto que se lee."""
    audio = await session.get(TtsAudio, audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    return audio


@router.get("/audios/{audio_id}/file")
async def get_audio_file(
    audio_id: int,
    download: bool = False,
    session: AsyncSession = Depends(get_session),
):
    """Sirve el MP3 (admite peticiones Range). Con `download=true` fuerza la descarga."""
    audio = await session.get(TtsAudio, audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    if audio.status != "done" or not audio.filepath or not Path(audio.filepath).exists():
        raise HTTPException(status_code=409, detail="El audio todavía no está disponible")
    return FileResponse(
        audio.filepath,
        media_type="audio/mpeg",
        filename=f"{audio.title}.mp3",
        content_disposition_type="attachment" if download else "inline",
    )


@router.delete("/audios/{audio_id}", status_code=204)
async def delete_audio(audio_id: int, session: AsyncSession = Depends(get_session)):
    """Elimina el audio y su MP3. Si se está generando, el trabajo se cancela."""
    audio = await session.get(TtsAudio, audio_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio no encontrado")

    if audio.filepath:
        Path(audio.filepath).unlink(missing_ok=True)

    await session.delete(audio)
    await session.commit()
