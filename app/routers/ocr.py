import base64
import logging
import shutil
from pathlib import Path

import fitz
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.ocr import OcrDocument
from app.schemas.ocr import OcrDocumentListResponse, OcrDocumentResponse
from app.services.ocr_service import ALLOWED_EXTENSIONS, run_ocr

logger = logging.getLogger(__name__)

router = APIRouter()

_THUMB_MAX_PX = 400


def _make_thumbnail_b64(filepath: Path) -> str | None:
    """Genera una miniatura JPEG (max 400px) y la devuelve como base64 data URL."""
    try:
        doc = fitz.open(str(filepath))
        page = doc[0]
        # Escalar para que el lado más largo sea _THUMB_MAX_PX
        w, h = page.rect.width, page.rect.height
        scale = _THUMB_MAX_PX / max(w, h)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
        jpeg_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        logger.warning("Thumbnail generation failed for %s", filepath, exc_info=True)
        return None


async def _process_ocr(document_id: int) -> None:
    """Tarea en background: ejecuta OCR y actualiza el registro en DB."""
    from app.database import async_session

    async with async_session() as session:
        doc = await session.get(OcrDocument, document_id)
        if not doc:
            return
        try:
            doc.status = "processing"
            await session.commit()

            markdown, page_count = await run_ocr(Path(doc.filepath), doc.format)

            doc.markdown_content = markdown
            doc.page_count = page_count
            doc.status = "done"
        except Exception as exc:
            logger.exception("OCR failed for document %d", document_id)
            doc.status = "error"
            doc.error_message = str(exc)
        await session.commit()


@router.post("/documents", response_model=OcrDocumentResponse, status_code=201)
async def upload_ocr_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Sube un archivo (imagen o PDF) y lanza el OCR en background."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado. Permitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    settings.ocr_uploads_dir.mkdir(parents=True, exist_ok=True)

    dest = settings.ocr_uploads_dir / file.filename
    counter = 1
    stem = Path(file.filename).stem
    while dest.exists():
        dest = settings.ocr_uploads_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size_bytes = dest.stat().st_size
    thumbnail_b64 = _make_thumbnail_b64(dest)

    doc = OcrDocument(
        filename=dest.name,
        filepath=str(dest),
        format=suffix.lstrip("."),
        size_bytes=size_bytes,
        status="pending",
        thumbnail_b64=thumbnail_b64,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    background_tasks.add_task(_process_ocr, doc.id)

    return doc


@router.get("/documents", response_model=list[OcrDocumentListResponse])
async def list_ocr_documents(session: AsyncSession = Depends(get_session)):
    """Lista todos los documentos OCR ordenados del más reciente al más antiguo."""
    result = await session.execute(
        select(OcrDocument).order_by(OcrDocument.created_at.desc())
    )
    return result.scalars().all()


@router.get("/documents/{document_id}", response_model=OcrDocumentResponse)
async def get_ocr_document(
    document_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Devuelve el documento OCR con su contenido Markdown (cuando status=done)."""
    doc = await session.get(OcrDocument, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return doc


@router.delete("/documents/{document_id}", status_code=204)
async def delete_ocr_document(
    document_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Elimina el documento OCR y el archivo original del disco."""
    doc = await session.get(OcrDocument, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    filepath = Path(doc.filepath)
    if filepath.exists():
        filepath.unlink()

    await session.delete(doc)
    await session.commit()
