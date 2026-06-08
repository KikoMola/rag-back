"""Servicio de OCR: procesa imágenes y PDFs con deepseek-ocr vía Ollama."""
import base64
import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.services.ollama_service import ocr_image

logger = logging.getLogger(__name__)

# Formatos de imagen directamente soportados como input multimodal
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def _image_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _pdf_page_to_png_b64(page: fitz.Page, dpi: int = 200) -> str:
    """Renderiza una página PDF a PNG y devuelve base64."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return _image_bytes_to_b64(pix.tobytes("png"))



async def run_ocr(filepath: Path, fmt: str, model: str | None = None) -> tuple[str, int]:
    """Ejecuta OCR sobre el archivo y devuelve (markdown_content, page_count).

    - Imágenes: una sola llamada OCR.
    - PDFs: renderiza cada página a imagen y concatena resultados.
    """
    ext = Path(filepath).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        raw = Path(filepath).read_bytes()
        b64 = _image_bytes_to_b64(raw)
        markdown = await ocr_image(b64, model=model)
        return markdown.strip(), 1

    if ext in PDF_EXTENSIONS:
        doc = fitz.open(str(filepath))
        page_count = doc.page_count
        parts: list[str] = []
        for i, page in enumerate(doc):
            b64 = _pdf_page_to_png_b64(page)
            page_md = await ocr_image(b64, model=model)
            header = f"## Página {i + 1}\n\n" if page_count > 1 else ""
            parts.append(header + page_md.strip())
        doc.close()
        separator = "\n\n---\n\n" if page_count > 1 else ""
        return separator.join(parts), page_count

    raise ValueError(f"Formato no soportado para OCR: {ext}")
