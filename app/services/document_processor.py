import re
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from ebooklib import epub

from app.config import settings


# ─── Extracción de texto ─────────────────────────────────────────────

def extract_text(filepath: str) -> str:
    """Extraer texto plano de un documento según su extensión."""
    path = Path(filepath)
    ext = path.suffix.lower()
    extractors = {
        ".pdf": _extract_pdf,
        ".docx": _extract_docx,
        ".epub": _extract_epub,
        ".html": _extract_html,
        ".htm": _extract_html,
        ".txt": _extract_text,
        ".md": _extract_text,
    }
    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(f"Formato no soportado: {ext}")
    return extractor(filepath)


def _extract_pdf(filepath: str) -> str:
    doc = fitz.open(filepath)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)


def _extract_docx(filepath: str) -> str:
    doc = DocxDocument(filepath)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _extract_epub(filepath: str) -> str:
    book = epub.read_epub(filepath)
    text_parts = []
    for item in book.get_items_of_type(9):  # ITEM_DOCUMENT
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            text_parts.append(text.strip())
    return "\n\n".join(text_parts)


def _extract_html(filepath: str) -> str:
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_text(filepath: str) -> str:
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return f.read()


# ─── Chunking ────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Dividir texto en chunks de tamaño aproximado (por palabras) con solapamiento."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not text:
        return []

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def get_format_from_filename(filename: str) -> str:
    """Obtener formato normalizado a partir del nombre de archivo."""
    ext = Path(filename).suffix.lower().lstrip(".")
    format_map = {"htm": "html"}
    return format_map.get(ext, ext)
