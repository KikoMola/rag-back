"""Preparación de texto para lectura en voz alta (TTS).

Extrae el texto de un documento y lo limpia de todo lo que suena mal leído:
cabeceras y pies repetidos, números de página, tablas, guiones de corte de
línea, URLs, marcas de Markdown...
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from app.services.document_processor import extract_text

# Zona superior/inferior de la página donde buscar cabeceras y pies
_MARGIN_RATIO = 0.09
# Un bloque con menos proporción de letras que esta se considera tabla/fórmula
_MIN_LETTER_RATIO = 0.55

_SENTENCE_END = ".!?…:;"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡\"«(0-9])")


class TextExtractionError(ValueError):
    """El documento no contiene texto utilizable para generar audio."""


# ─── Extracción ──────────────────────────────────────────────────────

def extract_for_speech(
    filepath: str,
    page_start: int | None = None,
    page_end: int | None = None,
) -> str:
    """Devuelve el texto del documento listo para leer, un párrafo por línea en blanco."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        paragraphs = _pdf_paragraphs(filepath, page_start, page_end)
    else:
        if page_start is not None or page_end is not None:
            raise TextExtractionError("El rango de páginas solo se admite en documentos PDF.")
        paragraphs = re.split(r"\n\s*\n", extract_text(filepath))
    return normalize_for_speech("\n\n".join(paragraphs))


def _pdf_paragraphs(filepath: str, page_start: int | None, page_end: int | None) -> list[str]:
    doc = fitz.open(filepath)
    try:
        total = doc.page_count
        first = page_start or 1
        last = page_end or total
        if first < 1 or last > total or first > last:
            raise TextExtractionError(
                f"Rango de páginas no válido: el documento tiene {total} páginas."
            )

        # (texto, en_margen) por página. Se recorre el documento entero, no solo el
        # rango pedido, para poder detectar cabeceras repetidas aunque el rango sea corto.
        all_pages: list[list[tuple[str, bool]]] = []
        for page in doc:
            height = page.rect.height
            blocks = []
            for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks"):
                if block_type != 0 or not text.strip():
                    continue
                in_margin = y1 <= height * _MARGIN_RATIO or y0 >= height * (1 - _MARGIN_RATIO)
                blocks.append((text, in_margin))
            all_pages.append(blocks)
    finally:
        doc.close()

    # Cabeceras/pies: bloques en el margen que se repiten (ignorando cifras) en varias páginas
    margin_counts: Counter[str] = Counter()
    for blocks in all_pages:
        for key in {_margin_key(text) for text, in_margin in blocks if in_margin}:
            margin_counts[key] += 1
    repeat_threshold = 3
    pages = all_pages[first - 1 : last]

    paragraphs: list[str] = []
    for blocks in pages:
        for text, in_margin in blocks:
            if in_margin:
                key = _margin_key(text)
                if not key.strip("# ") or margin_counts[key] >= repeat_threshold:
                    continue  # número de página suelto o cabecera/pie repetido
            cleaned = _join_lines(text)
            if not cleaned or _looks_like_table(cleaned):
                continue
            # Un párrafo cortado por salto de página/columna continúa en minúscula
            if paragraphs and paragraphs[-1][-1] not in _SENTENCE_END and cleaned[0].islower():
                paragraphs[-1] = _glue(paragraphs[-1], cleaned)
            else:
                paragraphs.append(cleaned)
    return paragraphs


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.split())).lower()


def _join_lines(block: str) -> str:
    """Une las líneas de un bloque deshaciendo los cortes de palabra con guion."""
    block = block.replace("­", "")
    block = re.sub(r"(\w)-\n\s*([a-záéíóúüñ])", r"\1\2", block)
    return " ".join(block.split())


def _glue(left: str, right: str) -> str:
    if left.endswith("-") and len(left) > 1 and left[-2].isalpha():
        return left[:-1] + right
    return f"{left} {right}"


def _looks_like_table(text: str) -> bool:
    compact = text.replace(" ", "")
    if not compact:
        return True
    letters = sum(ch.isalpha() for ch in compact)
    return letters / len(compact) < _MIN_LETTER_RATIO


# ─── Normalización ───────────────────────────────────────────────────

def normalize_for_speech(text: str) -> str:
    """Limpia el texto de elementos que no deben leerse y normaliza párrafos."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Markdown
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*([-*_]\s*){3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|`)", "", text)
    text = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", text)

    # Elementos no pronunciables
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[\d+(?:\s*[,–-]\s*\d+)*\]", "", text)  # citas numéricas [3], [1-4]
    text = re.sub(r"^\s*[•●▪■◦·–—-]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[•●▪■◦]", " ", text)
    text = re.sub(r"\.{4,}|_{3,}|-{3,}", " ", text)  # líneas de puntos de los índices

    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        paragraph = " ".join(raw.split())
        paragraph = re.sub(r"\s+([.,;:!?)»])", r"\1", paragraph)
        if not any(ch.isalnum() for ch in paragraph):
            continue
        # Títulos y líneas sin puntuación final: forzar una pausa
        if paragraph[-1] not in _SENTENCE_END + ")»\"":
            paragraph += "."
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def split_paragraphs(text: str) -> list[list[str]]:
    """Divide un texto normalizado en párrafos y cada párrafo en frases."""
    result: list[list[str]] = []
    for paragraph in text.split("\n\n"):
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
        if sentences:
            result.append(sentences)
    return result
