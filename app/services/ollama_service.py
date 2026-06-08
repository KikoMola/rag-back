import json
import logging
from collections.abc import AsyncGenerator
from typing import TypedDict

import httpx
from app.config import settings

logger = logging.getLogger(__name__)
OLLAMA_API = settings.ollama_base_url


class StreamToken(TypedDict):
    type: str   # "thinking" | "content"
    token: str


async def chat_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> AsyncGenerator[StreamToken, None]:
    """Streaming de respuesta del LLM. Yield thinking y content tokens por separado."""
    model = model or settings.ollama_chat_model
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        async with client.stream("POST", f"{OLLAMA_API}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})

                thinking = msg.get("thinking", "")
                if thinking:
                    yield StreamToken(type="thinking", token=thinking)

                content = msg.get("content", "")
                if content:
                    yield StreamToken(type="content", token=content)

                if chunk.get("done"):
                    break


async def chat_complete(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> str:
    """Respuesta completa (sin streaming). Útil para tareas internas."""
    model = model or settings.ollama_chat_model
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        response = await client.post(f"{OLLAMA_API}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


async def generate_embeddings(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Generar vectores de embedding para una lista de textos."""
    model = model or settings.ollama_embed_model
    embeddings: list[list[float]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        for i, text in enumerate(texts):
            response = await client.post(
                f"{OLLAMA_API}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            if response.status_code != 200:
                logger.error(
                    "Embedding failed chunk %d/%d (status %d): %s",
                    i + 1, len(texts), response.status_code, response.text,
                )
                response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
    return embeddings


async def check_ollama_status() -> dict:
    """Verificar que Ollama está accesible y listar modelos disponibles."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{OLLAMA_API}/api/tags")
            response.raise_for_status()
            data = response.json()
            model_names = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": model_names}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def generate_title(first_user_message: str) -> str:
    """Genera un título corto (≤6 palabras) para una conversación."""
    messages = [
        {
            "role": "system",
            "content": (
                "Genera un título muy corto (máximo 6 palabras) para una conversación "
                "que empieza con el siguiente mensaje del usuario. "
                "Responde SOLO con el título, sin comillas ni explicaciones adicionales."
            ),
        },
        {"role": "user", "content": first_user_message[:500]},
    ]
    try:
        title = await chat_complete(messages)
        title = title.strip().split("\n")[0][:100]
        return title or "Nueva conversación"
    except Exception:
        return "Nueva conversación"


async def ocr_image(image_b64: str, model: str | None = None) -> str:
    """Ejecuta OCR sobre una imagen en base64 usando un modelo multimodal."""
    model = model or settings.ocr_model
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Extract all text from this image and return it formatted as clean Markdown. "
                    "Preserve the document structure: headings, lists, tables, paragraphs. "
                    "Do not add any commentary, preamble or explanation — output ONLY the Markdown text."
                ),
                "images": [image_b64],
            }
        ],
        "stream": True,
    }
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        async with client.stream("POST", f"{OLLAMA_API}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                parts.append(chunk.get("message", {}).get("content", ""))
                if chunk.get("done"):
                    break
    return "".join(parts)


def _strip_ocr_preamble(text: str) -> str:
    """Elimina líneas de autopresentación que algunos modelos OCR añaden al inicio.

    Si el strip vaciaría el resultado por completo, devuelve el texto original
    para no perder contenido legítimo.
    """
    lines = text.splitlines()
    skip_prefixes = (
        "you are",
        "i am",
        "as an ocr",
        "as a helpful",
        "sure,",
        "sure!",
        "here is",
        "here's",
        "the following",
        "below is",
        "certainly,",
        "certainly!",
        "of course",
    )
    stripped = list(lines)
    while stripped and stripped[0].strip().lower().startswith(skip_prefixes):
        stripped.pop(0)
    # Elimina líneas vacías iniciales resultantes
    while stripped and not stripped[0].strip():
        stripped.pop(0)

    # Salvaguarda: si el strip se lleva todo, devolver el original sin tocar
    result = "\n".join(stripped)
    return result if result.strip() else text.strip()

