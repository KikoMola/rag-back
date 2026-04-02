import json
import logging
from collections.abc import AsyncGenerator

import httpx
from app.config import settings

logger = logging.getLogger(__name__)
OLLAMA_API = settings.ollama_base_url


async def chat_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming de respuesta del LLM. Yield de cada token."""
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
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
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
