import logging
import re
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session
from app.models.document import Document
from app.services import ollama_service
from app.services.document_processor import chunk_text, extract_text
from app.services.ollama_service import StreamToken
from app.vectorstore import chroma_client

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "knowledge_col_"

RAG_SYSTEM_PROMPT = (
    "Eres C.O.R.E., un asistente de inteligencia artificial especializado en "
    "análisis de documentos. "
    "Responde ÚNICAMENTE basándote en la información del contexto proporcionado. "
    "Si la información no está en el contexto, indícalo claramente sin inventar nada. "
    "Responde en español con rigor técnico y académico: explica los conceptos en "
    "profundidad, menciona detalles relevantes, define términos técnicos cuando sea "
    "necesario, y estructura la respuesta de forma clara con párrafos bien "
    "desarrollados. No des respuestas superficiales ni telegráficas."
)


def _chroma_collection_name(collection_id: int) -> str:
    """Nombre de la colección en ChromaDB. Prefijo para evitar colisiones."""
    return f"{COLLECTION_PREFIX}{collection_id}"


# ─── INGESTA ─────────────────────────────────────────────────────────

async def ingest_document(document_id: int) -> None:
    """
    Tarea en background: procesar un documento recién subido.

    Pipeline:
    1. Leer registro de BD
    2. Extraer texto según formato
    3. Dividir en chunks con solapamiento
    4. Generar embeddings via Ollama
    5. Almacenar en ChromaDB con metadata
    6. Actualizar estado en BD
    """
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            return

        try:
            doc.status = "processing"
            await session.commit()

            # 1. Extraer texto
            text = extract_text(doc.filepath)
            if not text.strip():
                doc.status = "error"
                doc.error_message = "No se pudo extraer texto del documento"
                await session.commit()
                return

            # 2. Chunk
            chunks = chunk_text(text)
            if not chunks:
                doc.status = "error"
                doc.error_message = "El documento no generó fragmentos de texto"
                await session.commit()
                return

            # 3. Generar embeddings
            embeddings = await ollama_service.generate_embeddings(chunks)

            # 4. Almacenar en ChromaDB
            collection_name = _chroma_collection_name(doc.collection_id)
            ids = [f"doc_{doc.id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source_id": str(doc.id),
                    "filename": doc.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]

            chroma_client.add_documents(
                collection_name=collection_name,
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            # 5. Actualizar estado
            doc.chunk_count = len(chunks)
            doc.status = "indexed"
            doc.error_message = None
            await session.commit()

            logger.info(
                "Document %s indexed: %d chunks in collection %s",
                doc.filename, len(chunks), collection_name,
            )

        except Exception as exc:
            doc.status = "error"
            doc.error_message = str(exc)[:500]
            await session.commit()
            logger.exception("Error indexing document %s", doc.filename)


# ─── HELPERS ─────────────────────────────────────────────────────────

# Stop words comunes en español (preguntas, artículos, preposiciones, verbos auxiliares)
_STOP_WORDS = frozenset({
    "me", "puedes", "puede", "podrias", "podrías", "explicar", "explicame",
    "explícame", "hacer", "hazme", "dame", "dar", "decir", "dime",
    "de", "la", "el", "las", "los", "un", "una", "unos", "unas",
    "en", "con", "por", "para", "del", "al", "que", "qué",
    "como", "cómo", "es", "son", "fue", "ser", "estar",
    "muy", "más", "mas", "menos", "ya", "si", "sí", "no",
    "y", "o", "a", "e", "u", "pero", "ni", "sin",
    "su", "sus", "mi", "mis", "tu", "tus",
    "este", "esta", "estos", "estas", "ese", "esa",
    "lo", "le", "les", "se", "nos", "os",
    "hay", "ha", "han", "he", "has",
    "sobre", "entre", "desde", "hasta", "tras",
    "manera", "forma", "detalladamente", "detallada", "detalle",
    "breve", "brevemente", "resumida", "resumen", "resumidamente",
    "favor", "gracias", "hola", "bien",
})


def _extract_key_phrases(question: str) -> list[str]:
    """
    Extrae frases clave de la pregunta del usuario para búsqueda keyword.

    Estrategia:
    1. Texto entrecomillado (prioridad máxima, el usuario lo destaca explícitamente)
    2. Bigramas de palabras significativas (elimina stop words)
    3. Palabras individuales largas (>5 chars) como fallback
    """
    phrases: list[str] = []

    # 1. Entrecomillado explícito: "...", '...', «...», "..."
    quoted = re.findall(r'["\u201c\u00ab\'](.*?)["\u201d\u00bb\']', question)
    phrases.extend(q.strip() for q in quoted if len(q.strip()) > 2)

    # 2. Bigramas de palabras significativas
    words = re.findall(r"\w+", question.lower())
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    for i in range(len(content_words) - 1):
        bigram = f"{content_words[i]} {content_words[i + 1]}"
        if bigram.lower() not in [p.lower() for p in phrases]:
            phrases.append(bigram)

    # 3. Palabras individuales largas como último recurso
    for w in content_words:
        if len(w) > 5 and w not in [p.lower() for p in phrases]:
            phrases.append(w)

    return phrases


async def _detect_mentioned_filename(
    question: str,
    collection_ids: list[int],
) -> str | None:
    """
    Revisa si la pregunta del usuario menciona literalmente alguno de los
    nombres de archivo indexados en las colecciones indicadas.
    Devuelve el filename tal cual está en la BD (y en los metadatos de ChromaDB)
    o None si no detecta ninguno.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Document.filename).where(
                Document.collection_id.in_(collection_ids),
                Document.status == "indexed",
            )
        )
        filenames = [row[0] for row in result.all()]

    # Ordenar de más largo a más corto para evitar matches parciales
    filenames.sort(key=len, reverse=True)

    question_lower = question.lower()
    for fname in filenames:
        # Buscar el nombre con o sin extensión
        name_lower = fname.lower()
        name_no_ext = name_lower.rsplit(".", 1)[0] if "." in name_lower else name_lower
        if name_lower in question_lower or name_no_ext in question_lower:
            return fname

    return None


# ─── QUERY ───────────────────────────────────────────────────────────

async def query_knowledge(
    question: str,
    collection_ids: list[int] | None = None,
    top_k: int = 5,
) -> AsyncGenerator[StreamToken, None]:
    """
    Pipeline de consulta RAG con búsqueda híbrida (semántica + keyword):
    1. Convertir la pregunta en un embedding
    2. Determinar colecciones objetivo
    3. Detectar filtros (filename mencionado, frases clave)
    4. Búsqueda semántica (embedding similarity)
    5. Búsqueda keyword ($contains en texto de chunks)
    6. Fusionar y deduplicar resultados
    7. Construir prompt con contexto y hacer streaming
    """
    # 1. Embedding de la pregunta
    query_embedding = (await ollama_service.generate_embeddings([question]))[0]

    # 2. Determinar en qué colecciones buscar
    if not collection_ids:
        async with async_session() as session:
            from app.models.document import Collection
            result = await session.execute(select(Collection.id))
            collection_ids = [row[0] for row in result.all()]

    # 3. Detectar filtros
    mentioned_filename = await _detect_mentioned_filename(question, collection_ids)
    where_filter: dict | None = None
    if mentioned_filename:
        where_filter = {"filename": mentioned_filename}

    key_phrases = _extract_key_phrases(question)

    # 4. Búsqueda semántica (over-fetch ×3)
    fetch_k = top_k * 3
    seen_texts: set[str] = set()
    all_chunks: list[dict] = []

    for col_id in collection_ids:
        collection_name = _chroma_collection_name(col_id)
        if not chroma_client.collection_exists(collection_name):
            continue
        results = chroma_client.query_collection(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=fetch_k,
            where=where_filter,
        )
        for doc_text, meta, dist in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            text_key = doc_text[:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_chunks.append({
                    "text": doc_text,
                    "filename": meta.get("filename", "desconocido"),
                    "distance": dist,
                    "keyword_match": False,
                })

    # 5. Búsqueda keyword: para cada frase clave, buscar chunks que
    #    contengan ese texto literal (complementa la búsqueda semántica)
    for phrase in key_phrases:
        for col_id in collection_ids:
            collection_name = _chroma_collection_name(col_id)
            if not chroma_client.collection_exists(collection_name):
                continue
            try:
                kw_results = chroma_client.query_collection(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where=where_filter,
                    where_document={"$contains": phrase},
                )
                for doc_text, meta, dist in zip(
                    kw_results.get("documents", [[]])[0],
                    kw_results.get("metadatas", [[]])[0],
                    kw_results.get("distances", [[]])[0],
                ):
                    text_key = doc_text[:200]
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_chunks.append({
                            "text": doc_text,
                            "filename": meta.get("filename", "desconocido"),
                            "distance": dist,
                            "keyword_match": True,
                        })
            except Exception:
                # $contains puede fallar si la frase no existe; ignorar
                continue

    # 6. Ordenar: priorizar keyword matches, luego por distancia
    max_distance = settings.rag_max_distance
    all_chunks = [c for c in all_chunks if c["distance"] <= max_distance]
    all_chunks.sort(key=lambda c: (not c["keyword_match"], c["distance"]))
    top_chunks = all_chunks[:top_k]

    logger.info(
        "RAG hybrid retrieval: query=%r | semantic=%d | total=%d | top_k=%d | sources=%s",
        question[:80],
        sum(1 for c in all_chunks if not c["keyword_match"]),
        len(all_chunks),
        len(top_chunks),
        {c["filename"] for c in top_chunks},
    )

    if not top_chunks:
        yield StreamToken(type="content", token=(
            "No he encontrado información relevante en la biblioteca "
            "del conocimiento para responder a tu pregunta."
        ))
        return

    # 7. Construir el contexto para el prompt
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        context_parts.append(
            f"[Fuente: {chunk['filename']} | Fragmento {i}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 8. Construir mensajes para el LLM
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONTEXTO:\n{context}\n\n"
                f"---\n\n"
                f"PREGUNTA: {question}"
            ),
        },
    ]

    # 9. Stream de la respuesta
    async for stream_token in ollama_service.chat_stream(messages):
        yield stream_token


# ─── LIMPIEZA ────────────────────────────────────────────────────────

def remove_document_from_index(document_id: int, collection_id: int) -> None:
    """Eliminar todos los chunks de un documento de ChromaDB."""
    collection_name = _chroma_collection_name(collection_id)
    chroma_client.delete_documents_by_source(collection_name, str(document_id))
