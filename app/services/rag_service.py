import logging
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models.document import Document
from app.services import ollama_service
from app.services.document_processor import chunk_text, extract_text
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


# ─── QUERY ───────────────────────────────────────────────────────────

async def query_knowledge(
    question: str,
    collection_ids: list[int] | None = None,
    top_k: int = 5,
) -> AsyncGenerator[str, None]:
    """
    Pipeline de consulta RAG:
    1. Convertir la pregunta en un embedding
    2. Buscar chunks similares en ChromaDB
    3. Construir el prompt con el contexto recuperado
    4. Hacer streaming de la respuesta del LLM
    """
    # 1. Embedding de la pregunta
    query_embedding = (await ollama_service.generate_embeddings([question]))[0]

    # 2. Determinar en qué colecciones buscar
    if not collection_ids:
        async with async_session() as session:
            from app.models.document import Collection
            result = await session.execute(select(Collection.id))
            collection_ids = [row[0] for row in result.all()]

    # 3. Buscar en cada colección y agregar resultados
    all_chunks: list[dict] = []
    for col_id in collection_ids:
        collection_name = _chroma_collection_name(col_id)
        if not chroma_client.collection_exists(collection_name):
            continue
        results = chroma_client.query_collection(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=top_k,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_text, meta, dist in zip(documents, metadatas, distances):
            all_chunks.append({
                "text": doc_text,
                "filename": meta.get("filename", "desconocido"),
                "distance": dist,
            })

    # 4. Ordenar por relevancia (menor distancia = más parecido)
    all_chunks.sort(key=lambda c: c["distance"])
    top_chunks = all_chunks[:top_k]

    if not top_chunks:
        yield ("No he encontrado información relevante en la biblioteca "
               "del conocimiento para responder a tu pregunta.")
        return

    # 5. Construir el contexto para el prompt
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        context_parts.append(
            f"[Fuente: {chunk['filename']} | Fragmento {i}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 6. Construir mensajes para el LLM
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

    # 7. Stream de la respuesta
    async for token in ollama_service.chat_stream(messages):
        yield token


# ─── LIMPIEZA ────────────────────────────────────────────────────────

def remove_document_from_index(document_id: int, collection_id: int) -> None:
    """Eliminar todos los chunks de un documento de ChromaDB."""
    collection_name = _chroma_collection_name(collection_id)
    chroma_client.delete_documents_by_source(collection_name, str(document_id))
