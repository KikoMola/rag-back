import chromadb
from app.config import settings

_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Singleton: una sola conexión a ChromaDB para toda la app."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _client


def get_or_create_collection(name: str) -> chromadb.Collection:
    """Obtener (o crear) una colección con métrica coseno."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """Añadir documentos con sus embeddings a una colección."""
    collection = get_or_create_collection(collection_name)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(
    collection_name: str,
    query_embedding: list[float],
    n_results: int = 5,
) -> dict:
    """Buscar los n documentos más similares a un embedding."""
    collection = get_or_create_collection(collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def delete_documents_by_source(collection_name: str, source_id: str) -> None:
    """Eliminar todos los chunks de un documento específico."""
    collection = get_or_create_collection(collection_name)
    collection.delete(where={"source_id": source_id})


def delete_collection(name: str) -> None:
    """Eliminar una colección entera de ChromaDB."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=name)
    except ValueError:
        pass


def collection_exists(name: str) -> bool:
    """Verificar si una colección existe."""
    client = get_chroma_client()
    try:
        client.get_collection(name=name)
        return True
    except Exception:
        return False
