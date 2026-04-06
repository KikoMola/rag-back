from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    db_path: Path = data_dir / "core.db"
    chroma_dir: Path = data_dir / "chroma"
    uploads_dir: Path = data_dir / "uploads"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:26b"
    ollama_embed_model: str = "mxbai-embed-large"

    # --- RAG ---
    chunk_size: int = 200
    chunk_overlap: int = 20
    rag_top_k: int = 10
    rag_max_distance: float = 1.2

    # --- CORS ---
    cors_origins: list[str] = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ]

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"

    def ensure_dirs(self) -> None:
        """Crear directorios de datos si no existen."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "env_prefix": "CORE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
