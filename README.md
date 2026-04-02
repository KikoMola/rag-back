# RAG Local — Backend

Sistema de **Retrieval-Augmented Generation** (RAG) 100% local con FastAPI, Ollama y ChromaDB. Sin APIs de pago, sin telemetría, tus datos se quedan en tu máquina.

---

## Tabla de contenidos / Table of Contents

- [Español](#español)
- [English](#english)

---

## Español

### Requisitos previos

- **Python 3.11+**
- **Ollama** instalado y corriendo ([descargar](https://ollama.com/download))

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/KikoMola/rag-back.git
cd rag-back

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual (Windows)
.venv\Scripts\Activate.ps1

# 3. Activar entorno virtual (Linux/macOS)
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

### Descargar modelos de Ollama

```bash
ollama pull mxbai-embed-large    # Embeddings (obligatorio)
ollama pull gemma3:27b           # Chat — 24 GB+ VRAM
# Alternativas más ligeras:
# ollama pull gemma3:12b         # 12-16 GB VRAM
# ollama pull gemma3:4b          # 8 GB VRAM
# ollama pull llama3.2:3b        # 4-6 GB VRAM / CPU
```

### Configuración (opcional)

Crea un fichero `.env` en la raíz para sobreescribir valores por defecto:

```env
CORE_OLLAMA_CHAT_MODEL=gemma3:12b
CORE_CHUNK_SIZE=300
CORE_RAG_TOP_K=8
```

### Ejecutar

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La documentación interactiva (Swagger) estará disponible en: **http://localhost:8000/docs**

### Endpoints principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/knowledge/collections` | `GET` | Listar colecciones |
| `/api/knowledge/collections` | `POST` | Crear colección |
| `/api/knowledge/collections/{id}` | `DELETE` | Eliminar colección |
| `/api/knowledge/collections/{id}/documents` | `POST` | Subir documentos |
| `/api/knowledge/collections/{id}/documents` | `GET` | Listar documentos |
| `/api/knowledge/collections/{id}/documents/{id}` | `GET` | Estado de un documento |
| `/api/knowledge/collections/{id}/documents/{id}` | `DELETE` | Eliminar documento |
| `/api/knowledge/collections/{id}/query` | `POST` | Pregunta RAG (SSE) |
| `/api/chat/conversations` | `GET` | Listar conversaciones |
| `/api/chat/conversations` | `POST` | Crear conversación |
| `/api/chat/conversations/{id}` | `GET` | Obtener conversación con mensajes |
| `/api/chat/conversations/{id}` | `DELETE` | Eliminar conversación |
| `/api/chat/conversations/{id}/messages` | `POST` | Enviar mensaje + stream (SSE) |

### Formatos de documento soportados

PDF, DOCX, EPUB, HTML, TXT, Markdown

### Stack tecnológico

| Componente | Función |
|-----------|---------|
| **FastAPI** | API REST async |
| **Ollama** | LLM local + embeddings |
| **ChromaDB** | Base de datos vectorial |
| **SQLite** | Metadata (conversaciones, documentos) |
| **SSE** | Streaming de respuestas |

---

## English

### Prerequisites

- **Python 3.11+**
- **Ollama** installed and running ([download](https://ollama.com/download))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/KikoMola/rag-back.git
cd rag-back

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# 3. Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Download Ollama models

```bash
ollama pull mxbai-embed-large    # Embeddings (required)
ollama pull gemma3:27b           # Chat — 24 GB+ VRAM
# Lighter alternatives:
# ollama pull gemma3:12b         # 12-16 GB VRAM
# ollama pull gemma3:4b          # 8 GB VRAM
# ollama pull llama3.2:3b        # 4-6 GB VRAM / CPU
```

### Configuration (optional)

Create a `.env` file at the project root to override defaults:

```env
CORE_OLLAMA_CHAT_MODEL=gemma3:12b
CORE_CHUNK_SIZE=300
CORE_RAG_TOP_K=8
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs (Swagger) available at: **http://localhost:8000/docs**

### Main endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/knowledge/collections` | `GET` | List collections |
| `/api/knowledge/collections` | `POST` | Create collection |
| `/api/knowledge/collections/{id}` | `DELETE` | Delete collection |
| `/api/knowledge/collections/{id}/documents` | `POST` | Upload documents |
| `/api/knowledge/collections/{id}/documents` | `GET` | List documents |
| `/api/knowledge/collections/{id}/documents/{id}` | `GET` | Document status |
| `/api/knowledge/collections/{id}/documents/{id}` | `DELETE` | Delete document |
| `/api/knowledge/collections/{id}/query` | `POST` | RAG query (SSE) |
| `/api/chat/conversations` | `GET` | List conversations |
| `/api/chat/conversations` | `POST` | Create conversation |
| `/api/chat/conversations/{id}` | `GET` | Get conversation with messages |
| `/api/chat/conversations/{id}` | `DELETE` | Delete conversation |
| `/api/chat/conversations/{id}/messages` | `POST` | Send message + stream (SSE) |

### Supported document formats

PDF, DOCX, EPUB, HTML, TXT, Markdown

### Tech stack

| Component | Role |
|-----------|------|
| **FastAPI** | Async REST API |
| **Ollama** | Local LLM + embeddings |
| **ChromaDB** | Vector database |
| **SQLite** | Metadata (conversations, documents) |
| **SSE** | Response streaming |
