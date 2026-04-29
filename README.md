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
- **24 GB+ VRAM** recomendados para el modelo por defecto (Gemma4:26b)

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
ollama pull gemma4:26b           # Chat con reasoning — 24 GB+ VRAM (recomendado)
ollama pull qwen3.6:27b          # Modelo coding (código, programación) — 24 GB+ VRAM
# Alternativas más ligeras:
# ollama pull gemma3:12b         # 12-16 GB VRAM
# ollama pull gemma3:4b          # 8 GB VRAM
# ollama pull llama3.2:3b        # 4-6 GB VRAM / CPU
```

> **Nota**: Gemma4:26b incluye capacidades de **reasoning** (cadena de pensamiento), lo que mejora significativamente la calidad de las respuestas RAG. El streaming diferencia entre tokens de `thinking` y `content`.
>
> **Modelo coding**: Para preguntas sobre código o programación usa `qwen3.6:27b`, pasándolo como campo `model` en el body del mensaje.

### Configuración (opcional)

Crea un fichero `.env` en la raíz para sobreescribir valores por defecto:

```env
CORE_OLLAMA_CHAT_MODEL=gemma4:26b    # Modelo de chat por defecto
CORE_OLLAMA_EMBED_MODEL=mxbai-embed-large
CORE_CHUNK_SIZE=200                  # Palabras por chunk (default: 200)
CORE_CHUNK_OVERLAP=20                # Solapamiento entre chunks (default: 20)
CORE_RAG_TOP_K=10                    # Chunks de contexto enviados al LLM (default: 10)
CORE_RAG_MAX_DISTANCE=1.2           # Umbral máximo de distancia coseno (default: 1.2)
```

> El modelo se puede sobreescribir **por mensaje** enviando el campo `model` en el body. Ejemplo: `{ "content": "Escribe una función en Python...", "model": "qwen3.6:27b" }`.

### Ejecutar

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La documentación interactiva (Swagger) estará disponible en: **http://localhost:8000/docs**

### Pipeline RAG — Búsqueda híbrida

El sistema utiliza un pipeline de búsqueda **híbrida** (semántica + keyword) para maximizar la relevancia del contexto:

1. **Embedding de la pregunta** — Genera el vector de la query con `mxbai-embed-large`.
2. **Detección de filtros** — Detecta si el usuario menciona un archivo concreto y aplica filtro por metadata.
3. **Extracción de frases clave** — Extrae términos significativos de la pregunta (entrecomillados, bigramas y palabras largas) eliminando stop words.
4. **Búsqueda semántica** — Recupera `top_k × 3` candidatos por similitud de embeddings (over-fetch).
5. **Búsqueda keyword** — Para cada frase clave, busca chunks que contengan ese texto literal (`$contains`).
6. **Fusión y ranking** — Deduplica, filtra por umbral de distancia, y prioriza los keyword matches sobre los semánticos.
7. **Generación** — Construye el prompt con el contexto y hace streaming de la respuesta del LLM.

### Endpoints principales

**Dashboard**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/dashboard` | `GET` | Dashboard con estadísticas generales |
| `/api/dashboard/search` | `GET` | Buscar colecciones por nombre/descripción |

**Knowledge (colecciones y documentos)**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/knowledge/collections` | `GET` | Listar colecciones |
| `/api/knowledge/collections` | `POST` | Crear colección |
| `/api/knowledge/collections/{id}` | `DELETE` | Eliminar colección |
| `/api/knowledge/collections/{id}/documents` | `POST` | Subir documentos (múltiples) |
| `/api/knowledge/collections/{id}/documents` | `GET` | Listar documentos |
| `/api/knowledge/collections/{id}/documents/{id}` | `GET` | Estado de un documento (incluye `summary`) |
| `/api/knowledge/collections/{id}/documents/{id}` | `DELETE` | Eliminar documento |
| `/api/knowledge/collections/{id}/query` | `POST` | Pregunta RAG (SSE streaming) |
| `/api/knowledge/collections/{id}/suggested-questions` | `GET` | Sugerencias de preguntas (cacheadas) |
| `/api/knowledge/documents/compare` | `POST` | Comparar dos documentos (SSE streaming) |

**Chat (conversaciones)**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/chat/conversations` | `GET` | Listar conversaciones |
| `/api/chat/conversations` | `POST` | Crear conversación (`mode`: `general`\|`rag`) |
| `/api/chat/conversations/search?q=` | `GET` | Buscar texto en historial |
| `/api/chat/conversations/{id}` | `GET` | Obtener conversación con mensajes |
| `/api/chat/conversations/{id}` | `DELETE` | Eliminar conversación |
| `/api/chat/conversations/{id}/messages` | `POST` | Enviar mensaje + stream (SSE). Acepta `model` opcional |
| `/api/chat/conversations/{id}/fork` | `POST` | Bifurcar conversación hasta un mensaje |
| `/api/chat/conversations/{id}/forks` | `GET` | Listar bifurcaciones de una conversación |
| `/api/chat/conversations/{id}/export?format=md\|pdf` | `GET` | Exportar conversación (Markdown o PDF) |

**Tags (etiquetas)**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/tags/tags` | `GET` | Listar todas las etiquetas |
| `/api/tags/tags` | `POST` | Crear etiqueta (`name`, `color`) |
| `/api/tags/tags/{id}` | `DELETE` | Eliminar etiqueta |
| `/api/tags/conversations/{id}/tags` | `GET` | Tags de una conversación |
| `/api/tags/conversations/{id}/tags` | `POST` | Asignar tag a conversación |
| `/api/tags/conversations/{id}/tags/{tag_id}` | `DELETE` | Desasignar tag de conversación |
| `/api/tags/documents/{id}/tags` | `GET` | Tags de un documento |
| `/api/tags/documents/{id}/tags` | `POST` | Asignar tag a documento |
| `/api/tags/documents/{id}/tags/{tag_id}` | `DELETE` | Desasignar tag de documento |

### Eventos SSE (streaming)

Las respuestas de chat y RAG se envían mediante Server-Sent Events. Los eventos disponibles son:

| Evento | Descripción | Datos |
|--------|-------------|-------|
| `thinking` | Token de razonamiento interno (cadena de pensamiento) | `{ "token": "..." }` |
| `sources` | Fuentes RAG utilizadas — llega **antes** de los tokens (solo modo RAG) | `[{ "filename", "chunk_index", "excerpt" }]` |
| `token` | Token de respuesta visible para el usuario | `{ "token": "..." }` |
| `done` | Fin de la respuesta | `{ "message_id": 42 }` |
| `error` | Error durante la generación | `{ "error": "..." }` |

### Formatos de documento soportados

PDF, DOCX, EPUB, HTML, TXT, Markdown

### Nuevas funcionalidades

- **Modo dual de chat** — Conversaciones en modo `general` (Gemma4:26b / Qwen3.6:27b) o `rag` (con colecciones seleccionadas)
- **Modelo por mensaje** — Indica `model: "qwen3.6:27b"` en el body para usar el modelo coding en cualquier mensaje
- **Bifurcaciones** — Ramifica cualquier conversación a partir de un mensaje concreto
- **Títulos automáticos** — El título se genera automáticamente tras el primer mensaje
- **Exportación** — Descarga conversaciones en Markdown o PDF
- **Búsqueda** — Busca texto en el historial de todas las conversaciones
- **Citas con referencias** — El evento `sources` devuelve los chunks RAG usados antes de la respuesta
- **Sugerencias de preguntas** — Se generan y cachean automáticamente por colección
- **Resumen de documentos** — Generado automáticamente tras la indexación
- **Comparación de documentos** — Contrasta dos documentos vía SSE streaming
- **Etiquetas** — Sistema de tags para organizar conversaciones y documentos

### Stack tecnológico

| Componente | Función |
|-----------|-------|
| **FastAPI** | API REST async |
| **Ollama** | LLM local (Gemma4:26b · Qwen3.6:27b) + embeddings (mxbai-embed-large) |
| **ChromaDB** | Base de datos vectorial (distancia coseno) |
| **SQLite + SQLAlchemy** | Metadata async (conversaciones, documentos, colecciones, tags) |
| **SSE (sse-starlette)** | Streaming de respuestas con thinking/sources/content |
| **fpdf2** | Exportación de conversaciones a PDF |
| **PyMuPDF** | Extracción de texto de PDF |
| **python-docx** | Extracción de texto de DOCX |
| **ebooklib** | Extracción de texto de EPUB |

---

## English

### Prerequisites

- **Python 3.11+**
- **Ollama** installed and running ([download](https://ollama.com/download))
- **24 GB+ VRAM** recommended for the default model (Gemma4:26b)

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
ollama pull gemma4:26b           # Chat with reasoning — 24 GB+ VRAM (recommended)
ollama pull qwen3.6:27b          # Coding model (code & programming) — 24 GB+ VRAM
# Lighter alternatives:
# ollama pull gemma3:12b         # 12-16 GB VRAM
# ollama pull gemma3:4b          # 8 GB VRAM
# ollama pull llama3.2:3b        # 4-6 GB VRAM / CPU
```

> **Note**: Gemma4:26b includes **reasoning** capabilities (chain of thought), which significantly improves RAG response quality. Streaming differentiates between `thinking` and `content` tokens.
>
> **Coding model**: For code-related questions use `qwen3.6:27b` by passing it as the `model` field in the message body.

### Configuration (optional)

Create a `.env` file at the project root to override defaults:

```env
CORE_OLLAMA_CHAT_MODEL=gemma4:26b    # Default chat model
CORE_OLLAMA_EMBED_MODEL=mxbai-embed-large
CORE_CHUNK_SIZE=200                  # Words per chunk (default: 200)
CORE_CHUNK_OVERLAP=20                # Overlap between chunks (default: 20)
CORE_RAG_TOP_K=10                    # Context chunks sent to the LLM (default: 10)
CORE_RAG_MAX_DISTANCE=1.2           # Max cosine distance threshold (default: 1.2)
```

> The model can be overridden **per message** by sending the `model` field in the body. Example: `{ "content": "Write a Python function...", "model": "qwen3.6:27b" }`.

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs (Swagger) available at: **http://localhost:8000/docs**

### RAG Pipeline — Hybrid search

The system uses a **hybrid** search pipeline (semantic + keyword) to maximize context relevance:

1. **Query embedding** — Generates the query vector with `mxbai-embed-large`.
2. **Filter detection** — Detects if the user mentions a specific file and applies a metadata filter.
3. **Key phrase extraction** — Extracts significant terms from the query (quoted text, bigrams and long words) after removing stop words.
4. **Semantic search** — Retrieves `top_k × 3` candidates by embedding similarity (over-fetch).
5. **Keyword search** — For each key phrase, finds chunks containing that literal text (`$contains`).
6. **Merge and ranking** — Deduplicates, filters by distance threshold, and prioritizes keyword matches over semantic ones.
7. **Generation** — Builds the prompt with context and streams the LLM response.

### Main endpoints

**Dashboard**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard` | `GET` | Dashboard with general statistics |
| `/api/dashboard/search` | `GET` | Search collections by name/description |

**Knowledge (collections & documents)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/knowledge/collections` | `GET` | List collections |
| `/api/knowledge/collections` | `POST` | Create collection |
| `/api/knowledge/collections/{id}` | `DELETE` | Delete collection |
| `/api/knowledge/collections/{id}/documents` | `POST` | Upload documents (multiple) |
| `/api/knowledge/collections/{id}/documents` | `GET` | List documents |
| `/api/knowledge/collections/{id}/documents/{id}` | `GET` | Document status (includes `summary`) |
| `/api/knowledge/collections/{id}/documents/{id}` | `DELETE` | Delete document |
| `/api/knowledge/collections/{id}/query` | `POST` | RAG query (SSE streaming) |
| `/api/knowledge/collections/{id}/suggested-questions` | `GET` | Suggested questions (cached) |
| `/api/knowledge/documents/compare` | `POST` | Compare two documents (SSE streaming) |

**Chat (conversations)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat/conversations` | `GET` | List conversations |
| `/api/chat/conversations` | `POST` | Create conversation (`mode`: `general`\|`rag`) |
| `/api/chat/conversations/search?q=` | `GET` | Search text across conversation history |
| `/api/chat/conversations/{id}` | `GET` | Get conversation with messages |
| `/api/chat/conversations/{id}` | `DELETE` | Delete conversation |
| `/api/chat/conversations/{id}/messages` | `POST` | Send message + stream (SSE). Optional `model` field |
| `/api/chat/conversations/{id}/fork` | `POST` | Fork conversation up to a specific message |
| `/api/chat/conversations/{id}/forks` | `GET` | List forks of a conversation |
| `/api/chat/conversations/{id}/export?format=md\|pdf` | `GET` | Export conversation (Markdown or PDF) |

**Tags**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tags/tags` | `GET` | List all tags |
| `/api/tags/tags` | `POST` | Create tag (`name`, `color`) |
| `/api/tags/tags/{id}` | `DELETE` | Delete tag |
| `/api/tags/conversations/{id}/tags` | `GET` | Tags of a conversation |
| `/api/tags/conversations/{id}/tags` | `POST` | Assign tag to conversation |
| `/api/tags/conversations/{id}/tags/{tag_id}` | `DELETE` | Remove tag from conversation |
| `/api/tags/documents/{id}/tags` | `GET` | Tags of a document |
| `/api/tags/documents/{id}/tags` | `POST` | Assign tag to document |
| `/api/tags/documents/{id}/tags/{tag_id}` | `DELETE` | Remove tag from document |

### SSE events (streaming)

Chat and RAG responses are sent via Server-Sent Events with the following event types:

| Event | Description | Data |
|-------|-------------|------|
| `thinking` | Internal reasoning token (chain of thought) | `{ "token": "..." }` |
| `sources` | RAG sources used — fires **before** tokens (RAG mode only) | `[{ "filename", "chunk_index", "excerpt" }]` |
| `token` | User-visible response token | `{ "token": "..." }` |
| `done` | End of response | `{ "message_id": 42 }` |
| `error` | Error during generation | `{ "error": "..." }` |

### Supported document formats

PDF, DOCX, EPUB, HTML, TXT, Markdown

### New features

- **Dual chat mode** — Conversations in `general` mode (Gemma4:26b / Qwen3.6:27b) or `rag` mode (with selected collections)
- **Per-message model** — Pass `model: "qwen3.6:27b"` in the body to use the coding model on any message
- **Conversation forking** — Branch any conversation from a specific message
- **Auto-titles** — Conversation title is auto-generated after the first message
- **Export** — Download conversations as Markdown or PDF
- **Search** — Full-text search across all conversation history
- **Source citations** — The `sources` event returns the RAG chunks used, before the response
- **Suggested questions** — Auto-generated and cached per collection
- **Document summaries** — Auto-generated after indexing
- **Document comparison** — Compare two documents via SSE streaming
- **Tags** — Label system for organizing conversations and documents

### Tech stack

| Component | Role |
|-----------|------|
| **FastAPI** | Async REST API |
| **Ollama** | Local LLM (Gemma4:26b · Qwen3.6:27b) + embeddings (mxbai-embed-large) |
| **ChromaDB** | Vector database (cosine distance) |
| **SQLite + SQLAlchemy** | Async metadata (conversations, documents, collections, tags) |
| **SSE (sse-starlette)** | Response streaming with thinking/sources/content |
| **fpdf2** | Conversation export to PDF |
| **PyMuPDF** | PDF text extraction |
| **python-docx** | DOCX text extraction |
| **ebooklib** | EPUB text extraction |
