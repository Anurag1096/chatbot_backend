# Python Learning Path — FastAPI Layered API Style

A practical guide for mastering the Python skills needed to build and extend this chatbot backend (`chatbot_Be`).

This style is **not MVC**. It is a **layered API architecture**: routers → schemas → services → (future) database.

---

## Table of Contents

1. [Python Fundamentals](#1-python-fundamentals-must-have)
2. [Object-Oriented Python](#2-object-oriented-python-light-not-heavy-oop)
3. [Generators & Async Generators](#3-generators--async-generators-important-for-sse)
4. [Pydantic & Data Validation](#4-pydantic--data-validation)
5. [FastAPI Core Concepts](#5-fastapi-core-concepts)
6. [API / HTTP Basics](#6-api--http-basics)
7. [Architecture Thinking](#7-architecture-thinking-the-style-part)
8. [Async Python (Deeper)](#8-async-python-go-deeper-when-you-add-ragllm)
9. [Next Steps for RAG Chatbot](#9-next-steps-for-your-rag-chatbot)
10. [Suggested Learning Order](#10-suggested-learning-order)
11. [How to Practice](#11-how-to-practice-best-way)
12. [Resources](#12-resources-focused-not-overwhelming)

---

## 1. Python Fundamentals (must-have)

These show up everywhere in the codebase.

| Topic | Why it matters in this project |
|-------|-------------------------------|
| **Functions & classes** | `ChatService`, route handlers |
| **Type hints** | `str`, `list[HistoryMessage]`, `dict[str, Any]` |
| **`typing` module** | `Literal`, `Annotated`, `AsyncIterable` |
| **Modules & packages** | `app/routers/`, `from app.services import ...` |
| **Virtual environments** | `.venv`, `pip install -r requirements.txt` |
| **Exceptions** | `HTTPException`, error handling in APIs |
| **Context managers** | `with` blocks, cleanup patterns |
| **`async` / `await` basics** | SSE streaming, non-blocking I/O |

**Practice:** Rewrite small pieces of the app without looking — a class, a typed function, an import across folders.

---

## 2. Object-Oriented Python (light, not heavy OOP)

You don't need deep inheritance trees. Focus on:

- **Classes as containers for logic** — `ChatService`
- **Methods** — `build_reply()`, `stream_response()`
- **When to use a class vs plain functions** — service class vs utility functions

**Goal:** Understand "logic lives in a service class, routes just call it."

---

## 3. Generators & Async Generators (important for SSE)

Streaming code depends on this.

```python
# Regular generator
def tokenize(text):
    for word in words:
        yield word

# Async generator (used in this project)
async def stream_response(...):
    for token in tokens:
        yield StreamChunk(delta=token)
        await asyncio.sleep(0.04)
```

Learn:

- `yield` vs `return`
- `async def` + `async for`
- Why streaming sends data piece-by-piece instead of all at once

**Practice:** Build a tiny script that streams lines from a file with `yield`.

---

## 4. Pydantic & Data Validation

`app/schemas/chat.py` is all Pydantic.

Learn:

- `BaseModel`
- Field types and defaults
- `Field(alias="conversationId")` for camelCase JSON from the frontend
- `model_dump()`, `exclude_none=True`
- Validation errors (what happens when JSON is wrong)

**Goal:** Define request/response models confidently and understand why schemas are not database models.

---

## 5. FastAPI Core Concepts

| Concept | Where in this project |
|---------|----------------------|
| **App & routers** | `app/main.py`, `APIRouter` |
| **Path operations** | `@router.post("/chat")` |
| **Request body** | `ChatRequest` |
| **Dependency injection** | `ChatServiceDep = Annotated[..., Depends(...)]` |
| **Middleware** | CORS |
| **Streaming responses** | `EventSourceResponse` |
| **HTTP status codes** | 400, 422, 500 |

**Practice:** Add a new endpoint yourself, e.g. `GET /version`, using a router + schema.

---

## 6. API / HTTP Basics

Backend development isn't only Python — you need HTTP.

Learn:

- **Methods:** GET, POST
- **Headers:** `Content-Type`, `Accept`, CORS
- **Status codes:** 200, 400, 422, 500
- **JSON request/response bodies**
- **SSE (Server-Sent Events)** — how `data: {...}\n\n` works

**Practice:** Use `curl` on `/health` and `/chat` and read the raw response.

```bash
curl http://127.0.0.1:8000/health

curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"Hello","history":[]}'
```

---

## 7. Architecture Thinking (the "style" part)

Mental model for this project:

```
Router       → HTTP in/out
Schema       → validate data shape
Service      → business logic
Dependency   → wire things together
Config       → settings in one place
```

Key ideas:

- **Separation of concerns** — routes stay thin
- **Dependency injection** — don't construct everything inside routes
- **DTO vs domain model** — `ChatRequest` is not a DB table
- **React is the View** — backend returns data, not HTML

**Pattern name:** Router → Service → Repository (when you add a database).

See also: `docs/ARCHITECTURE.md` for a full walkthrough of this codebase.

---

## 8. Async Python (go deeper when you add RAG/LLM)

Currently the code only `await asyncio.sleep()`. With RAG you will `await` real I/O.

Learn:

- When to use `async def` vs regular `def`
- `asyncio.gather()` for parallel tasks
- Not blocking the event loop (no heavy sync work inside `async`)
- Calling external APIs asynchronously (LLM, vector DB)

**FastAPI rule:** Use `async` only when the code inside is truly async; otherwise use normal `def`.

---

## 9. Next Steps for Your RAG Chatbot

After the fundamentals above, learn these for the full app:

| Topic | Used for |
|-------|----------|
| **Environment variables** | API keys in `config.py` |
| **HTTPX** | Calling OpenAI / other LLM APIs |
| **File handling** | Document upload for RAG |
| **SQLModel / SQLAlchemy** | Saving conversations |
| **Vector DB basics** | Chroma, Pinecone, etc. |
| **Prompt + retrieval flow** | Real RAG in `services/rag.py` |
| **Testing** | `pytest` + FastAPI `TestClient` |
| **Deployment** | Railway, Render, Docker |

---

## 10. Suggested Learning Order

```
Python basics + types
        ↓
Functions, classes, modules
        ↓
Pydantic models
        ↓
FastAPI routes + Depends
        ↓
Generators + async/await
        ↓
SSE + streaming APIs
        ↓
Layered architecture
        ↓
RAG, DB, deployment
```

---

## 11. How to Practice (best way)

1. **Read your own codebase** — map each file to router / schema / service / config.
2. **Add small features** — e.g. `GET /version`, logging, request timing.
3. **Replace placeholder logic** — call a real LLM inside `ChatService`.
4. **Write tests** — test `ChatService.build_reply()` without HTTP.
5. **Break things on purpose** — invalid JSON, empty message, cancel mid-stream.

### Exercises mapped to this repo

| Exercise | File(s) to edit |
|----------|-----------------|
| Add `GET /version` | `app/routers/health.py` |
| Add API key to config | `app/core/config.py` |
| Change stream speed | `app/core/config.py` → `stream_token_delay_seconds` |
| Add error chunk on failure | `app/services/chat.py`, `app/routers/chat.py` |
| Add document upload endpoint | New `app/routers/documents.py`, `app/services/rag.py` |

---

## 12. Resources (focused, not overwhelming)

| Resource | Topic |
|----------|-------|
| [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) | Routes, Depends, streaming |
| [Pydantic Docs](https://docs.pydantic.dev/) | Schemas and validation |
| [Python typing docs](https://docs.python.org/3/library/typing.html) | `Annotated`, `AsyncIterable` |
| [Real Python — Generators](https://realpython.com/introduction-to-python-generators/) | `yield` and streaming |
| [Real Python — asyncio](https://realpython.com/async-io-python-asyncio/) | Async fundamentals |

---

## Quick Summary

To master **this** style of programming:

> **Python types + Pydantic + FastAPI routers/Depends + async generators + layered architecture (router → service → data)**

MVC is less relevant here. Think in **API layers**, not HTML views.

---

## Map: Concepts → Project Files

| Concept | File |
|---------|------|
| App setup & CORS | `app/main.py` |
| Settings | `app/core/config.py` |
| Request/response shapes | `app/schemas/chat.py` |
| Business logic | `app/services/chat.py` |
| Dependency injection | `app/dependencies.py` |
| HTTP endpoints | `app/routers/chat.py`, `app/routers/health.py` |
| Server entry point | `main.py` |
