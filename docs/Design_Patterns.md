# Python FastAPI Backend Design Patterns Guide

> A comprehensive guide to building scalable, maintainable Python backends using FastAPI, LangGraph, and modern async patterns — grounded in the actual ele-real-ph1-backend codebase.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Project Structure](#project-structure)
3. [Component-as-Service Architecture](#component-as-service-architecture)
4. [State Management Patterns](#state-management-patterns)
5. [Hybrid Search & RRF Fusion](#hybrid-search--rrf-fusion)
6. [RAG Ingestion Patterns](#rag-ingestion-patterns)
7. [API Design Patterns](#api-design-patterns)
8. [Session & Output Storage](#session--output-storage)
9. [Configuration Management](#configuration-management)
10. [Error Handling Strategy](#error-handling-strategy)
11. [Async Patterns](#async-patterns)
12. [Singleton Services](#singleton-services)
13. [LLM Prompt Authoring](#llm-prompt-authoring)
14. [Quick Reference](#quick-reference)

---

## Design Philosophy

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Component Isolation** | Each feature lives in its own directory with standardized file structure |
| **Single Responsibility** | Services handle business logic; agents handle workflow integration; routers handle HTTP |
| **Type Safety** | Pydantic models for all data boundaries; TypedDict for LangGraph state |
| **Async-First** | All I/O operations are async; enables non-blocking concurrent execution |
| **Partial State Updates** | Workflow agents return only changed fields; framework handles merging |
| **Audit Everything** | Every LLM call, search, and decision is persisted for debugging and compliance |

### Why TypedDict for LangGraph State, Not Pydantic

LangGraph nodes return **partial state updates** — each node returns only the fields it changed. LangGraph's runtime merges these partials into the cumulative state. Pydantic models require all required fields at instantiation, which would force every node to copy the entire state even for unchanged fields.

`TypedDict` with `total=False` solves this: all keys are optional, so a node can return just `{"functional_query": "...", "technical_query": "..."}` and LangGraph will shallow-merge it into the existing state without clobbering fields set by previous nodes.

```python
# ✅ CORRECT: TypedDict with total=False
class PipelineState(TypedDict, total=False):
    session_id: str
    input_story: str
    functional_query: str   # set by node 1
    technical_query: str    # set by node 1
    top5_stories: list      # set by node 2
    component_mapping: dict # set by node 3
    status: str
    error_message: str

# ❌ WRONG: Pydantic model — requires all fields on construction
class PipelineState(BaseModel):
    session_id: str
    input_story: str
    functional_query: Optional[str] = None  # verbose and fragile
```

### When to Use This Architecture

This architecture is well-suited for:

- **Multi-agent orchestration systems** (LangGraph, LangChain)
- **RAG (Retrieval-Augmented Generation) applications**
- **API-first backends** with streaming requirements
- **Systems requiring audit trails** for compliance
- **Applications with complex, multi-step workflows**

---

## Project Structure

### Actual Directory Layout

```
ele-real-ph1-backend/
├── src/
│   ├── __init__.py
│   └── components/
│       ├── base/                    # Shared abstractions
│       │   ├── component.py         # BaseComponent ABC
│       │   ├── config.py            # Settings (dotenv)
│       │   └── exceptions.py        # Exception hierarchy
│       │
│       ├── decomposition/           # Node 1: query decomposition
│       │   ├── models.py
│       │   ├── service.py
│       │   ├── agent.py
│       │   └── prompts.py
│       │
│       ├── hybrid_search/           # Node 2: dense + sparse + RRF
│       │   ├── models.py
│       │   ├── service.py           # RRF fusion logic lives here
│       │   ├── agent.py
│       │   ├── chroma_store.py      # Dense search (OpenAI embeddings)
│       │   └── bm25_index.py        # Sparse search (BM25Okapi)
│       │
│       ├── generation/              # Node 3: GPT-4o component mapping
│       │   ├── models.py
│       │   ├── service.py           # Includes component ID validation
│       │   ├── agent.py
│       │   └── prompts.py
│       │
│       ├── ingestion/               # Data prep (run via scripts/ingest.py)
│       │   ├── models.py
│       │   └── service.py           # Chunking + Chroma/BM25 population
│       │
│       └── orchestrator/            # LangGraph pipeline assembly
│           ├── state.py             # PipelineState TypedDict
│           └── workflow.py          # Graph definition + compile
│
├── src/utils/
│   └── audit.py                     # AuditTrailManager
│
├── scripts/
│   └── ingest.py                    # CLI wrapper for ingestion service
│
├── data/
│   ├── raw/
│   │   ├── stories.json             # Historical Jira stories
│   │   └── tdds/                    # {story_id}_tdd.md files
│   ├── chroma_db/                   # Chroma persistent store (generated)
│   └── bm25_index.pkl               # Serialized BM25 index (generated)
│
├── config/
│   └── domain.json                  # Component catalogue (FC-xxx, TC-xxx)
│
├── output/                          # Session audit trails
│   └── {YYYY-MM-DD-HHMM}/
│       └── {session_id}/
│
├── main.py                          # CLI entry point
├── api_server.py                    # FastAPI server
└── requirements.txt
```

### Directory Responsibilities

| Directory | Responsibility |
|-----------|----------------|
| `src/components/` | Feature modules with standardized structure |
| `src/components/base/` | Shared abstractions (BaseComponent, Settings, Exceptions) |
| `src/components/hybrid_search/` | Dense + sparse retrieval, RRF fusion |
| `src/components/ingestion/` | Data chunking + store population |
| `src/components/orchestrator/` | LangGraph graph definition and state |
| `src/utils/` | General utilities (audit trail) |
| `scripts/` | One-off CLI scripts (ingest.py) |
| `data/` | Persistent storage (chroma_db, bm25 index, raw data) |
| `config/` | Domain catalogue JSON |
| `output/` | Per-session audit artifacts |

---

## Component-as-Service Architecture

### The BaseComponent Pattern

Create an abstract base class that all feature services extend:

```python
# src/components/base/component.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")

class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """Abstract base for all component services."""

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Unique identifier for logging and audit."""
        pass

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """Main processing entry point."""
        pass

    async def __call__(self, request: TRequest) -> TResponse:
        """Allow direct invocation: component(request)."""
        return await self.process(request)
```

**Benefits:**
- Type-safe via Python generics
- Enforces consistent interface
- Easy to mock for testing
- Works with both REST and LangGraph workflow integrations

### Standard Component File Structure

Every feature component should have these files:

#### 1. models.py — Data Schemas

```python
# src/components/{feature}/models.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class FeatureRequest:
    session_id: str
    input_data: str

@dataclass
class FeatureResponse:
    session_id: str
    result: dict[str, Any] = field(default_factory=dict)
```

> **Note:** This codebase uses `dataclasses` for internal pipeline models (request/response) and `Pydantic` only at the HTTP API boundary. Dataclasses are lighter-weight and sufficient for internal contracts.

#### 2. service.py — Business Logic

```python
# src/components/{feature}/service.py
from src.components.base.component import BaseComponent
from src.utils.audit import AuditTrailManager
from .models import FeatureRequest, FeatureResponse

_STEP = "step_N_feature_name"

class FeatureService(BaseComponent[FeatureRequest, FeatureResponse]):

    @property
    def component_name(self) -> str:
        return _STEP

    async def process(self, request: FeatureRequest) -> FeatureResponse:
        audit = AuditTrailManager(request.session_id)
        audit.start_timer(_STEP)

        audit.save_json("request.json", {"input": request.input_data}, subfolder=_STEP)

        result = await self._do_work(request.input_data)

        audit.save_json("parsed_output.json", result, subfolder=_STEP)
        audit.stop_timer(_STEP)

        return FeatureResponse(session_id=request.session_id, result=result)
```

#### 3. agent.py — LangGraph Node Wrapper

```python
# src/components/{feature}/agent.py
from typing import Any
from .service import FeatureService
from .models import FeatureRequest

_service: FeatureService | None = None

def _get_service() -> FeatureService:
    """Lazy singleton init."""
    global _service
    if _service is None:
        _service = FeatureService()
    return _service

async def feature_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: returns ONLY changed fields (partial state update)."""
    service = _get_service()
    request = FeatureRequest(
        session_id=state["session_id"],
        input_data=state["input_story"],
    )
    response = await service.process(request)

    # Return ONLY new/changed fields
    return {
        "feature_output": response.result,
        "status": "feature_complete",
    }
```

### Component Integration Points

| Integration | How | Use Case |
|-------------|-----|----------|
| **LangGraph Node** | Via `agent.py` wrapper | Pipeline orchestration |
| **REST API** | Via `router.py` endpoints | Direct HTTP calls |
| **Direct Call** | Via `service.process()` | Internal service-to-service |
| **Testing** | Via service instance | Unit/integration tests |

---

## State Management Patterns

### Workflow State Definition

Use `TypedDict` with `total=False` for partial state updates:

```python
# src/components/orchestrator/state.py
from typing import Any, TypedDict

class PipelineState(TypedDict, total=False):
    """State flowing through the LangGraph pipeline.

    Fields are populated progressively by each node.
    total=False enables agents to return partial state updates.
    """

    # Session (generated once, threaded through all nodes)
    session_id: str

    # Input
    input_story: str

    # Node 1: Query Decomposition
    functional_query: str
    technical_query: str

    # Node 2: Hybrid Search + RRF Fusion
    top5_stories: list[dict[str, Any]]

    # Node 3: Generation
    component_mapping: dict[str, Any]

    # Pipeline metadata
    status: str
    error_message: str
```

### Partial State Update Pattern

**Critical Rule:** Agents return ONLY the fields they change.

```python
# ✅ CORRECT: Partial update
async def decomposition_agent(state: dict[str, Any]) -> dict[str, Any]:
    result = await service.process(...)
    return {
        "functional_query": result.functional_query,
        "technical_query": result.technical_query,
        "status": "decomposition_complete",
    }

# ❌ WRONG: Copying unchanged fields
async def decomposition_agent(state: dict[str, Any]) -> dict[str, Any]:
    result = await service.process(...)
    return {
        "session_id": state["session_id"],    # DON'T copy!
        "input_story": state["input_story"],  # DON'T copy!
        "functional_query": result.functional_query,
        "status": "decomposition_complete",
    }
```

**Why This Matters:**
- LangGraph auto-merges partial returns via shallow dict merge
- Prevents accidental state loss
- Enables parallel agent execution
- Reduces boilerplate code

### Workflow Graph Definition

```python
# src/components/orchestrator/workflow.py
from langgraph.graph import StateGraph, END
from .state import PipelineState

def build_pipeline() -> StateGraph:
    workflow = StateGraph(PipelineState)

    workflow.add_node("query_decomposition", decomposition_agent)
    workflow.add_node("hybrid_search", hybrid_search_agent)
    workflow.add_node("generation", generation_agent)

    workflow.set_entry_point("query_decomposition")
    workflow.add_edge("query_decomposition", "hybrid_search")
    workflow.add_edge("hybrid_search", "generation")
    workflow.add_edge("generation", END)

    return workflow.compile()
```

---

## Hybrid Search & RRF Fusion

This is the core retrieval pattern of the project. It combines two fundamentally different retrieval mechanisms and merges them using Reciprocal Rank Fusion.

### Why Two Indices?

| Index | Type | Strength | Weakness |
|-------|------|----------|----------|
| **Chroma + OpenAI embeddings** | Dense (semantic) | Captures meaning, synonyms, related concepts | Misses exact keyword matches (e.g., "834 feed", procedure codes) |
| **BM25Okapi** | Sparse (lexical) | Exact keyword and term-frequency matching | No semantic understanding |

Healthcare payer stories contain domain-specific codes ("834 feed", "HIPAA 270/271", "CPT codes") that dense embeddings may normalize away. BM25 catches these while dense search catches semantic similarity.

### Two-Pass RRF Pattern

The search executes **4 retrieval calls** per query, then merges with 2 passes of RRF:

```
functional_query → dense search  ─┐
functional_query → BM25 search   ─┤─ RRF pass 1 → functional_merged ─┐
                                   │                                    │
technical_query  → dense search  ─┐                                    ├─ RRF pass 2 → final_merged
technical_query  → BM25 search   ─┤─ RRF pass 1 → technical_merged  ─┘
```

**Why two RRF passes instead of one?**
Merging all 4 lists in a single pass would allow the functional sub-query (usually producing more hits) to dominate. The two-pass approach gives each query dimension equal weight before the final merge.

### RRF Implementation

```python
@staticmethod
def _rrf_merge(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    score(doc) = sum(1 / (k + rank)) across all lists where doc appears.
    k=60 is the standard constant that dampens the influence of rank.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = item

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [{**chunk_data[cid], "rrf_score": scores[cid]} for cid in sorted_ids]
```

**Key properties:**
- Score normalization-free — ranks from different score spaces (cosine similarity vs. BM25 score) are directly comparable
- `k=60` is the standard constant; higher values flatten the curve (less top-rank bias)
- Documents appearing in more lists naturally get higher scores

### Post-RRF Deduplication

After merging, chunks from the same story compete. The deduplication step keeps only the **top-ranked chunk per story_id** to ensure diversity in the final result set:

```python
@staticmethod
def _deduplicate_by_story(ranked: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """Keep only the highest-ranked chunk per story_id."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in ranked:
        sid = item["story_id"]
        if sid not in seen:
            seen.add(sid)
            deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped
```

### Search Constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `TOP_K_RETRIEVAL` | 20 | Results per individual search call |
| `TOP_K_FINAL` | 5 | Distinct stories returned after dedup |
| `RRF_K` | 60 | Rank dampening constant |

---

## RAG Ingestion Patterns

### Ingestion Prerequisites

**CRITICAL:** `scripts/ingest.py` must run before any pipeline invocation. It populates both the Chroma vector store and the BM25 serialized index. Running the pipeline without ingestion will produce empty results or errors.

```bash
python scripts/ingest.py   # run once before using the pipeline
```

### Token-Based Chunking

Documents are chunked by token count, not character count:

```python
CHUNK_SIZE = 512     # tokens per chunk
CHUNK_OVERLAP = 128  # token overlap between consecutive chunks
```

**Why token-based?** Embedding models have context windows measured in tokens (e.g., `text-embedding-3-large` supports up to 8191 tokens). Chunking by tokens ensures each chunk fits within the embedding model's window. Character-based chunking can produce chunks that exceed token limits for certain scripts or token-dense text.

### Dual Store Population

Each ingested chunk is written to **both** stores:

```python
# Dense store: Chroma with OpenAI embeddings
chroma_store.add_documents(chunks)

# Sparse store: BM25Okapi in-memory, serialized to disk
bm25_index.build(chunks)
bm25_index.save("data/bm25_index.pkl")
```

The BM25 index file is generated from controlled internal data (not user input) and is read only within the same trusted environment. It is loaded at service startup.

### Chroma vs FAISS Decision

| | Chroma | FAISS |
|--|--------|-------|
| API style | Document-oriented (add docs with metadata) | Matrix-oriented (add raw vectors) |
| Persistence | Built-in, via `persist_directory` | Manual (save/load index files) |
| Metadata filtering | Native (`where={"story_id": "JIRA-101"}`) | Manual post-filter |
| Setup complexity | Low | Medium-high |

Chroma was chosen because the pipeline needs to filter results by `story_id` and `source_type` metadata. FAISS would require maintaining a separate metadata store and applying post-retrieval filters manually.

---

## API Design Patterns

### Router Registration Pattern

```python
# api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Component Mapping API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/pipeline/run")
async def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Execute the full 3-node pipeline for a Jira story."""
    pipeline = build_pipeline()
    result = await pipeline.ainvoke({
        "session_id": str(uuid4()),
        "input_story": request.story,
        "status": "started",
    })
    return PipelineResponse(component_mapping=result["component_mapping"])
```

### SSE Streaming Pattern

For real-time progress updates:

```python
from fastapi.responses import StreamingResponse
import json

@router.post("/run/stream")
async def run_stream(request: PipelineRequest) -> StreamingResponse:
    """Execute pipeline with real-time SSE progress updates."""
    async def event_generator():
        yield f"event: start\ndata: {json.dumps({'session_id': session_id})}\n\n"

        async for event in pipeline.astream(initial_state):
            yield f"event: progress\ndata: {json.dumps(event)}\n\n"

        yield f"event: complete\ndata: {json.dumps({'status': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
```

---

## Session & Output Storage

### Session Directory Structure

```
output/
└── {YYYY-MM-DD-HHMM}/
    └── {session_id}/
        ├── session_metadata.json        # Timing + steps completed
        ├── step_1_query_decomposition/
        │   ├── request.json             # Input story
        │   ├── input_prompt.txt         # Full prompt sent to LLM
        │   ├── raw_response.txt         # Raw LLM output
        │   └── parsed_output.json       # {functional_query, technical_query}
        ├── step_2_hybrid_search/
        │   ├── request.json             # {functional_query, technical_query}
        │   └── parsed_output.json       # top5_stories with full content
        ├── step_3_generation/
        │   ├── request.json             # {input_story, top5_story_ids}
        │   ├── input_prompt.txt         # Full prompt sent to LLM
        │   ├── raw_response.txt         # Raw LLM output
        │   └── parsed_output.json       # {functional_components, technical_components}
        └── final_output.json            # Complete {input_story, component_mapping}
```

### AuditTrailManager Pattern

```python
# src/utils/audit.py
from pathlib import Path
from datetime import datetime
import json, time

class AuditTrailManager:
    """Manages session audit trail persistence."""

    def __init__(self, session_id: str, base_dir: str = "output"):
        self.session_id = session_id
        self.session_dir = self._get_or_create_session_dir(base_dir)
        self._timers: dict[str, float] = {}

    def start_timer(self, step_name: str) -> None:
        self._timers[step_name] = time.time()

    def stop_timer(self, step_name: str) -> None:
        if step_name in self._timers:
            elapsed_ms = int((time.time() - self._timers[step_name]) * 1000)
            self.record_timing(step_name, elapsed_ms)

    def save_json(self, filename: str, data: dict, subfolder: str = None) -> Path:
        target_dir = self.session_dir / subfolder if subfolder else self.session_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def save_text(self, filename: str, content: str, subfolder: str = None) -> Path:
        target_dir = self.session_dir / subfolder if subfolder else self.session_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / filename
        filepath.write_text(content)
        return filepath
```

### Using AuditTrailManager in Services

The standard pattern in every service's `process()` method:

```python
async def process(self, request: FeatureRequest) -> FeatureResponse:
    audit = AuditTrailManager(request.session_id)
    audit.start_timer(_STEP)

    # 1. Save inputs
    audit.save_json("request.json", {...}, subfolder=_STEP)

    # 2. For LLM services: save the full prompt
    audit.save_text("input_prompt.txt", prompt, subfolder=_STEP)

    # 3. Call LLM / search / external service
    raw = await self._client.generate(prompt)

    # 4. Save raw output before parsing
    audit.save_text("raw_response.txt", raw, subfolder=_STEP)

    # 5. Parse and save structured output
    parsed = json.loads(raw)
    audit.save_json("parsed_output.json", parsed, subfolder=_STEP)

    audit.stop_timer(_STEP)
    return FeatureResponse(...)
```

**Why save raw LLM output before parsing?** LLM responses can be malformed JSON. Saving the raw output before the `json.loads()` call means you can debug parse failures even after the exception is raised.

---

## Configuration Management

### Dotenv Settings Pattern

```python
# src/components/base/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# LLM
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Retrieval tuning
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "5"))
RRF_K: int = int(os.getenv("RRF_K", "60"))

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "128"))

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
BM25_INDEX_PATH = str(BASE_DIR / "data" / "bm25_index.pkl")
DOMAIN_CONFIG_PATH = BASE_DIR / "config" / "domain.json"
```

### Configuration Hierarchy

Settings are loaded in this priority order (highest to lowest):

1. **Environment variables** (highest priority)
2. **.env file**
3. **Default values** (in `os.getenv("KEY", "default")` calls)

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API authentication |
| `LLM_MODEL` | `gpt-4o` | Model for decomposition and generation |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Model for dense embeddings |
| `TOP_K_RETRIEVAL` | `20` | Results per individual search call |
| `TOP_K_FINAL` | `5` | Distinct stories after deduplication |
| `RRF_K` | `60` | RRF rank dampening constant |
| `CHUNK_SIZE` | `512` | Tokens per ingestion chunk |
| `CHUNK_OVERLAP` | `128` | Token overlap between chunks |

---

## Error Handling Strategy

### Exception Hierarchy

```python
# src/components/base/exceptions.py

class LLMResponseError(Exception):
    """Raised when LLM output cannot be parsed as valid JSON."""

    def __init__(self, component: str, raw_response: str, reason: str):
        self.component = component
        self.raw_response = raw_response
        self.reason = reason
        super().__init__(f"[{component}] LLM response parse failed: {reason}")
```

### Component ID Validation — Hallucination Guard

After GPT-4o returns a component mapping, the generation service **validates every returned component ID against `domain.json`** and strips any that don't exist:

```python
# src/components/generation/service.py

# Validate that all component IDs exist in domain.json
valid_ids = {c["id"] for c in components}
for category in ("functional_components", "technical_components"):
    mapping[category] = [
        comp for comp in mapping.get(category, [])
        if comp.get("component_id") in valid_ids
    ]

# Enforce top-N limits — sort by confidence_score desc, then slice
limits = {"functional_components": TOP_N_FUNCTIONAL, "technical_components": TOP_N_TECHNICAL}
for category, limit in limits.items():
    mapping[category] = sorted(
        mapping.get(category, []),
        key=lambda c: float(c.get("confidence_score", 0)),
        reverse=True,
    )[:limit]
```

**Why validate?** GPT-4o can hallucinate component IDs that look plausible (`FC-099`) but don't exist in the catalogue. Silent acceptance would corrupt downstream systems consuming the mapping. The validation step ensures only real IDs from `domain.json` are ever returned.

### Error Handling in Agents

```python
async def feature_agent(state: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await _get_service().process(request)
        return {"output": response.result, "status": "complete"}
    except LLMResponseError as e:
        return {"status": "error", "error_message": str(e)}
    except Exception as e:
        return {"status": "error", "error_message": f"Unexpected: {e}"}
```

---

## Async Patterns

### AsyncOpenAI for LLM Calls

All LLM calls use `AsyncOpenAI` (not the sync client):

```python
from openai import AsyncOpenAI

class GenerationService(BaseComponent[...]):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def process(self, request: GenerationRequest) -> GenerationResponse:
        response = await self._client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,           # Deterministic output for mapping tasks
            messages=[
                {"role": "system", "content": "Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
```

**Why `temperature=0`?** Component mapping is a deterministic lookup task — the same story should always produce the same mapping. Temperature 0 removes sampling randomness, maximizing reproducibility.

### Pipeline Invocation

The CLI uses `asyncio.run()` to run the async pipeline from a synchronous entry point:

```python
# main.py
import asyncio

async def main(story: str) -> None:
    pipeline = build_pipeline()
    result = await pipeline.ainvoke({
        "session_id": str(uuid4()),
        "input_story": story,
        "status": "started",
    })
    print(json.dumps(result["component_mapping"], indent=2))

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else SAMPLE_STORY))
```

### Parallel Async Operations

Use `asyncio.gather()` for independent concurrent operations:

```python
# Run both dense searches in parallel
dense_func, dense_tech = await asyncio.gather(
    self._chroma.aquery(request.functional_query, top_k=TOP_K_RETRIEVAL),
    self._chroma.aquery(request.technical_query, top_k=TOP_K_RETRIEVAL),
)
```

---

## Singleton Services

### When to Use Singletons

Use singletons for services that:
- Are expensive to initialize (DB connections, model loading, BM25 index loading)
- Should maintain consistent state across requests
- Are stateless (don't need per-request isolation)

### Function-Based Singleton (Preferred)

```python
# src/components/{feature}/agent.py
_service: FeatureService | None = None

def _get_service() -> FeatureService:
    """Lazy singleton init — service created on first call only."""
    global _service
    if _service is None:
        _service = FeatureService()
    return _service

async def feature_agent(state: dict) -> dict:
    service = _get_service()
    # ...
```

### Singleton for Search Infrastructure

The Chroma and BM25 stores are singletons because loading the serialized BM25 index and connecting to Chroma are expensive operations:

```python
# src/components/hybrid_search/agent.py
_service: HybridSearchService | None = None

def _get_service() -> HybridSearchService:
    global _service
    if _service is None:
        chroma = ChromaStore()        # Connects to Chroma, loads embedding model
        bm25 = BM25Index.load(...)    # Deserializes index file from disk
        _service = HybridSearchService(chroma=chroma, bm25=bm25)
    return _service
```

---

## LLM Prompt Authoring

### XML Tag Section Delimiters

All LLM prompts use XML/HTML-style tags to mark logical sections. This is preferred over markdown `##` headings because:

- **Unambiguous boundaries** — angle-bracket tags cannot appear as literal content inside most domain text, whereas `##` headings can
- **Consistent parsing** — GPT-4o and Claude both have strong structural awareness of XML-style tags in prompts
- **Easier diffs** — section additions/removals are immediately visible

### Standard Tag Vocabulary

| Tag | Purpose |
|-----|---------|
| `<task>` | Describes what the model must do |
| `<rules>` | Hard constraints the model must obey |
| `<input_story>` | The Jira user story being processed |
| `<component_catalogue>` | Injected domain catalogue content |
| `<supporting_evidence>` | RAG-retrieved historical stories + TDDs |
| `<output_format>` | JSON schema / response format instructions |

### Decomposition Prompt Structure

```
You are a query decomposition engine for a healthcare payer system.

<task>
... description of what to produce ...
</task>

<rules>
- Constraint 1
- Constraint 2
- Output valid JSON only. No markdown fencing.
</rules>

<input_story>
{story}
</input_story>

<output_format>
{{"functional_query": "...", "technical_query": "..."}}
</output_format>
```

### Generation Prompt Structure

```
You are a component mapping engine for a healthcare payer system.

<task>
... description of what to produce ...
</task>

<component_catalogue>
{component_catalogue}
</component_catalogue>

<rules>
- Only use component IDs from the catalogue above
- Output valid JSON only. No markdown fencing.
</rules>

<input_story>
{input_story}
</input_story>

<supporting_evidence>
{evidence}
</supporting_evidence>

<output_format>
{{"functional_components": [...], "technical_components": [...]}}
</output_format>
```

### Rules for New Prompts

When adding a new `prompts.py`:

1. Always wrap dynamic injected content (catalogue, evidence, story) in its own named tag pair
2. Group all hard constraints in a single `<rules>` block
3. Keep output format instructions in `<output_format>` — never inline them with the task description
4. The JSON-only instruction (`no markdown fencing`) must live inside `<rules>` or `<output_format>`, not in prose
5. Validate LLM output against known IDs/schema after parsing — never trust the model alone

---

## Quick Reference

### Component Checklist

When creating a new component:

- [ ] Create directory: `src/components/{feature}/`
- [ ] Create `models.py` with Request/Response dataclasses
- [ ] Create `service.py` extending `BaseComponent`
- [ ] Create `agent.py` with `_get_service()` lazy singleton and agent function
- [ ] Add agent to workflow graph in `orchestrator/workflow.py`
- [ ] Add state fields to `PipelineState` in `orchestrator/state.py`
- [ ] Add `prompts.py` if using LLM
- [ ] Add `router.py` if exposing via REST
- [ ] Add timing/audit to service `process()` method

### State Update Rules

| Rule | Description |
|------|-------------|
| Return partial | Only return changed fields |
| No copying | Don't copy unchanged state fields |
| Status tracking | Always update `status` field |
| Error routing | Set `status: "error"` + `error_message` for failure |

### File Naming Conventions

| File | Purpose |
|------|---------|
| `models.py` | Dataclass request/response schemas |
| `service.py` | Business logic (extends BaseComponent) |
| `agent.py` | LangGraph node wrapper + singleton getter |
| `router.py` | FastAPI endpoints |
| `prompts.py` | LLM prompt templates |
| `chroma_store.py` | Dense vector store wrapper |
| `bm25_index.py` | Sparse BM25 index wrapper |

### Common Patterns Summary

| Pattern | Location | Purpose |
|---------|----------|---------|
| BaseComponent | `base/component.py` | Type-safe service interface |
| Singleton | `*/agent.py` | Expensive resource sharing |
| Partial State | `orchestrator/state.py` | LangGraph state management |
| Two-Pass RRF | `hybrid_search/service.py` | Balanced multi-source fusion |
| Component ID Validation | `generation/service.py` | Hallucination guard |
| Audit Trail | `utils/audit.py` | Session persistence |
| Exception Hierarchy | `base/exceptions.py` | Consistent error handling |
| Dotenv Settings | `base/config.py` | Configuration management |

### Pipeline Constants Quick Reference

| Constant | Default | File |
|----------|---------|------|
| `LLM_MODEL` | `gpt-4o` | `base/config.py` |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | `base/config.py` |
| `TOP_K_RETRIEVAL` | `20` | `base/config.py` |
| `TOP_K_FINAL` | `5` | `base/config.py` |
| `RRF_K` | `60` | `base/config.py` |
| `CHUNK_SIZE` | `512` | `base/config.py` |
| `CHUNK_OVERLAP` | `128` | `base/config.py` |
