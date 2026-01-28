# Python FastAPI Backend Design Patterns Guide

> A comprehensive guide to building scalable, maintainable Python backends using FastAPI, LangGraph, and modern async patterns.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Project Structure](#project-structure)
3. [Component-as-Service Architecture](#component-as-service-architecture)
4. [State Management Patterns](#state-management-patterns)
5. [API Design Patterns](#api-design-patterns)
6. [Session & Output Storage](#session--output-storage)
7. [Configuration Management](#configuration-management)
8. [Error Handling Strategy](#error-handling-strategy)
9. [Async Patterns](#async-patterns)
10. [Singleton Services](#singleton-services)
11. [Quick Reference](#quick-reference)

---

## Design Philosophy

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Component Isolation** | Each feature lives in its own directory with standardized file structure |
| **Single Responsibility** | Services handle business logic; agents handle workflow integration; routers handle HTTP |
| **Type Safety** | Pydantic models for all data boundaries; Generic types for reusable base classes |
| **Async-First** | All I/O operations are async; enables non-blocking concurrent execution |
| **Partial State Updates** | Workflow agents return only changed fields; framework handles merging |
| **Audit Everything** | Every LLM call, search, and decision is persisted for debugging and compliance |

### When to Use This Architecture

This architecture is well-suited for:

- **Multi-agent orchestration systems** (LangGraph, LangChain)
- **RAG (Retrieval-Augmented Generation) applications**
- **API-first backends** with streaming requirements
- **Systems requiring audit trails** for compliance
- **Applications with complex, multi-step workflows**

---

## Project Structure

### Recommended Directory Layout

```
my-backend/
├── app/                          # Main application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI app initialization
│   │
│   ├── components/               # Feature components
│   │   ├── base/                 # Shared base classes
│   │   │   ├── component.py      # BaseComponent abstract class
│   │   │   ├── config.py         # Settings (Pydantic)
│   │   │   ├── exceptions.py     # Exception hierarchy
│   │   │   └── logging.py        # Structured logging setup
│   │   │
│   │   ├── {feature}/            # Each feature in own folder
│   │   │   ├── models.py         # Pydantic schemas
│   │   │   ├── service.py        # Business logic
│   │   │   ├── agent.py          # Workflow node wrapper
│   │   │   ├── router.py         # FastAPI endpoints
│   │   │   └── prompts.py        # LLM prompts (if applicable)
│   │   │
│   │   ├── orchestrator/         # Workflow orchestration
│   │   │   ├── workflow.py       # Graph definition
│   │   │   ├── state.py          # State TypedDict
│   │   │   └── service.py        # Execution service
│   │   │
│   │   └── session/              # Session management
│   │
│   ├── rag/                      # RAG layer (if applicable)
│   │   ├── embeddings.py         # Embedding service
│   │   ├── vector_store.py       # Vector DB wrapper
│   │   └── hybrid_search.py      # Search fusion logic
│   │
│   ├── services/                 # Shared business services
│   │   └── context_assembler.py  # Document loading
│   │
│   └── utils/                    # Utilities
│       ├── audit.py              # Audit trail manager
│       ├── json_repair.py        # LLM output parsing
│       └── ollama_client.py      # LLM client
│
├── scripts/                      # Database & utility scripts
│   ├── init_vector_db.py
│   └── reindex.py
│
├── data/                         # Persistent data
│   ├── raw/                      # Source data files
│   └── sessions/                 # Session audit trails
│
├── config/                       # Configuration files
│   └── settings.yaml
│
├── tests/                        # Test files
│
├── requirements.txt
├── .env.example
├── CLAUDE.md                     # AI assistant context
└── README.md
```

### Directory Responsibilities

| Directory | Responsibility |
|-----------|----------------|
| `app/components/` | Feature modules with standardized structure |
| `app/components/base/` | Shared abstractions (BaseComponent, Settings, Exceptions) |
| `app/rag/` | Vector storage, embeddings, search logic |
| `app/services/` | Cross-cutting business services |
| `app/utils/` | General utilities (audit, JSON repair, clients) |
| `scripts/` | One-off database scripts, migrations |
| `data/` | Persistent storage (vector DB, sessions, uploads) |

---

## Component-as-Service Architecture

### The BaseComponent Pattern

Create an abstract base class that all feature services extend:

```python
# app/components/base/component.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")

class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """Abstract base for all component services."""

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Unique identifier for logging and metrics."""
        pass

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """Main processing entry point."""
        pass

    async def health_check(self) -> dict:
        """Component-level health status."""
        return {"component": self.component_name, "status": "healthy"}

    async def __call__(self, request: TRequest) -> TResponse:
        """Allow direct invocation: component(request)."""
        return await self.process(request)
```

**Benefits:**
- Type-safe via Python generics
- Enforces consistent interface
- Easy to mock for testing
- Works with both REST and workflow integrations

### Standard Component File Structure

Every feature component should have these files:

#### 1. models.py - Data Schemas

```python
# app/components/{feature}/models.py
from pydantic import BaseModel, Field
from typing import Optional, List

class FeatureRequest(BaseModel):
    """Input schema for the feature."""
    session_id: str = Field(..., description="Session identifier")
    input_data: str = Field(..., min_length=1, max_length=10000)
    options: Optional[dict] = Field(default=None)

class FeatureResponse(BaseModel):
    """Output schema for the feature."""
    session_id: str
    result: dict
    metadata: Optional[dict] = None
```

#### 2. service.py - Business Logic

```python
# app/components/{feature}/service.py
from app.components.base.component import BaseComponent
from app.components.base.logging import get_logger
from .models import FeatureRequest, FeatureResponse

logger = get_logger("feature")

class FeatureService(BaseComponent[FeatureRequest, FeatureResponse]):
    """Business logic for the feature."""

    @property
    def component_name(self) -> str:
        return "feature"

    async def process(self, request: FeatureRequest) -> FeatureResponse:
        logger.info("processing_request", session_id=request.session_id)

        # Business logic here
        result = await self._do_work(request.input_data)

        return FeatureResponse(
            session_id=request.session_id,
            result=result,
        )

    async def _do_work(self, data: str) -> dict:
        """Internal processing logic."""
        # Implementation details
        pass
```

#### 3. agent.py - Workflow Node Wrapper

```python
# app/components/{feature}/agent.py
from typing import Any, Dict
from .service import FeatureService
from .models import FeatureRequest

_service: FeatureService | None = None

def get_service() -> FeatureService:
    """Singleton service getter."""
    global _service
    if _service is None:
        _service = FeatureService()
    return _service

async def feature_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node wrapper for the feature service.

    IMPORTANT: Returns ONLY changed fields (partial state update).
    """
    service = get_service()

    try:
        request = FeatureRequest(
            session_id=state["session_id"],
            input_data=state["input_data"],
        )
        response = await service.process(request)

        # Return ONLY changed fields
        return {
            "feature_output": response.result,
            "status": "feature_complete",
            "current_agent": "next_agent",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "current_agent": "error_handler",
        }
```

#### 4. router.py - API Endpoints

```python
# app/components/{feature}/router.py
from fastapi import APIRouter, Depends, HTTPException
from app.components.base.exceptions import ComponentError
from .service import FeatureService
from .models import FeatureRequest, FeatureResponse

router = APIRouter(prefix="/feature", tags=["Feature"])

_service: FeatureService | None = None

def get_service() -> FeatureService:
    global _service
    if _service is None:
        _service = FeatureService()
    return _service

@router.post("/process", response_model=FeatureResponse)
async def process_feature(
    request: FeatureRequest,
    service: FeatureService = Depends(get_service),
) -> FeatureResponse:
    """Process a feature request."""
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
```

### Component Integration Points

A component can be used in multiple ways:

| Integration | How | Use Case |
|-------------|-----|----------|
| **REST API** | Via router.py endpoints | Direct HTTP calls |
| **Workflow Node** | Via agent.py wrapper | LangGraph orchestration |
| **Direct Call** | Via service.process() | Internal service-to-service |
| **Testing** | Via service instance | Unit/integration tests |

---

## State Management Patterns

### Workflow State Definition

Use `TypedDict` with `total=False` for partial state updates:

```python
# app/components/orchestrator/state.py
from typing import TypedDict, List, Dict, Optional, Literal, Annotated
import operator

class WorkflowState(TypedDict, total=False):
    """Workflow state allowing partial updates.

    total=False means all fields are optional,
    enabling agents to return only changed fields.
    """

    # Session Context (set once at start)
    session_id: str
    input_text: str

    # Agent Outputs (set by individual agents)
    step1_output: Dict
    step2_output: Dict
    step3_output: Dict

    # Control Fields
    status: Literal[
        "started",
        "step1_complete",
        "step2_complete",
        "step3_complete",
        "completed",
        "error",
    ]
    current_agent: str
    error_message: Optional[str]

    # Append-only fields (use reducer)
    messages: Annotated[List[Dict], operator.add]
```

### Partial State Update Pattern

**Critical Rule:** Agents return ONLY the fields they change.

```python
# ✅ CORRECT: Partial update
async def my_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    result = await do_work(state["input_text"])
    return {
        "step1_output": result,
        "status": "step1_complete",
        "current_agent": "step2",
    }

# ❌ WRONG: Copying unchanged fields
async def my_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    result = await do_work(state["input_text"])
    return {
        "session_id": state["session_id"],  # DON'T copy!
        "input_text": state["input_text"],  # DON'T copy!
        "step1_output": result,
        "status": "step1_complete",
    }
```

**Why This Matters:**
- LangGraph auto-merges partial returns
- Prevents accidental state loss
- Enables parallel agent execution
- Reduces boilerplate code

### Append-Only Fields with Reducers

For fields that should accumulate rather than replace:

```python
from typing import Annotated
import operator

class WorkflowState(TypedDict, total=False):
    # This field APPENDS new items instead of replacing
    messages: Annotated[List[Dict], operator.add]

# In agent
return {
    "messages": [{"role": "agent1", "content": "Done"}]
}
# Result: New message APPENDED to existing list
```

### Workflow Graph Definition

```python
# app/components/orchestrator/workflow.py
from langgraph.graph import StateGraph, END
from .state import WorkflowState

def create_workflow() -> StateGraph:
    """Create the complete workflow graph."""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("step1", step1_agent)
    workflow.add_node("step2", step2_agent)
    workflow.add_node("step3", step3_agent)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("step1")

    # Linear edges
    workflow.add_edge("step1", "step2")
    workflow.add_edge("step2", "step3")
    workflow.add_edge("step3", END)

    # Conditional routing
    workflow.add_conditional_edges(
        "step1",
        route_after_step1,
        {
            "step2": "step2",
            "error_handler": "error_handler",
        }
    )

    workflow.add_edge("error_handler", END)

    return workflow.compile()

def route_after_step1(state: WorkflowState) -> str:
    """Route based on state after step1."""
    if state.get("status") == "error":
        return "error_handler"
    return "step2"
```

---

## API Design Patterns

### Router Registration Pattern

```python
# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.components.base.config import get_settings
from app.components.base.logging import configure_logging
from app.components.feature.router import router as feature_router
from app.components.session.router import router as session_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup
    configure_logging(settings.environment)
    # Initialize singletons, verify connections
    yield
    # Shutdown
    # Cleanup resources

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount component routers
app.include_router(session_router, prefix="/api/v1")
app.include_router(feature_router, prefix="/api/v1")

# System endpoints
@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}
```

### SSE Streaming Pattern

For real-time progress updates:

```python
# app/components/orchestrator/router.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

@router.post("/run/stream")
async def run_stream(request: PipelineRequest) -> StreamingResponse:
    """Execute pipeline with real-time SSE progress updates."""
    service = get_service()
    return StreamingResponse(
        service.process_streaming(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

# In service
async def process_streaming(self, request: PipelineRequest):
    """Generator that yields SSE events."""
    yield f"event: start\ndata: {json.dumps({'session_id': request.session_id})}\n\n"

    for step_result in await self.execute_steps():
        yield f"event: progress\ndata: {json.dumps(step_result)}\n\n"

    yield f"event: complete\ndata: {json.dumps({'status': 'done'})}\n\n"
```

### API Versioning

Use URL path prefixing for API versioning:

```
/api/v1/feature/endpoint  # Current version
/api/v2/feature/endpoint  # Future version (when needed)
```

---

## Session & Output Storage

### Session Directory Structure

Organize session outputs by date and session ID:

```
sessions/
└── {YYYY-MM-DD-HHMM}/
    └── {session_id}/
        ├── session_metadata.json    # Overall metadata
        ├── step1_input/
        │   ├── request.json         # Input data
        │   └── extracted_data.json  # Processed input
        ├── step2_processing/
        │   ├── llm_request.json     # LLM call metadata
        │   ├── input_prompt.txt     # Full prompt sent
        │   ├── raw_response.txt     # Raw LLM output
        │   └── parsed_output.json   # Parsed/structured output
        └── final_output.json        # Complete result
```

### AuditTrailManager Pattern

```python
# app/utils/audit.py
from pathlib import Path
from datetime import datetime
import json

class AuditTrailManager:
    """Manages session audit trail persistence."""

    def __init__(self, session_id: str, base_dir: str = "sessions"):
        self.session_id = session_id
        self.session_dir = self._get_or_create_session_dir(base_dir)
        self._init_metadata()

    def _get_or_create_session_dir(self, base_dir: str) -> Path:
        """Find existing or create new session directory."""
        base = Path(base_dir)

        # Check if session already exists
        for date_dir in base.iterdir():
            session_path = date_dir / self.session_id
            if session_path.exists():
                return session_path

        # Create new with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        session_dir = base / timestamp / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _init_metadata(self):
        """Initialize session metadata file."""
        metadata_file = self.session_dir / "session_metadata.json"
        if not metadata_file.exists():
            self.save_json("session_metadata.json", {
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(),
                "steps_completed": [],
                "timing": {},
            })

    def save_json(self, filename: str, data: dict, subfolder: str = None) -> Path:
        """Save JSON artifact."""
        target_dir = self.session_dir / subfolder if subfolder else self.session_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def save_text(self, filename: str, content: str, subfolder: str = None) -> Path:
        """Save text artifact."""
        target_dir = self.session_dir / subfolder if subfolder else self.session_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / filename
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def load_json(self, filename: str, subfolder: str = None) -> dict:
        """Load JSON artifact."""
        target_dir = self.session_dir / subfolder if subfolder else self.session_dir
        filepath = target_dir / filename
        with open(filepath) as f:
            return json.load(f)

    def record_timing(self, step_name: str, duration_ms: int):
        """Record step timing in metadata."""
        metadata = self.load_json("session_metadata.json")
        metadata["timing"][step_name] = duration_ms
        self.save_json("session_metadata.json", metadata)

    def add_step_completed(self, step_name: str):
        """Mark a step as completed."""
        metadata = self.load_json("session_metadata.json")
        if step_name not in metadata["steps_completed"]:
            metadata["steps_completed"].append(step_name)
        self.save_json("session_metadata.json", metadata)

    def update_metadata(self, updates: dict):
        """Update metadata with new values."""
        metadata = self.load_json("session_metadata.json")
        metadata.update(updates)
        self.save_json("session_metadata.json", metadata)
```

### Using AuditTrailManager in Services

```python
# In a service that calls an LLM
async def process(self, request: FeatureRequest) -> FeatureResponse:
    audit = AuditTrailManager(request.session_id)
    start_time = time.time()

    # Save input
    audit.save_json("request.json", request.model_dump(), subfolder="step2_processing")

    # Build and save prompt
    prompt = self._build_prompt(request)
    audit.save_text("input_prompt.txt", prompt, subfolder="step2_processing")

    # Call LLM and save metadata
    response, metadata = await self.llm_client.generate(prompt)
    audit.save_json("llm_request.json", metadata.__dict__, subfolder="step2_processing")
    audit.save_text("raw_response.txt", response, subfolder="step2_processing")

    # Parse and save output
    parsed = self._parse_response(response)
    audit.save_json("parsed_output.json", parsed, subfolder="step2_processing")

    # Record timing
    elapsed_ms = int((time.time() - start_time) * 1000)
    audit.record_timing("step2_processing", elapsed_ms)
    audit.add_step_completed("step2")

    return FeatureResponse(session_id=request.session_id, result=parsed)
```

---

## Configuration Management

### Pydantic Settings Pattern

```python
# app/components/base/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    """Centralized configuration loaded from environment."""

    # Application
    app_name: str = "My Application"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]

    # LLM (example: Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    ollama_timeout: int = 120
    ollama_temperature: float = 0.7
    ollama_max_tokens: int = 4096

    # Vector DB (example: ChromaDB)
    chroma_persist_dir: str = "./data/chroma"

    # Paths
    data_dir: str = "./data"
    sessions_dir: str = "./sessions"
    uploads_dir: str = "./data/uploads"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra env vars

@lru_cache()
def get_settings() -> Settings:
    """Singleton settings (cached by lru_cache decorator)."""
    return Settings()
```

### Configuration Hierarchy

Settings are loaded in this priority order (highest to lowest):

1. **Environment variables** (highest priority)
2. **.env file**
3. **Default values** (in Settings class)

### Using Settings

```python
from app.components.base.config import get_settings

settings = get_settings()

# Use settings throughout the application
client = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    timeout=settings.ollama_timeout,
)
```

---

## Error Handling Strategy

### Exception Hierarchy

```python
# app/components/base/exceptions.py
from typing import Dict, Any, Optional

class ComponentError(Exception):
    """Base exception for all component errors."""

    def __init__(
        self,
        message: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.component = component
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "details": self.details,
        }

# Specific exceptions
class SessionNotFoundError(ComponentError):
    """Session does not exist."""
    pass

class LLMUnavailableError(ComponentError):
    """LLM service is unavailable."""
    pass

class ResponseParsingError(ComponentError):
    """Failed to parse LLM response."""
    pass

class ValidationError(ComponentError):
    """Input validation failed."""
    pass
```

### Error Handling in Routes

```python
@router.post("/process", response_model=FeatureResponse)
async def process(
    request: FeatureRequest,
    service: FeatureService = Depends(get_service),
):
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.exception("unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})
```

### Error Handling in Agents

```python
async def my_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Normal processing
        return {
            "output": result,
            "status": "step_complete",
        }
    except ComponentError as e:
        # Return error state - workflow routes to error_handler
        return {
            "status": "error",
            "error_message": str(e),
            "current_agent": "error_handler",
        }
```

---

## Async Patterns

### Async Service Methods

All I/O-bound operations should be async:

```python
class MyService(BaseComponent[Request, Response]):

    async def process(self, request: Request) -> Response:
        """Main entry point - async."""
        # Parallel operations when possible
        result1, result2 = await asyncio.gather(
            self._fetch_data(request.id),
            self._call_llm(request.text),
        )
        return Response(data=result1, analysis=result2)

    async def _fetch_data(self, id: str) -> dict:
        """Async HTTP call."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_url}/{id}")
            return response.json()

    async def _call_llm(self, text: str) -> str:
        """Async LLM call."""
        return await self.llm_client.generate(text)
```

### Async Agent Functions

```python
async def my_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent nodes must be async for workflow execution."""
    service = get_service()
    response = await service.process(request)
    return {"output": response.model_dump()}
```

---

## Singleton Services

### When to Use Singletons

Use singletons for services that:
- Are expensive to initialize (DB connections, model loading)
- Should maintain consistent state across requests
- Are stateless (don't need per-request isolation)

### Singleton Implementation Pattern

```python
# Module-level singleton
_instance: Optional["MyService"] = None

class MyService:
    """Service with singleton pattern."""

    def __init__(self):
        # Expensive initialization
        self.connection = self._connect()

    @classmethod
    def get_instance(cls) -> "MyService":
        """Get or create singleton instance."""
        global _instance
        if _instance is None:
            _instance = cls()
        return _instance

# Usage
service = MyService.get_instance()
```

### Alternative: Function-Based Singleton

```python
_service: MyService | None = None

def get_service() -> MyService:
    """Singleton getter function."""
    global _service
    if _service is None:
        _service = MyService()
    return _service

# In router
@router.post("/endpoint")
async def endpoint(service: MyService = Depends(get_service)):
    return await service.process()
```

---

## Quick Reference

### Component Checklist

When creating a new component:

- [ ] Create directory: `app/components/{feature}/`
- [ ] Create `models.py` with Request/Response schemas
- [ ] Create `service.py` extending BaseComponent
- [ ] Create `agent.py` with singleton getter and agent function
- [ ] Create `router.py` with FastAPI endpoints
- [ ] Create `prompts.py` if using LLM
- [ ] Add router to `app/main.py`
- [ ] Add agent to workflow graph (if applicable)
- [ ] Write tests

### State Update Rules

| Rule | Description |
|------|-------------|
| Return partial | Only return changed fields |
| No copying | Don't copy unchanged state fields |
| Use reducers | For append-only fields (messages, logs) |
| Status tracking | Always update `status` and `current_agent` |
| Error routing | Set `status: "error"` for workflow routing |

### File Naming Conventions

| File | Purpose |
|------|---------|
| `models.py` | Pydantic request/response schemas |
| `service.py` | Business logic (extends BaseComponent) |
| `agent.py` | LangGraph node wrapper |
| `router.py` | FastAPI endpoints |
| `prompts.py` | LLM prompt templates |
| `__init__.py` | Public exports |

### Common Patterns Summary

| Pattern | Location | Purpose |
|---------|----------|---------|
| BaseComponent | `base/component.py` | Type-safe service interface |
| Singleton | `*/agent.py`, `rag/*.py` | Expensive resource sharing |
| Partial State | `orchestrator/state.py` | Workflow state management |
| Audit Trail | `utils/audit.py` | Session persistence |
| Exception Hierarchy | `base/exceptions.py` | Consistent error handling |
| Pydantic Settings | `base/config.py` | Configuration management |

---

## Next Steps

1. **Copy the directory structure** to your new project
2. **Implement BaseComponent** and exception hierarchy
3. **Create your first component** following the pattern
4. **Add workflow orchestration** if needed
5. **Set up audit trail** for debugging and compliance

This guide provides the foundational patterns. Adapt them to your specific requirements while maintaining the core principles of isolation, type safety, and async-first design.
