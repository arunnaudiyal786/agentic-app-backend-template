# FastAPI Patterns

## Table of Contents
1. [Application Structure](#application-structure)
2. [Router Composition](#router-composition)
3. [SSE Streaming](#sse-streaming)
4. [Dependency Injection](#dependency-injection)
5. [CORS Configuration](#cors-configuration)
6. [Error Handling](#error-handling)

---

## Application Structure

### Main Application File (`api_server.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from components.embedding.router import router as embedding_router
from components.retrieval.router import router as retrieval_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    print("Starting up...")
    yield
    # Shutdown: cleanup resources
    print("Shutting down...")

app = FastAPI(
    title="Multi-Agent Backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount component routers
app.include_router(embedding_router, prefix="/v2")
app.include_router(retrieval_router, prefix="/v2")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

## Router Composition

### Standard Router Pattern

```python
# components/{component}/router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from .service import ComponentService
from .models import ComponentRequest, ComponentResponse
from ..base.exceptions import ComponentError

router = APIRouter(prefix="/component", tags=["Component"])

# Singleton service with lazy initialization
_service: Optional[ComponentService] = None

def get_service() -> ComponentService:
    global _service
    if _service is None:
        _service = ComponentService()
    return _service

@router.post("/process", response_model=ComponentResponse)
async def process(
    request: ComponentRequest,
    service: ComponentService = Depends(get_service)
) -> ComponentResponse:
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(service: ComponentService = Depends(get_service)):
    return await service.health_check()
```

---

## SSE Streaming

### Server-Sent Events for Real-Time Updates

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio

@app.post("/api/process")
async def process_with_streaming(request: ProcessRequest):
    async def stream_updates() -> AsyncGenerator[str, None]:
        try:
            # Initialize processing
            yield f"data: {json.dumps({'status': 'started', 'agent': 'init'})}\n\n"

            # Process each stage
            for agent_name in ["classification", "retrieval", "resolution"]:
                yield f"data: {json.dumps({'status': 'processing', 'agent': agent_name})}\n\n"

                # Actual processing
                result = await process_agent(agent_name, request)

                yield f"data: {json.dumps({'status': 'complete', 'agent': agent_name, 'data': result})}\n\n"

            # Final result
            yield f"data: {json.dumps({'status': 'finished', 'final': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream_updates(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
```

### Frontend SSE Consumption

```typescript
const eventSource = new EventSource('/api/process', {
  method: 'POST',
  body: JSON.stringify(request)
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateUI(data);
};

eventSource.onerror = () => {
  eventSource.close();
};
```

---

## Dependency Injection

### Service Singleton Pattern

```python
from functools import lru_cache
from typing import Annotated
from fastapi import Depends

class DatabaseService:
    def __init__(self):
        self.connection = None

    async def connect(self):
        # Initialize connection
        pass

@lru_cache()
def get_database_service() -> DatabaseService:
    return DatabaseService()

# Type alias for cleaner signatures
DatabaseDep = Annotated[DatabaseService, Depends(get_database_service)]

@router.get("/items")
async def get_items(db: DatabaseDep):
    return await db.fetch_items()
```

### Configuration Injection

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

SettingsDep = Annotated[Settings, Depends(get_settings)]

@router.post("/generate")
async def generate(request: Request, settings: SettingsDep):
    client = OpenAI(api_key=settings.openai_api_key)
    # ...
```

---

## CORS Configuration

### Development Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Production Configuration

```python
from config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # From env
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

## Error Handling

### Global Exception Handler

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ComponentError)
async def component_error_handler(request: Request, exc: ComponentError):
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalError",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.debug else None
        }
    )
```

### Request Validation

```python
from pydantic import BaseModel, Field, field_validator

class ProcessRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=10)
    priority: str = Field(default="medium")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        allowed = ["low", "medium", "high", "critical"]
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of: {allowed}")
        return v.lower()
```
