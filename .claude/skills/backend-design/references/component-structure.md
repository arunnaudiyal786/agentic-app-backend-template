# Component Structure Patterns

## Table of Contents
1. [Component Architecture](#component-architecture)
2. [Base Component Class](#base-component-class)
3. [Agent Implementation](#agent-implementation)
4. [Tools Implementation](#tools-implementation)
5. [Service Layer](#service-layer)
6. [Router Implementation](#router-implementation)
7. [Full Component Example](#full-component-example)

---

## Component Architecture

### Directory Structure

```
components/
├── base/
│   ├── __init__.py
│   ├── component.py       # Abstract base class
│   ├── config.py          # Base configuration
│   └── exceptions.py      # Exception hierarchy
├── embedding/
│   ├── __init__.py
│   ├── agent.py           # LangGraph node function
│   ├── tools.py           # LangChain @tool functions
│   ├── service.py         # Business logic
│   ├── models.py          # Pydantic models
│   └── router.py          # FastAPI endpoints
├── retrieval/
│   └── ...
├── classification/
│   └── ...
└── resolution/
    └── ...
```

### Component Responsibilities

| File | Responsibility |
|------|----------------|
| `agent.py` | LangGraph node - orchestrates the component in workflow |
| `tools.py` | LangChain tools - atomic operations callable by LLM |
| `service.py` | Business logic - core processing, external integrations |
| `models.py` | Pydantic models - request/response contracts |
| `router.py` | FastAPI router - HTTP endpoints for direct access |
| `prompts.py` | Prompt templates - LLM instruction templates (optional) |

---

## Base Component Class

### Abstract Base (`components/base/component.py`)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel

TRequest = TypeVar("TRequest", bound=BaseModel)
TResponse = TypeVar("TResponse", bound=BaseModel)

class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """Abstract base for all components."""

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Unique identifier for this component."""
        pass

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """Main processing entry point."""
        pass

    async def health_check(self) -> dict:
        """Check component health status."""
        return {
            "component": self.component_name,
            "status": "healthy"
        }

    async def __call__(self, request: TRequest) -> TResponse:
        """Allow component to be called directly."""
        return await self.process(request)
```

### Base Configuration (`components/base/config.py`)

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class BaseComponentConfig(BaseSettings):
    """Base configuration for components."""
    openai_api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_retries: int = 3
    timeout_seconds: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_base_config() -> BaseComponentConfig:
    return BaseComponentConfig()
```

---

## Agent Implementation

### LangGraph Node Pattern (`agent.py`)

```python
from typing import Dict, Any
from ..base.exceptions import ProcessingError
from .service import RetrievalService
from .models import RetrievalRequest

# Singleton service instance
_service: RetrievalService | None = None

def get_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service

async def retrieval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for retrieval processing.

    Args:
        state: Current workflow state

    Returns:
        Partial state update dict
    """
    try:
        service = get_service()

        # Build request from state
        request = RetrievalRequest(
            query=state["input_text"],
            domain=state.get("classified_domain"),
            top_k=state.get("top_k", 10)
        )

        # Process
        response = await service.process(request)

        # Return partial state update
        return {
            "similar_items": response.results,
            "similarity_scores": response.scores,
            "status": "success",
            "current_agent": "labeling",
            "messages": [{
                "role": "retrieval",
                "content": f"Found {len(response.results)} similar items"
            }]
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Retrieval failed: {str(e)}",
            "current_agent": "error_handler"
        }
```

---

## Tools Implementation

### LangChain Tools Pattern (`tools.py`)

```python
from langchain_core.tools import tool
from typing import Dict, List
import json

@tool
def search_similar_items(
    query: str,
    top_k: int = 10,
    domain: str | None = None
) -> List[Dict]:
    """Search for similar items in the vector store.

    Args:
        query: Search query text
        top_k: Number of results to return
        domain: Optional domain filter

    Returns:
        List of similar items with scores
    """
    from .service import RetrievalService
    service = RetrievalService()

    results = service.search_sync(
        query=query,
        top_k=top_k,
        domain_filter=domain
    )

    return [
        {
            "id": r.id,
            "title": r.title,
            "score": r.score,
            "metadata": r.metadata
        }
        for r in results
    ]

@tool
def classify_domain(text: str) -> Dict:
    """Classify text into a domain category.

    Args:
        text: Text to classify

    Returns:
        Classification result with domain and confidence
    """
    from .service import ClassificationService
    service = ClassificationService()

    result = service.classify_sync(text)
    return {
        "domain": result.domain,
        "confidence": result.confidence,
        "reasoning": result.reasoning
    }

@tool
def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    from .service import EmbeddingService
    service = EmbeddingService()
    return service.embed_sync(text)
```

---

## Service Layer

### Service Class Pattern (`service.py`)

```python
from typing import List, Optional
import asyncio
from openai import AsyncOpenAI

from ..base.component import BaseComponent
from ..base.config import get_base_config
from ..base.exceptions import ProcessingError, ExternalServiceError
from .models import RetrievalRequest, RetrievalResponse, SimilarItem

class RetrievalService(BaseComponent[RetrievalRequest, RetrievalResponse]):
    """Retrieval service with FAISS integration."""

    def __init__(self):
        self.config = get_base_config()
        self.client = AsyncOpenAI(api_key=self.config.openai_api_key)
        self._index = None  # Lazy loaded
        self._metadata = None

    @property
    def component_name(self) -> str:
        return "retrieval"

    async def process(self, request: RetrievalRequest) -> RetrievalResponse:
        """Main processing entry point."""
        # Ensure index is loaded
        await self._ensure_index_loaded()

        # Generate query embedding
        query_embedding = await self._generate_embedding(request.query)

        # Search FAISS index
        scores, indices = self._index.search(
            query_embedding.reshape(1, -1),
            request.top_k
        )

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                metadata = self._metadata[idx]
                results.append(SimilarItem(
                    id=metadata["id"],
                    title=metadata["title"],
                    score=float(score),
                    metadata=metadata
                ))

        # Apply domain filter if specified
        if request.domain:
            results = [r for r in results if r.metadata.get("domain") == request.domain]

        return RetrievalResponse(
            results=results,
            scores=[r.score for r in results],
            query=request.query
        )

    async def _ensure_index_loaded(self):
        """Lazy load FAISS index."""
        if self._index is None:
            import faiss
            import json

            self._index = faiss.read_index("data/faiss_index/index.faiss")
            with open("data/faiss_index/metadata.json") as f:
                self._metadata = json.load(f)

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
                # L2 normalize for cosine similarity
                return embedding / np.linalg.norm(embedding)

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise ExternalServiceError(f"Embedding failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def search_sync(self, query: str, top_k: int, domain_filter: str = None) -> List[SimilarItem]:
        """Synchronous search for tool usage."""
        return asyncio.run(self.process(RetrievalRequest(
            query=query,
            top_k=top_k,
            domain=domain_filter
        ))).results
```

---

## Router Implementation

### FastAPI Router Pattern (`router.py`)

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from .service import RetrievalService
from .models import RetrievalRequest, RetrievalResponse
from ..base.exceptions import ComponentError

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

_service: Optional[RetrievalService] = None

def get_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service

@router.post("/search", response_model=RetrievalResponse)
async def search(
    request: RetrievalRequest,
    service: RetrievalService = Depends(get_service)
) -> RetrievalResponse:
    """Search for similar items."""
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(service: RetrievalService = Depends(get_service)):
    """Check service health."""
    return await service.health_check()

@router.post("/batch-search")
async def batch_search(
    requests: List[RetrievalRequest],
    service: RetrievalService = Depends(get_service)
) -> List[RetrievalResponse]:
    """Batch search for multiple queries."""
    import asyncio
    results = await asyncio.gather(
        *[service.process(req) for req in requests],
        return_exceptions=True
    )

    responses = []
    for result in results:
        if isinstance(result, Exception):
            responses.append(RetrievalResponse(results=[], scores=[], error=str(result)))
        else:
            responses.append(result)

    return responses
```

---

## Full Component Example

### Complete Embedding Component

```
components/embedding/
├── __init__.py
├── agent.py
├── tools.py
├── service.py
├── models.py
└── router.py
```

**`models.py`**
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")
    model: str = Field(default="text-embedding-3-large")

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: int
    usage: dict

class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: str = Field(default="text-embedding-3-large")

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    total_tokens: int
```

**`service.py`**
```python
class EmbeddingService(BaseComponent[EmbeddingRequest, EmbeddingResponse]):
    # ... (implementation as shown above)
```

**`agent.py`**
```python
async def embedding_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... (implementation as shown above)
```

**`tools.py`**
```python
@tool
def generate_embedding(text: str) -> List[float]:
    # ... (implementation as shown above)
```

**`router.py`**
```python
router = APIRouter(prefix="/embedding", tags=["Embedding"])
# ... (implementation as shown above)
```
