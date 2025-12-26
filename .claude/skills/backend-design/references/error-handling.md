# Error Handling Patterns

## Table of Contents
1. [Exception Hierarchy](#exception-hierarchy)
2. [Component Exceptions](#component-exceptions)
3. [FastAPI Error Handling](#fastapi-error-handling)
4. [Agent Error Handling](#agent-error-handling)
5. [Retry Patterns](#retry-patterns)
6. [Error Response Format](#error-response-format)

---

## Exception Hierarchy

### Base Exception Structure (`components/base/exceptions.py`)

```python
from typing import Dict, Any, Optional

class ComponentError(Exception):
    """Base exception for all component errors."""

    def __init__(
        self,
        message: str,
        component: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.component = component
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "details": self.details
        }

class ConfigurationError(ComponentError):
    """Raised when configuration is invalid or missing."""
    pass

class ValidationError(ComponentError):
    """Raised when input validation fails."""
    pass

class ProcessingError(ComponentError):
    """Raised when processing fails during execution."""
    pass

class ExternalServiceError(ComponentError):
    """Raised when external service (OpenAI, FAISS) fails."""

    def __init__(
        self,
        message: str,
        service: str,
        component: str = "unknown",
        status_code: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        super().__init__(message, component, details)
        self.service = service
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["service"] = self.service
        if self.status_code:
            base["status_code"] = self.status_code
        return base

class TimeoutError(ComponentError):
    """Raised when operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        component: str = "unknown"
    ):
        super().__init__(message, component, {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds
```

---

## Component Exceptions

### Component-Specific Errors

```python
# components/embedding/exceptions.py
from ..base.exceptions import ComponentError, ExternalServiceError

class EmbeddingError(ComponentError):
    """Base error for embedding component."""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, component="embedding", details=details)

class EmbeddingGenerationError(EmbeddingError):
    """Failed to generate embedding."""
    pass

class EmbeddingDimensionError(EmbeddingError):
    """Embedding has wrong dimensions."""
    pass

# components/retrieval/exceptions.py
class RetrievalError(ComponentError):
    """Base error for retrieval component."""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, component="retrieval", details=details)

class IndexNotFoundError(RetrievalError):
    """FAISS index file not found."""
    pass

class SearchError(RetrievalError):
    """Search operation failed."""
    pass
```

---

## FastAPI Error Handling

### Global Exception Handlers

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from components.base.exceptions import (
    ComponentError,
    ValidationError,
    ExternalServiceError,
    TimeoutError
)

app = FastAPI()

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors with 400 status."""
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )

@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError):
    """Handle external service errors with 502 status."""
    return JSONResponse(
        status_code=502,
        content=exc.to_dict()
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    """Handle timeout errors with 504 status."""
    return JSONResponse(
        status_code=504,
        content=exc.to_dict()
    )

@app.exception_handler(ComponentError)
async def component_error_handler(request: Request, exc: ComponentError):
    """Handle generic component errors with 500 status."""
    return JSONResponse(
        status_code=500,
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__}
        }
    )
```

### Router-Level Error Handling

```python
from fastapi import APIRouter, HTTPException, Depends
from components.base.exceptions import ComponentError

router = APIRouter(prefix="/embedding", tags=["Embedding"])

@router.post("/generate")
async def generate_embedding(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_service)
):
    try:
        return await service.process(request)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

    except ExternalServiceError as e:
        raise HTTPException(status_code=502, detail=e.to_dict())

    except ComponentError as e:
        raise HTTPException(status_code=500, detail=e.to_dict())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "UnexpectedError",
                "message": str(e)
            }
        )
```

---

## Agent Error Handling

### LangGraph Node Error Pattern

```python
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node with comprehensive error handling."""
    try:
        # Validate required inputs
        if not state.get("input_text"):
            raise ValidationError("Missing required input_text")

        # Process
        result = await process_with_timeout(state["input_text"])

        # Success response
        return {
            "result": result,
            "status": "success",
            "current_agent": "next_node"
        }

    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        return {
            "status": "error",
            "error_message": e.message,
            "error_type": "validation",
            "current_agent": "error_handler"
        }

    except ExternalServiceError as e:
        logger.error(f"External service error: {e.message}")
        return {
            "status": "error",
            "error_message": f"Service unavailable: {e.service}",
            "error_type": "external_service",
            "current_agent": "error_handler"
        }

    except Exception as e:
        logger.exception(f"Unexpected error in processing_node")
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "error_type": "unexpected",
            "current_agent": "error_handler"
        }
```

### Error Handler Node

```python
async def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graceful error handling with fallback output."""
    error_message = state.get("error_message", "Unknown error")
    error_type = state.get("error_type", "unknown")
    failed_agent = state.get("current_agent", "unknown")

    # Log error details
    logger.error(f"Error in {failed_agent}: {error_message} ({error_type})")

    # Generate fallback based on error type
    if error_type == "validation":
        fallback = generate_validation_fallback(state)
    elif error_type == "external_service":
        fallback = generate_service_fallback(state)
    else:
        fallback = generate_generic_fallback(state)

    return {
        "result": fallback,
        "status": "completed_with_errors",
        "error_details": {
            "failed_agent": failed_agent,
            "error_type": error_type,
            "message": error_message
        },
        "messages": [{
            "role": "error_handler",
            "content": f"Recovered from error: {error_message}"
        }]
    }

def generate_generic_fallback(state: Dict) -> Dict:
    """Generate fallback output when processing fails."""
    return {
        "recommendation": "Manual review required",
        "confidence": 0.0,
        "warnings": [
            "Automatic processing failed",
            "Please review manually"
        ],
        "partial_results": {
            "input_received": bool(state.get("input_text")),
            "classification_completed": bool(state.get("classified_domain")),
            "retrieval_completed": bool(state.get("similar_tickets"))
        }
    }
```

---

## Retry Patterns

### Exponential Backoff

```python
import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,)
) -> T:
    """Execute function with exponential backoff retry."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()

        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries - 1:
                raise

            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
            await asyncio.sleep(delay)

    raise last_exception
```

### Usage in Service

```python
class EmbeddingService:
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic."""
        async def _generate():
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=text
            )
            return response.data[0].embedding

        try:
            return await retry_with_backoff(
                _generate,
                max_retries=3,
                base_delay=1.0,
                retryable_exceptions=(RateLimitError, APIConnectionError)
            )
        except Exception as e:
            raise ExternalServiceError(
                message=f"Embedding generation failed: {e}",
                service="openai",
                component="embedding"
            )
```

### Timeout Pattern

```python
import asyncio
from components.base.exceptions import TimeoutError

async def process_with_timeout(
    coro,
    timeout_seconds: float = 30.0,
    operation_name: str = "operation"
) -> Any:
    """Execute coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(
            message=f"{operation_name} timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            component="processing"
        )
```

---

## Error Response Format

### Standard Error Response

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str  # Error class name
    message: str  # Human-readable message
    component: str  # Component that raised error
    details: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Missing required field: title",
                "component": "classification",
                "details": {"field": "title", "constraint": "required"}
            }
        }
```

### SSE Error Event

```python
def format_error_event(error: ComponentError) -> str:
    """Format error for SSE stream."""
    return f"data: {json.dumps({
        'status': 'error',
        'agent': error.component,
        'error': error.to_dict(),
        'timestamp': datetime.now().isoformat()
    })}\n\n"
```

### Structured Logging

```python
import logging
import json

class StructuredLogger:
    """Logger with structured output for errors."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def error(self, message: str, error: Exception = None, **context):
        log_data = {
            "message": message,
            "level": "ERROR",
            **context
        }
        if error:
            log_data["error_type"] = type(error).__name__
            log_data["error_message"] = str(error)
            if hasattr(error, "to_dict"):
                log_data["error_details"] = error.to_dict()

        self.logger.error(json.dumps(log_data))
```
