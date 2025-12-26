#!/usr/bin/env python3
"""
Backend Project Scaffolding Script

Generates a new FastAPI backend project with LangGraph multi-agent workflow
structure based on proven patterns.

Usage:
    python scaffold_backend.py <project-name> --path <output-dir>
    python scaffold_backend.py my-api --with-langgraph --with-faiss --with-sse
"""

import argparse
import os
from pathlib import Path
from typing import List

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

API_SERVER_TEMPLATE = '''"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import routers
# from components.embedding.router import router as embedding_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    print("Starting up...")
    yield
    print("Shutting down...")

app = FastAPI(
    title="{project_name}",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount component routers
# app.include_router(embedding_router, prefix="/v2")

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}
'''

CONFIG_TEMPLATE = '''"""
Centralized Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

class Config(BaseSettings):
    """Application configuration loaded from environment."""

    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Model Configuration
    model_name: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-large")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)

    # Processing Thresholds
    confidence_threshold: float = Field(default=0.7)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_config() -> Config:
    return Config()

config = get_config()
'''

BASE_COMPONENT_TEMPLATE = '''"""
Base Component Classes
"""
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
        pass

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        pass

    async def health_check(self) -> dict:
        return {"component": self.component_name, "status": "healthy"}
'''

BASE_EXCEPTIONS_TEMPLATE = '''"""
Exception Hierarchy
"""
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
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "details": self.details
        }

class ConfigurationError(ComponentError):
    """Configuration is invalid or missing."""
    pass

class ValidationError(ComponentError):
    """Input validation failed."""
    pass

class ProcessingError(ComponentError):
    """Processing failed during execution."""
    pass

class ExternalServiceError(ComponentError):
    """External service (OpenAI, etc.) failed."""

    def __init__(
        self,
        message: str,
        service: str,
        component: str = "unknown",
        details: Optional[Dict] = None
    ):
        super().__init__(message, component, details)
        self.service = service
'''

STATE_TEMPLATE = '''"""
Workflow State Definition
"""
from typing import TypedDict, Annotated, List, Dict, Literal, Any
import operator

class WorkflowState(TypedDict, total=False):
    """Workflow state with partial update support.

    total=False makes all fields optional for partial updates.
    """
    # Input fields
    input_id: str
    input_text: str
    metadata: Dict[str, Any]

    # Processing outputs (customize per workflow)
    result: Dict
    confidence: float

    # Accumulated messages
    messages: Annotated[List[Dict], operator.add]

    # Control fields
    status: Literal["pending", "processing", "success", "error", "completed"]
    current_agent: str
    error_message: str

RoutingDecision = Literal["next_node", "error_handler", "end"]
'''

WORKFLOW_TEMPLATE = '''"""
LangGraph Workflow Definition
"""
from langgraph.graph import StateGraph, END
from .state import WorkflowState, RoutingDecision

def route_by_status(state: WorkflowState) -> RoutingDecision:
    """Route based on processing status."""
    if state.get("status") == "error":
        return "error_handler"
    return "next_node"

async def example_node(state: WorkflowState) -> dict:
    """Example processing node."""
    return {
        "result": {"processed": True},
        "status": "success",
        "messages": [{"role": "example", "content": "Processed"}]
    }

async def error_handler_node(state: WorkflowState) -> dict:
    """Handle errors gracefully."""
    return {
        "result": {"error_handled": True},
        "status": "completed_with_errors",
        "messages": [{"role": "error_handler", "content": state.get("error_message", "Unknown")}]
    }

def create_workflow() -> StateGraph:
    """Create and compile the workflow."""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("example", example_node)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("example")

    # Add edges
    workflow.add_edge("example", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
'''

EXAMPLE_COMPONENT_AGENT = '''"""
LangGraph Node Function
"""
from typing import Dict, Any
from .service import ExampleService

_service: ExampleService | None = None

def get_service() -> ExampleService:
    global _service
    if _service is None:
        _service = ExampleService()
    return _service

async def example_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for example processing."""
    try:
        service = get_service()
        result = await service.process(state.get("input_text", ""))

        return {
            "result": result,
            "status": "success",
            "current_agent": "next",
            "messages": [{"role": "example", "content": "Processed successfully"}]
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "current_agent": "error_handler"
        }
'''

EXAMPLE_COMPONENT_SERVICE = '''"""
Business Logic Service
"""
from typing import Dict
from ..base.component import BaseComponent
from .models import ExampleRequest, ExampleResponse

class ExampleService(BaseComponent[ExampleRequest, ExampleResponse]):
    """Example service implementation."""

    @property
    def component_name(self) -> str:
        return "example"

    async def process(self, request: ExampleRequest) -> ExampleResponse:
        # Implement business logic here
        return ExampleResponse(
            result={"processed": True},
            message="Success"
        )
'''

EXAMPLE_COMPONENT_MODELS = '''"""
Pydantic Request/Response Models
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional

class ExampleRequest(BaseModel):
    """Request model."""
    text: str = Field(..., min_length=1, description="Input text")
    options: Optional[Dict[str, str]] = Field(default_factory=dict)

class ExampleResponse(BaseModel):
    """Response model."""
    result: Dict
    message: str
    confidence: float = Field(default=1.0)
'''

EXAMPLE_COMPONENT_ROUTER = '''"""
FastAPI Router
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from .service import ExampleService
from .models import ExampleRequest, ExampleResponse
from ..base.exceptions import ComponentError

router = APIRouter(prefix="/example", tags=["Example"])

_service: Optional[ExampleService] = None

def get_service() -> ExampleService:
    global _service
    if _service is None:
        _service = ExampleService()
    return _service

@router.post("/process", response_model=ExampleResponse)
async def process(
    request: ExampleRequest,
    service: ExampleService = Depends(get_service)
) -> ExampleResponse:
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

@router.get("/health")
async def health_check(service: ExampleService = Depends(get_service)):
    return await service.health_check()
'''

REQUIREMENTS_TEMPLATE = '''# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0

# LangChain & LangGraph
langchain>=0.1.0
langchain-openai>=0.0.5
langgraph>=0.0.20

# OpenAI
openai>=1.10.0

# Vector Store (optional)
faiss-cpu>=1.7.4
numpy>=1.24.0

# Utilities
httpx>=0.26.0
PyYAML>=6.0.1
'''

ENV_EXAMPLE_TEMPLATE = '''# API Keys
OPENAI_API_KEY=sk-...

# Model Configuration
MODEL_NAME=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
TEMPERATURE=0.0
MAX_TOKENS=4096

# Processing
CONFIDENCE_THRESHOLD=0.7

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
'''

GITIGNORE_TEMPLATE = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local

# Data
data/faiss_index/
output/

# Logs
*.log
logs/
'''

# ═══════════════════════════════════════════════════════════════════════════════
# SCAFFOLDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def write_file(path: Path, content: str) -> None:
    """Write content to file."""
    path.write_text(content)
    print(f"  Created: {path}")

def scaffold_project(
    project_name: str,
    output_path: Path,
    with_langgraph: bool = True,
    with_faiss: bool = False,
    with_sse: bool = False
) -> None:
    """Scaffold a new backend project."""
    project_dir = output_path / project_name

    print(f"\nScaffolding project: {project_name}")
    print(f"Location: {project_dir}")
    print(f"Options: langgraph={with_langgraph}, faiss={with_faiss}, sse={with_sse}\n")

    # Create directory structure
    directories = [
        project_dir,
        project_dir / "config",
        project_dir / "components" / "base",
        project_dir / "components" / "example",
        project_dir / "src" / "orchestrator",
        project_dir / "src" / "models",
        project_dir / "src" / "utils",
        project_dir / "data",
        project_dir / "input",
        project_dir / "output",
        project_dir / "scripts",
    ]

    for directory in directories:
        create_directory(directory)

    # Create __init__.py files
    init_paths = [
        project_dir / "components" / "__init__.py",
        project_dir / "components" / "base" / "__init__.py",
        project_dir / "components" / "example" / "__init__.py",
        project_dir / "src" / "__init__.py",
        project_dir / "src" / "orchestrator" / "__init__.py",
        project_dir / "src" / "models" / "__init__.py",
        project_dir / "src" / "utils" / "__init__.py",
    ]
    for init_path in init_paths:
        write_file(init_path, "")

    # Write main files
    write_file(
        project_dir / "api_server.py",
        API_SERVER_TEMPLATE.format(project_name=project_name)
    )
    write_file(project_dir / "config" / "config.py", CONFIG_TEMPLATE)
    write_file(project_dir / "components" / "base" / "component.py", BASE_COMPONENT_TEMPLATE)
    write_file(project_dir / "components" / "base" / "exceptions.py", BASE_EXCEPTIONS_TEMPLATE)

    # Example component
    write_file(project_dir / "components" / "example" / "agent.py", EXAMPLE_COMPONENT_AGENT)
    write_file(project_dir / "components" / "example" / "service.py", EXAMPLE_COMPONENT_SERVICE)
    write_file(project_dir / "components" / "example" / "models.py", EXAMPLE_COMPONENT_MODELS)
    write_file(project_dir / "components" / "example" / "router.py", EXAMPLE_COMPONENT_ROUTER)

    # LangGraph workflow
    if with_langgraph:
        write_file(project_dir / "src" / "orchestrator" / "state.py", STATE_TEMPLATE)
        write_file(project_dir / "src" / "orchestrator" / "workflow.py", WORKFLOW_TEMPLATE)

    # Configuration files
    write_file(project_dir / "requirements.txt", REQUIREMENTS_TEMPLATE)
    write_file(project_dir / ".env.example", ENV_EXAMPLE_TEMPLATE)
    write_file(project_dir / ".gitignore", GITIGNORE_TEMPLATE)

    print(f"\nProject scaffolded successfully!")
    print(f"\nNext steps:")
    print(f"  1. cd {project_dir}")
    print(f"  2. python -m venv .venv && source .venv/bin/activate")
    print(f"  3. pip install -r requirements.txt")
    print(f"  4. cp .env.example .env  # Add your API keys")
    print(f"  5. uvicorn api_server:app --reload")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new FastAPI backend project"
    )
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--with-langgraph",
        action="store_true",
        default=True,
        help="Include LangGraph workflow (default: True)"
    )
    parser.add_argument(
        "--with-faiss",
        action="store_true",
        default=False,
        help="Include FAISS vector store setup"
    )
    parser.add_argument(
        "--with-sse",
        action="store_true",
        default=False,
        help="Include SSE streaming endpoints"
    )

    args = parser.parse_args()

    scaffold_project(
        project_name=args.project_name,
        output_path=args.path,
        with_langgraph=args.with_langgraph,
        with_faiss=args.with_faiss,
        with_sse=args.with_sse
    )


if __name__ == "__main__":
    main()
