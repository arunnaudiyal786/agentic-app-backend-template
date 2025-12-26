# Pydantic Models Patterns

## Table of Contents
1. [Model Basics](#model-basics)
2. [Request/Response Models](#requestresponse-models)
3. [Nested Models](#nested-models)
4. [Validation Patterns](#validation-patterns)
5. [Field Configuration](#field-configuration)
6. [Model Hierarchy](#model-hierarchy)

---

## Model Basics

### Simple Model Definition

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class TicketInput(BaseModel):
    """Input model for ticket processing."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=10)
    priority: str = Field(default="medium")
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TICK-001",
                "title": "Login page not loading",
                "description": "Users report the login page shows a blank screen",
                "priority": "high"
            }
        }
```

### Model with Computed Fields

```python
from pydantic import BaseModel, computed_field

class TicketWithComputed(BaseModel):
    title: str
    description: str

    @computed_field
    @property
    def combined_text(self) -> str:
        """Computed field for embedding generation."""
        return f"{self.title}\n\n{self.description}"

    @computed_field
    @property
    def text_length(self) -> int:
        return len(self.title) + len(self.description)
```

---

## Request/Response Models

### Standard Request Model

```python
class ProcessRequest(BaseModel):
    """Request model for processing endpoint."""
    query: str = Field(..., min_length=1, description="Query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    domain_filter: Optional[str] = Field(default=None, description="Filter by domain")
    include_metadata: bool = Field(default=True)

    class Config:
        # Validate on assignment
        validate_assignment = True
        # Allow extra fields to be ignored
        extra = "ignore"
```

### Standard Response Model

```python
class ProcessResponse(BaseModel):
    """Response model for processing endpoint."""
    results: List[Dict]
    total_count: int
    query: str
    processing_time_ms: int
    metadata: Optional[Dict] = None

    class Config:
        # Ensure JSON serialization works
        from_attributes = True
```

### Paginated Response

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

    @computed_field
    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size
```

---

## Nested Models

### Hierarchical Model Structure

```python
class ResolutionStep(BaseModel):
    """Single step in resolution plan."""
    step_number: int = Field(..., ge=1)
    action: str
    expected_outcome: str
    tools_required: List[str] = Field(default_factory=list)
    estimated_duration: Optional[str] = None

class TicketReference(BaseModel):
    """Reference to a historical ticket."""
    ticket_id: str
    title: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    relevant_section: Optional[str] = None

class ResolutionPlan(BaseModel):
    """Complete resolution plan."""
    summary: str
    test_plan: List[ResolutionStep]
    considerations: List[str] = Field(default_factory=list)
    references: List[TicketReference] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)

class FinalOutput(BaseModel):
    """Final processing output."""
    ticket_id: str
    classification: Dict[str, str]
    labels: Dict[str, List[str]]
    similar_tickets: List[TicketReference]
    resolution: ResolutionPlan
    processing_metadata: Dict
```

---

## Validation Patterns

### Custom Validators

```python
from pydantic import BaseModel, field_validator, model_validator

class TicketInput(BaseModel):
    ticket_id: str
    priority: str
    title: str
    description: str

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = ["low", "medium", "high", "critical"]
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of: {allowed}")
        return v.lower()

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, v: str) -> str:
        if not v.startswith(("TICK-", "INC-", "REQ-")):
            raise ValueError("Ticket ID must start with TICK-, INC-, or REQ-")
        return v.upper()

    @model_validator(mode="after")
    def validate_content(self) -> "TicketInput":
        """Cross-field validation."""
        if self.priority == "critical" and len(self.description) < 50:
            raise ValueError("Critical tickets require detailed descriptions")
        return self
```

### Conditional Validation

```python
from pydantic import BaseModel, model_validator
from typing import Optional

class ConditionalRequest(BaseModel):
    mode: str
    query: Optional[str] = None
    file_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "ConditionalRequest":
        if self.mode == "search" and not self.query:
            raise ValueError("Query required for search mode")
        if self.mode == "file" and not self.file_path:
            raise ValueError("File path required for file mode")
        return self
```

---

## Field Configuration

### Field Options

```python
from pydantic import BaseModel, Field
from typing import List

class DetailedModel(BaseModel):
    # Required field with description
    name: str = Field(..., description="Item name")

    # Optional with default
    count: int = Field(default=0, description="Item count")

    # With constraints
    score: float = Field(default=0.0, ge=0.0, le=1.0)

    # With length constraints
    tags: List[str] = Field(default_factory=list, max_length=10)

    # With pattern validation
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    # Deprecated field
    old_field: str = Field(default="", deprecated=True)

    # Field alias (for JSON serialization)
    internal_id: str = Field(..., alias="id")

    # Exclude from serialization
    secret: str = Field(..., exclude=True)
```

### Default Factories

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

class AutoFieldsModel(BaseModel):
    # Auto-generate ID
    id: str = Field(default_factory=lambda: str(uuid4()))

    # Auto-timestamp
    created_at: datetime = Field(default_factory=datetime.now)

    # Empty containers
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
```

---

## Model Hierarchy

### Inheritance Pattern

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class BaseTicket(BaseModel):
    """Base ticket with common fields."""
    ticket_id: str
    title: str
    description: str
    created_at: datetime = Field(default_factory=datetime.now)

class IncomingTicket(BaseTicket):
    """Ticket as received from input."""
    priority: str = Field(default="medium")
    source: Optional[str] = None

class ProcessedTicket(BaseTicket):
    """Ticket after processing."""
    classified_domain: str
    confidence: float
    labels: List[str]
    similar_ticket_ids: List[str]

class FinalTicket(ProcessedTicket):
    """Complete ticket with resolution."""
    resolution_plan: Dict
    processing_time_ms: int
    processed_at: datetime = Field(default_factory=datetime.now)
```

### Mixin Pattern

```python
class TimestampMixin(BaseModel):
    """Add timestamps to any model."""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class MetadataMixin(BaseModel):
    """Add metadata support."""
    metadata: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class AuditableTicket(BaseTicket, TimestampMixin, MetadataMixin):
    """Ticket with timestamps and metadata."""
    pass
```

### Union Types for Polymorphism

```python
from typing import Union, Literal
from pydantic import BaseModel

class TextInput(BaseModel):
    type: Literal["text"] = "text"
    content: str

class FileInput(BaseModel):
    type: Literal["file"] = "file"
    file_path: str
    file_type: str

class URLInput(BaseModel):
    type: Literal["url"] = "url"
    url: str

# Union type for multiple input formats
InputType = Union[TextInput, FileInput, URLInput]

class ProcessRequest(BaseModel):
    input: InputType
    options: Dict[str, str] = Field(default_factory=dict)
```

---

## Serialization

### JSON Serialization Options

```python
class SerializableModel(BaseModel):
    name: str
    created_at: datetime
    data: Dict

    class Config:
        # Use enum values instead of names
        use_enum_values = True
        # Serialize datetime as ISO format
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Serialization methods
model = SerializableModel(...)

# To dict
model.model_dump()

# To dict excluding None values
model.model_dump(exclude_none=True)

# To JSON string
model.model_dump_json()

# From dict
SerializableModel.model_validate(data_dict)

# From JSON string
SerializableModel.model_validate_json(json_string)
```
