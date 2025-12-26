# State Management Patterns

## Table of Contents
1. [TypedDict State](#typeddict-state)
2. [Partial Updates](#partial-updates)
3. [Message Accumulation](#message-accumulation)
4. [State Categories](#state-categories)
5. [Routing Decisions](#routing-decisions)
6. [Best Practices](#best-practices)

---

## TypedDict State

### Basic State Definition

```python
from typing import TypedDict, Annotated, List, Dict, Optional, Literal, Any
import operator

class WorkflowState(TypedDict, total=False):
    """Workflow state with partial update support.

    Using total=False makes all fields optional, allowing nodes
    to return partial state updates that get merged automatically.
    """
    # Input fields
    input_id: str
    input_text: str
    input_metadata: Dict[str, Any]

    # Processing outputs
    classification_result: Dict
    retrieval_results: List[Dict]
    resolution_plan: Dict

    # Accumulated messages
    messages: Annotated[List[Dict], operator.add]

    # Control fields
    status: str
    current_agent: str
    error_message: str
```

### Why `total=False`?

```python
# With total=True (default) - all fields required
class StrictState(TypedDict):
    field_a: str
    field_b: int

# Node must return ALL fields - problematic!
def node(state: StrictState) -> StrictState:
    return {
        "field_a": "updated",
        "field_b": state["field_b"]  # Must include unchanged fields
    }

# With total=False - all fields optional
class FlexibleState(TypedDict, total=False):
    field_a: str
    field_b: int

# Node returns only changed fields - clean!
def node(state: FlexibleState) -> Dict:
    return {"field_a": "updated"}  # field_b untouched
```

---

## Partial Updates

### How Partial Updates Work

```python
# Initial state
state = {
    "input_id": "123",
    "input_text": "Process this",
    "status": "pending"
}

# Agent A returns partial update
agent_a_output = {
    "classification": {"domain": "A"},
    "status": "classified"
}

# State after Agent A (automatic merge)
state = {
    "input_id": "123",           # Unchanged
    "input_text": "Process this", # Unchanged
    "status": "classified",       # Updated
    "classification": {"domain": "A"}  # Added
}

# Agent B returns partial update
agent_b_output = {
    "retrieval_results": [...],
    "status": "retrieved"
}

# State after Agent B
state = {
    "input_id": "123",
    "input_text": "Process this",
    "status": "retrieved",        # Updated
    "classification": {"domain": "A"},  # Unchanged
    "retrieval_results": [...]    # Added
}
```

### Node Return Pattern

```python
async def processing_node(state: WorkflowState) -> Dict:
    """Always return a partial dict, not the full state."""
    result = await process(state["input_text"])

    # CORRECT: Return only changed fields
    return {
        "processing_result": result,
        "status": "processed",
        "current_agent": "next_node"
    }

    # WRONG: Don't return unchanged fields
    # return {
    #     "input_id": state["input_id"],  # Unnecessary
    #     "input_text": state["input_text"],  # Unnecessary
    #     "processing_result": result,
    #     "status": "processed"
    # }
```

---

## Message Accumulation

### Using `operator.add` Reducer

```python
from typing import Annotated
import operator

class WorkflowState(TypedDict, total=False):
    # This field accumulates across nodes
    messages: Annotated[List[Dict], operator.add]

    # These fields replace (default behavior)
    status: str
    result: Dict

# Agent A
def agent_a(state) -> Dict:
    return {
        "status": "a_done",
        "messages": [{"role": "agent_a", "content": "Processed A"}]
    }

# Agent B
def agent_b(state) -> Dict:
    return {
        "status": "b_done",
        "messages": [{"role": "agent_b", "content": "Processed B"}]
    }

# After both agents, messages accumulates:
# messages = [
#     {"role": "agent_a", "content": "Processed A"},
#     {"role": "agent_b", "content": "Processed B"}
# ]
# status = "b_done" (replaced)
```

### Message Structure

```python
# Standard message format
message = {
    "role": "agent_name",       # Which agent produced this
    "content": "Human-readable message",
    "timestamp": "2024-01-01T12:00:00Z",
    "data": {...}               # Optional structured data
}

# Tool call message
tool_message = {
    "role": "tool",
    "tool_name": "search_similar",
    "tool_input": {"query": "..."},
    "tool_output": {...},
    "timestamp": "..."
}
```

---

## State Categories

### Recommended State Organization

```python
class TicketWorkflowState(TypedDict, total=False):
    """Organized state for ticket processing workflow."""

    # ═══════════════════════════════════════════════════
    # INPUT FIELDS - Set once at workflow start
    # ═══════════════════════════════════════════════════
    ticket_id: str
    title: str
    description: str
    priority: str
    metadata: Dict[str, Any]

    # ═══════════════════════════════════════════════════
    # CLASSIFICATION OUTPUT - Set by classification agent
    # ═══════════════════════════════════════════════════
    classified_domain: str
    domain_confidence: float
    domain_reasoning: str
    extracted_keywords: List[str]

    # ═══════════════════════════════════════════════════
    # RETRIEVAL OUTPUT - Set by retrieval agent
    # ═══════════════════════════════════════════════════
    similar_tickets: List[Dict]
    similarity_scores: List[float]
    retrieval_metadata: Dict

    # ═══════════════════════════════════════════════════
    # LABELING OUTPUT - Set by labeling agent
    # ═══════════════════════════════════════════════════
    category_labels: List[str]
    business_labels: List[str]
    technical_labels: List[str]
    label_confidences: Dict[str, float]

    # ═══════════════════════════════════════════════════
    # RESOLUTION OUTPUT - Set by resolution agent
    # ═══════════════════════════════════════════════════
    resolution_plan: Dict
    resolution_confidence: float
    test_steps: List[Dict]
    references: List[str]

    # ═══════════════════════════════════════════════════
    # WORKFLOW CONTROL - Updated throughout
    # ═══════════════════════════════════════════════════
    status: Literal["pending", "processing", "success", "error", "completed"]
    current_agent: str
    error_message: str
    processing_time_ms: int

    # ═══════════════════════════════════════════════════
    # ACCUMULATED FIELDS
    # ═══════════════════════════════════════════════════
    messages: Annotated[List[Dict], operator.add]
    tool_calls: Annotated[List[Dict], operator.add]
```

---

## Routing Decisions

### Routing Type Definition

```python
from typing import Literal

# Define all possible routing destinations
RoutingDecision = Literal[
    "classification",
    "retrieval",
    "labeling",
    "resolution",
    "error_handler",
    "end"
]
```

### Status-Based Routing Function

```python
def route_after_classification(state: WorkflowState) -> RoutingDecision:
    """Route based on classification status."""
    status = state.get("status")
    confidence = state.get("domain_confidence", 0.0)

    # Error handling
    if status == "error":
        return "error_handler"

    # Low confidence - needs review
    if confidence < 0.5:
        return "error_handler"

    # Success - continue pipeline
    return "retrieval"

def route_after_retrieval(state: WorkflowState) -> RoutingDecision:
    """Route based on retrieval results."""
    similar_tickets = state.get("similar_tickets", [])

    if state.get("status") == "error":
        return "error_handler"

    # No similar tickets found
    if len(similar_tickets) == 0:
        return "error_handler"  # Or novelty detection

    return "labeling"
```

### Wiring Routes in Workflow

```python
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("classification", classification_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("labeling", labeling_node)
workflow.add_node("resolution", resolution_node)
workflow.add_node("error_handler", error_handler_node)

# Add conditional edges
workflow.add_conditional_edges(
    "classification",
    route_after_classification,
    {
        "retrieval": "retrieval",
        "error_handler": "error_handler"
    }
)

workflow.add_conditional_edges(
    "retrieval",
    route_after_retrieval,
    {
        "labeling": "labeling",
        "error_handler": "error_handler"
    }
)
```

---

## Best Practices

### 1. Always Use `total=False`

```python
# DO
class State(TypedDict, total=False):
    ...

# DON'T
class State(TypedDict):  # total=True by default
    ...
```

### 2. Return Minimal Updates

```python
# DO
return {"new_field": value, "status": "done"}

# DON'T
return {**state, "new_field": value}
```

### 3. Use Annotated for Accumulation

```python
# DO - for lists that should accumulate
messages: Annotated[List[Dict], operator.add]

# DON'T - will replace instead of accumulate
messages: List[Dict]
```

### 4. Include Status in Every Return

```python
# DO
return {
    "result": data,
    "status": "success",  # Always include
    "current_agent": "next"
}
```

### 5. Handle Missing Fields Gracefully

```python
# DO
value = state.get("optional_field", default_value)

# DON'T
value = state["optional_field"]  # Might KeyError
```

### 6. Document State Fields

```python
class WorkflowState(TypedDict, total=False):
    """Workflow state for ticket processing.

    Input Fields:
        ticket_id: Unique identifier
        title: Ticket title

    Output Fields:
        result: Processing result
        confidence: Result confidence score
    """
    ticket_id: str
    title: str
    result: Dict
    confidence: float
```
