# LangGraph Workflow Patterns

## Table of Contents
1. [StateGraph Basics](#stategraph-basics)
2. [Node Functions](#node-functions)
3. [Conditional Routing](#conditional-routing)
4. [Error Handling](#error-handling)
5. [Parallel Execution](#parallel-execution)
6. [Streaming Updates](#streaming-updates)

---

## StateGraph Basics

### Workflow Definition

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Literal
import operator

class WorkflowState(TypedDict, total=False):
    """Workflow state with partial updates.

    total=False means all fields are optional,
    allowing nodes to return partial state updates.
    """
    # Input fields
    input_id: str
    input_text: str

    # Processing fields (accumulated via operator.add)
    messages: Annotated[List[Dict], operator.add]

    # Output fields
    result: Dict
    confidence: float

    # Control fields
    status: str
    current_agent: str
    error_message: str

# Define routing decisions as Literal type
RoutingDecision = Literal["next_node", "error_handler", "end"]

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("agent_a", agent_a_node)
    workflow.add_node("agent_b", agent_b_node)
    workflow.add_node("agent_c", agent_c_node)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("agent_a")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent_a",
        route_after_agent_a,
        {
            "agent_b": "agent_b",
            "error_handler": "error_handler"
        }
    )

    workflow.add_conditional_edges(
        "agent_b",
        route_after_agent_b,
        {
            "agent_c": "agent_c",
            "error_handler": "error_handler"
        }
    )

    # Terminal edges
    workflow.add_edge("agent_c", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
```

---

## Node Functions

### Basic Node Pattern

```python
async def agent_node(state: WorkflowState) -> Dict:
    """Node function that processes state and returns PARTIAL updates.

    CRITICAL: Return only the fields that changed, not the entire state.
    LangGraph merges the returned dict into the existing state.
    """
    try:
        # Extract input from state
        input_text = state.get("input_text", "")

        # Process
        result = await process_input(input_text)

        # Return PARTIAL state update
        return {
            "result": result,
            "confidence": 0.95,
            "status": "success",
            "current_agent": "agent_b",
            "messages": [{"role": "agent", "content": f"Processed: {result}"}]
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "current_agent": "error_handler"
        }
```

### Node with LangChain Tool Integration

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def classify_input(text: str) -> Dict:
    """Classify the input text into categories."""
    # Tool implementation
    return {"category": "A", "confidence": 0.9}

async def classification_node(state: WorkflowState) -> Dict:
    """Node that uses LangChain tools."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools([classify_input])

    messages = [
        {"role": "system", "content": "Classify the following input."},
        {"role": "user", "content": state["input_text"]}
    ]

    response = await llm_with_tools.ainvoke(messages)

    # Extract tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        result = classify_input.invoke(tool_call["args"])
        return {
            "result": result,
            "status": "success",
            "messages": [{"role": "assistant", "content": str(result)}]
        }

    return {"status": "error", "error_message": "No tool call made"}
```

---

## Conditional Routing

### Status-Based Routing

```python
def route_after_agent(state: WorkflowState) -> str:
    """Route based on processing status."""
    status = state.get("status", "")

    if status == "error":
        return "error_handler"

    if status == "skip":
        return "end"

    return "next_node"
```

### Confidence-Based Routing

```python
def route_by_confidence(state: WorkflowState) -> str:
    """Route based on confidence threshold."""
    confidence = state.get("confidence", 0.0)

    if confidence < 0.5:
        return "low_confidence_handler"
    elif confidence < 0.8:
        return "medium_confidence_path"
    else:
        return "high_confidence_path"
```

### Multi-Condition Routing

```python
def complex_router(state: WorkflowState) -> str:
    """Route based on multiple conditions."""
    status = state.get("status")
    result_type = state.get("result", {}).get("type")

    if status == "error":
        return "error_handler"

    routing_map = {
        "classification": "labeling_node",
        "retrieval": "resolution_node",
        "novelty": "escalation_node"
    }

    return routing_map.get(result_type, "default_node")
```

---

## Error Handling

### Error Handler Node

```python
async def error_handler_node(state: WorkflowState) -> Dict:
    """Graceful error handling with fallback output."""
    error_message = state.get("error_message", "Unknown error")
    current_agent = state.get("current_agent", "unknown")

    # Log error
    logger.error(f"Error in {current_agent}: {error_message}")

    # Generate fallback response
    fallback_result = {
        "status": "error",
        "error_details": {
            "agent": current_agent,
            "message": error_message
        },
        "fallback_output": generate_fallback_output(state)
    }

    return {
        "result": fallback_result,
        "status": "completed_with_errors",
        "messages": [{
            "role": "system",
            "content": f"Error handled: {error_message}"
        }]
    }

def generate_fallback_output(state: WorkflowState) -> Dict:
    """Generate sensible fallback when processing fails."""
    return {
        "recommendation": "Manual review required",
        "partial_results": state.get("result", {}),
        "confidence": 0.0
    }
```

### Try-Catch Pattern in Nodes

```python
async def resilient_node(state: WorkflowState) -> Dict:
    """Node with comprehensive error handling."""
    try:
        # Validate input
        if not state.get("input_text"):
            raise ValidationError("Missing input_text")

        # Process with timeout
        result = await asyncio.wait_for(
            process_input(state["input_text"]),
            timeout=30.0
        )

        return {"result": result, "status": "success"}

    except ValidationError as e:
        return {"status": "error", "error_message": f"Validation: {e}"}
    except asyncio.TimeoutError:
        return {"status": "error", "error_message": "Processing timeout"}
    except Exception as e:
        return {"status": "error", "error_message": f"Unexpected: {e}"}
```

---

## Parallel Execution

### Parallel Node Pattern

```python
async def parallel_classification_node(state: WorkflowState) -> Dict:
    """Run multiple classifiers in parallel."""
    input_text = state["input_text"]

    # Create parallel tasks
    tasks = [
        classify_domain_a(input_text),
        classify_domain_b(input_text),
        classify_domain_c(input_text)
    ]

    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    classifications = {}
    errors = []

    for domain, result in zip(["A", "B", "C"], results):
        if isinstance(result, Exception):
            errors.append(f"{domain}: {result}")
        else:
            classifications[domain] = result

    return {
        "classifications": classifications,
        "status": "success" if not errors else "partial_success",
        "error_message": "; ".join(errors) if errors else None
    }
```

---

## Streaming Updates

### Streaming Workflow Execution

```python
async def stream_workflow(initial_state: Dict) -> AsyncGenerator[Dict, None]:
    """Stream workflow updates as they happen."""
    workflow = create_workflow()

    # Use astream for streaming execution
    async for event in workflow.astream(initial_state):
        # event contains the node name and output
        for node_name, node_output in event.items():
            yield {
                "agent": node_name,
                "status": node_output.get("status", "processing"),
                "data": node_output,
                "timestamp": datetime.now().isoformat()
            }
```

### Integration with FastAPI SSE

```python
@app.post("/api/process")
async def process_request(request: ProcessRequest):
    async def stream_updates():
        initial_state = {
            "input_id": request.id,
            "input_text": request.text,
            "messages": []
        }

        async for update in stream_workflow(initial_state):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        stream_updates(),
        media_type="text/event-stream"
    )
```
