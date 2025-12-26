---
name: backend-design
description: |
  Design and scaffold Python FastAPI backends with multi-agent LangGraph workflows, FAISS vector stores, and component-based architecture. Use when:
  (1) Creating new FastAPI backend projects with async patterns
  (2) Designing LangGraph multi-agent orchestration workflows
  (3) Building component-based agent architectures (tools + nodes + services)
  (4) Implementing FAISS vector similarity search systems
  (5) Setting up SSE streaming endpoints for real-time updates
  (6) Structuring Pydantic models for API contracts
  (7) Creating configuration systems with environment variables and YAML
---

# Backend Design Skill

Design production-grade Python FastAPI backends with LangGraph multi-agent workflows.

## Quick Reference

| Pattern | Reference File |
|---------|----------------|
| FastAPI structure, routers, SSE | [references/fastapi-patterns.md](references/fastapi-patterns.md) |
| LangGraph workflows, routing | [references/langgraph-patterns.md](references/langgraph-patterns.md) |
| Component architecture | [references/component-structure.md](references/component-structure.md) |
| TypedDict state management | [references/state-management.md](references/state-management.md) |
| Pydantic models & validation | [references/pydantic-models.md](references/pydantic-models.md) |
| Config & environment vars | [references/configuration.md](references/configuration.md) |
| Error handling hierarchy | [references/error-handling.md](references/error-handling.md) |

## Core Architecture

```
project/
├── api_server.py              # FastAPI app entry point
├── main.py                    # CLI entry point
├── config/
│   ├── config.py              # Centralized configuration
│   └── schema_config.yaml     # Data schema definitions
├── components/                # Self-contained agent components
│   ├── base/                  # Base classes and exceptions
│   └── {component}/           # Each component folder
│       ├── agent.py           # LangGraph node function
│       ├── tools.py           # LangChain @tool functions
│       ├── service.py         # Business logic
│       ├── models.py          # Pydantic request/response
│       ├── router.py          # FastAPI endpoints
│       └── prompts.py         # LLM prompt templates
├── src/
│   ├── orchestrator/          # LangGraph workflow
│   │   ├── state.py           # TypedDict state definition
│   │   └── workflow.py        # StateGraph definition
│   ├── prompts/               # Shared prompt templates
│   ├── models/                # Shared Pydantic models
│   ├── vectorstore/           # FAISS management
│   └── utils/                 # Shared utilities
├── scripts/                   # Setup and maintenance scripts
├── data/                      # Data files and indexes
├── input/                     # Input files
└── output/                    # Processing results
```

## Workflow Design Pattern

Sequential multi-agent pipeline with conditional routing:

```python
# Each agent returns PARTIAL state (not complete state)
async def agent_node(state: WorkflowState) -> Dict:
    result = await process(state)
    return {
        "output_field": result,
        "status": "success",
        "current_agent": "next_agent"
    }

# Routing based on status
def route_after_agent(state: WorkflowState) -> str:
    if state.get("status") == "error":
        return "error_handler"
    return "next_node"
```

## Key Patterns Summary

### 1. Component Structure
Each component is self-contained with: `agent.py` (LangGraph node) + `tools.py` (LangChain tools) + `service.py` (business logic) + `router.py` (HTTP endpoints)

### 2. State Accumulation
Use `TypedDict` with `total=False` for partial updates. Use `Annotated[List, operator.add]` for list accumulation.

### 3. Async-First
All I/O operations async: `await service.process()`, `asyncio.gather()` for parallel execution.

### 4. Dependency Injection
Singleton services via FastAPI `Depends()` with lazy initialization.

### 5. SSE Streaming
`StreamingResponse` with async generators for real-time updates.

### 6. Configuration Externalization
Environment variables loaded via `python-dotenv`, YAML for schemas.

## Scaffolding New Projects

Run the scaffold script to generate a new backend project:

```bash
python .claude/skills/backend-design/scripts/scaffold_backend.py <project-name> --path <output-dir>
```

Options:
- `--with-langgraph` - Include LangGraph workflow boilerplate
- `--with-faiss` - Include FAISS vector store setup
- `--with-sse` - Include SSE streaming endpoints

## Implementation Checklist

When designing a new backend:

1. [ ] Define workflow state in `src/orchestrator/state.py`
2. [ ] Create component structure for each agent
3. [ ] Implement tools with `@tool` decorator
4. [ ] Wire up LangGraph StateGraph in `workflow.py`
5. [ ] Add FastAPI routers per component
6. [ ] Configure SSE streaming if needed
7. [ ] Set up configuration system
8. [ ] Add error handling hierarchy
9. [ ] Create setup/maintenance scripts

## Pattern Selection Guide

| Need | Pattern | Reference |
|------|---------|-----------|
| Sequential agents | LangGraph StateGraph | langgraph-patterns.md |
| Parallel classifiers | `asyncio.gather()` | component-structure.md |
| Real-time updates | SSE streaming | fastapi-patterns.md |
| Similarity search | FAISS + hybrid scoring | component-structure.md |
| LLM integration | LangChain tools + prompts | component-structure.md |
| API contracts | Pydantic BaseModel | pydantic-models.md |
| Multi-env config | BaseSettings + .env | configuration.md |
