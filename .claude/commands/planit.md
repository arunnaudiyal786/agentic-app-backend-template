# Plan: $ARGUMENTS

## Instructions

You are a senior engineer planning an implementation. Analyze the codebase and produce a detailed plan — **do not write or edit any code**.

### Steps

1. **Understand the task**: Parse the request in `$ARGUMENTS`. If anything is ambiguous or underspecified, use the `AskUserQuestion` tool to clarify before proceeding.
2. **Explore the codebase**: Read all relevant files, focusing on the modules, functions, and classes that would be affected. Identify entry points, dependencies, and existing patterns.
3. **Activate the virtual environment**: Use the `.venv` directory at the project root for any dependency checks (e.g., `source .venv/bin/activate && pip list`).
4. **Produce the plan**: Write a markdown file to `.claude/plans/IMPLEMENTATION_PLAN_<TITLE>.md` containing:

   - **Summary** — High-level approach and reasoning (2–4 sentences).
   - **Affected Files** — A table or list of every file/function/class to add or modify, with a one-line explanation per entry.
   - **Step-by-Step Changes** — Ordered list of discrete changes, each with:
     - The file and location (function, class, or line range).
     - What to do and why.
     - Any dependencies or ordering constraints.
   - **Design Considerations** — Patterns to follow, edge cases, and trade-offs.
   - **Risks & Open Questions** — Potential challenges and how to mitigate them.

### Constraints

- **No code changes.** Output is the plan file only.
- Replace `<TITLE>` in the filename with a short, snake_case descriptor of the task (e.g., `IMPLEMENTATION_PLAN_add_auth_middleware.md`).
- Prefer concrete file paths and function names over vague references.
