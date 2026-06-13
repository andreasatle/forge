# Forge — Claude Code Guide

## What Forge Is
A scheduler-centric agentic framework. The scheduler owns the DAG. 
Agents are opaque functions `AgentRequest → AgentResponse`. 
The DAG is the queue.

## Architecture
The core loop:
1. Scheduler finds READY nodes → dispatches up to `max_concurrency`
2. Agents return `AgentResponse` (with optional `follow_up` nodes)
3. Follow-ups are added to the DAG as PENDING nodes
4. On idle → global planner re-evaluates northstar → emits more work or terminates

**Boundaries — never cross these:**
- Scheduler does not import from agents
- Agents do not import from the scheduler
- Core models live in `forge/core/models.py` — do not redefine elsewhere

## Before Starting Any Task
Read docs/AGENT.md first — it is the authoritative bootstrap brief for Forge architecture, invariants, and design preferences.

## Code Rules
- **Pydantic** for data that crosses boundaries or needs validation
- **Dataclass** for internal config and lightweight containers
- **Enums** for every role, status, type, source — no raw string sentinels
- **Frozen models** — use `model_copy(update={...})` never mutate in place
- **Single edit rule** — always add imports and the code that uses them in one edit, never separately, or ruff will strip them
- Fail loudly on missing config — no silent defaults
- One parser per agent type — pure functions, no inline parsing
- **Fix at the right abstraction level** — if a fix applies to a class of problems, implement it at the base level, not in each instance. Example: retry logic belongs in `run_agent`, not in each agent. Tool loops belong in `run_agent`, not in `work_agent`. Before fixing something specific, ask: "where is the right place for this fix to live?"

## Adapter / Language Plugin Boundary
- **Adapters** (`adapters/*.yaml`) are language-agnostic — they own output
  format rules, WorkOutput structure, and tool-use instructions only.
  Never put language-specific content (file paths, package managers, 
  test frameworks, project manifest names) in an adapter.
- **Language plugins** (`languages/*.yaml`) own everything language-specific:
  project structure, package manager commands, test commands, import 
  conventions, manifest format (pyproject.toml, Cargo.toml, build.zig.zon),
  and the WorkOutput example for that language.
- When adding guidance, ask: "would this be wrong for a different language?"
  If yes → language plugin. If no → adapter.
  
## Testing
- Every module has a corresponding test file mirroring source structure
- Scheduler tests use mock runners returning canned `AgentResponse`s
- No test requires a running model instance
- One logical assertion per test where possible
- Always run pytest after all changes and fix any failures before reporting done

## What Claude Code May Do
- Edit existing files
- Create new files
- Run `pytest`
- Run `uv run forge`

## What Claude Code Must Not Do
- Modify `CLAUDE.md`
- Delete test files
- Change enum values without explicit instruction
- Add dependencies without explicit instruction
- Hardcode model names or API keys
- Add abstraction before the need is proven
