# Forge — Development Guide for Claude Code

## What Forge Is

Forge is a scheduler-centric agentic framework. The scheduler is the center of
the system. Agents are opaque workers. The DAG is the queue.

The core loop:

```
SchedulerState (DAG + State)
  ↓
Scheduler emits READY AgentRequests to Workers
  ↓
Workers return AgentResponses (with optional follow_up nodes)
  ↓
Integrator applies deltas to State
  ↓
DAG drains → Planner produces next subgraph → repeat
```

Do not conflate the scheduler with the agents. The scheduler does not know what
agents do internally. Agents do not know about the DAG.

---

## Hooks (Automatic)

The following run automatically — do not invoke them manually:

- **On session start**: codebase index is refreshed (`scripts/index_codebase.py`)
- **After any file write/edit**: Ruff (fix + format + check), Pyright, and index refresh

If Ruff or Pyright report errors after your edit, fix them before proceeding.

---

## Principles

### Simplicity first
Prefer the simplest solution that is correct. Do not add abstraction until the
need is proven. Do not add configuration until a hard-coded value causes a
problem.

### Enums and constants over string literals
Every role, status, type, and source must be an enum. No raw strings as
sentinel values. IDE support and typo safety matter.

### No underspecified configs
Every configuration value must have an explicit source. No silent defaults that
hide missing configuration. Fail loudly on startup if required config is
absent.

### Single mutation boundary
`StateService` is the only place State is mutated. Nothing else calls
`apply_delta` directly. No exceptions.

### Single parsing pipeline
Each agent type has exactly one parser for its output. Parsers are pure
functions. No inline parsing in agent logic.

### Immutability for shared data structures
`AgentRequest`, `AgentResponse`, `DAGNode`, and `SchedulerState` are frozen
Pydantic models. Use `model_copy(update={...})` for transitions. Never mutate
in place.

### Clean up your own mess
If you add a file, add its tests. If you add a dependency, document why. If
you refactor a module, remove the old version. No orphaned code.

### Define success criteria upfront
Before implementing any agent or scheduler feature, identify what a passing
test looks like. Do not write implementation before writing the test contract.

---

## Module Ownership

Each module has a single responsibility. Do not reach across boundaries.

```
forge/
  scheduler/      — DAG management, node transitions, emit/collect loop
  agents/         — agent implementations (planner, worker, integrator)
  models/         — ModelClient, Role, fallback chains
  state/          — State, DeltaState, StateService, StateStore
  adapters/       — project-type-specific behaviour (coding, document, audit)
  parsers/        — pure output parsers, one per agent type
  prompts/        — pure prompt render functions
  northstar/      — NorthStar loading and representation
  tools/          — research tools available to agents
  plugins/        — language plugins (Python AST etc.)
```

**The scheduler does not import from agents.**
**Agents do not import from the scheduler.**
**Parsers do not import from agents or the scheduler.**
**Prompts do not import from parsers.**

If you find yourself importing across these boundaries, the abstraction is
wrong. Stop and ask.

---

## Data Structures

Core types live in `forge/core/models.py`. Do not redefine them elsewhere.

The primary types are:

- `AgentRequest` — the unit of schedulable work
- `AgentResponse` — the result of executing a request
- `DAGNode` — a node in the scheduler's work graph
- `SchedulerState` — the full scheduler snapshot (DAG + config)
- `PlanSpec`, `WorkSpec`, `IntegrateSpec` — typed specs per agent type
- `AgentSpec` — discriminated union of the above

When adding fields to these types, consider whether the field belongs on the
spec (agent-specific) or the request (scheduler-visible). The scheduler should
only inspect `agent_type`, `source`, `priority`, and `dependencies`. Everything
else is opaque spec payload.

---

## Models and Inference

Forge runs against local Ollama models by default. Do not assume any paid API
is available. Do not hardcode model names.

Model configuration lives in YAML spec files. Every role has a named model
config slot. Fallback chains are defined in config, not in code.

When selecting a model client:

1. Check the spec file for the role
2. Fall back through the chain defined in config
3. Raise clearly if no model is available — do not silently degrade

Do not suggest Anthropic or OpenAI models as solutions to quality problems.
Improve the prompt or the agent design instead.

---

## Testing

Run tests with:

```bash
pytest
```

Rules:

- Every module has a corresponding test file mirroring the source structure
- Tests for parsers use fixed string inputs — no live model calls
- Tests for prompt renders assert on content presence, not exact strings
- Scheduler tests use a mock agent that returns canned `AgentResponse`s
- No test should require a running Ollama instance
- One logical assertion per test where possible

When writing new tests, put them in `tests/` mirroring the source path:

```
forge/scheduler/dag.py → tests/scheduler/test_dag.py
forge/parsers/plan.py  → tests/parsers/test_plan.py
```

---

## Scheduler Behaviour

The scheduler loop is async. `max_concurrency` controls how many agents run
simultaneously. Start with `max_concurrency = 1`.

Node transition rules:

- `PENDING` → `READY` when all dependencies are `COMPLETED`
- `READY` → `RUNNING` when emitted to a worker
- `RUNNING` → `COMPLETED` or `FAILED` on response
- `PENDING` | `READY` | `RUNNING` → `CANCELLED` when any ancestor is `FAILED`

The scheduler never directly mutates State. It passes `AgentResponse` to the
Integrator, which calls `StateService.apply_delta`.

User-sourced requests always have `Priority.HIGH` and are never `CANCELLED` by
ancestor failure — they are DAG roots.

---

## What Claude Code May Do

- Edit existing files
- Create new files
- Run tests (`pytest`)
- Run the scheduler (`python -m forge.run`)

## What Claude Code Must Not Do

- Modify `CLAUDE.md`
- Delete test files
- Change enum values without explicit instruction
- Add backward-compatibility shims
- Install dependencies without explicit instruction
- Hardcode model names or API keys

---

## Auditability

Every scheduler cycle writes a structured record to `forge.jsonl`:

- `PLAN` — planner request and response
- `WORK` — worker request and response  
- `INTEGRATE` — integrator request and response
- `NODE_CANCELLED` — which nodes were cancelled and why

This log is the primary debugging tool. Keep it complete. Never silence it.