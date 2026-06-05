# Forge

A scheduler-centric agentic framework. The scheduler owns the DAG; agents are opaque functions `AgentRequest → AgentResponse`.

## Status

The core loop runs end-to-end against a local Ollama instance:

- **Core models** (`forge/core/models.py`) — `AgentRequest`, `AgentResponse`, `DAGNode`, `SchedulerState`, and spec types (`PlanSpec`, `WorkSpec`, `IntegrateSpec`). All are frozen Pydantic models.
- **Scheduler** (`forge/core/scheduler.py`) — async DAG-driven dispatch loop. Respects `max_concurrency`, propagates failures to dependents, re-injects the global planner on idle, and terminates when the planner returns an empty follow-up list.
- **Runner** (`forge/core/runner.py`) — routes `AgentRequest`s to registered handlers by `AgentType`.
- **Planner agent** — calls Ollama, parses the response into a list of `WorkSpec` nodes, and emits them as follow-ups.
- **Worker agent** — calls Ollama with the objective and success condition, returns a `delta` with the result.
- **Adapter registry** (`forge/adapters/registry.py`) — file-backed registry loaded from the `adapters/` directory. Built-in adapters: `audit`, `coding`, `document`.
- **Workspace** (`forge/core/workspace.py`) — root directory for all run outputs. Owns `state.json`, `blackboard.json`, `outputs/`, and `logs/`.
- **Persistence** (`forge/core/persistence.py`) — saves and loads `SchedulerState` to/from `workspace/state.json`. Running nodes are reset to pending on resume.
- **Config** (`forge/core/config.py`) — YAML-driven `ForgeConfig` with `northstar`, `workspace`, `concurrency`, and `verbose`.
- **Entry point** (`forge/run.py`) — `start` and `reset` subcommands driven by a config file.

## Running

```bash
uv sync
```

Create a config file (or copy `forge.yaml`):

```yaml
northstar: "build a web scraper in Python"
workspace: ./workspaces/scraper
concurrency: 1
verbose: true
```

Start a run (resumes automatically if `state.json` exists in the workspace):

```bash
uv run forge start forge.yaml
```

Reset the workspace state and start fresh next time:

```bash
uv run forge reset forge.yaml
uv run forge start forge.yaml
```

Example output:

```
adapters: ['audit', 'coding', 'document']
→ dispatched: plan (...)
✓ completed: plan (...)
→ dispatched: work (...)
✓ completed: work (...)
~ idle: 6 nodes in DAG
→ dispatched: plan (...)
✓ completed: plan (...)
run saved: ./workspaces/scraper/state.json

Done — completed: 3, failed: 0, cancelled: 4

DAG summary:
  ...  plan        completed   deps=0  delta=None
  ...  work        completed   deps=0  delta=None
  ...  plan        completed   deps=0  delta=None
  ...  work        cancelled   deps=1  delta=None
```

## Tests

```bash
uv run pytest
```

72 tests covering the scheduler loop, runner routing, handler behaviour, adapter registry, workspace lifecycle, persistence, and config loading. No live model calls required.

## Architecture

```
forge.yaml (ForgeConfig)
  ↓
Workspace.init()  ←→  state.json (save/load/resume)
  ↓
SchedulerState (DAG + northstar + concurrency)
  ↓
Scheduler emits READY nodes to Runner
  ↓
Runner routes by AgentType → Handler
  │
  ├── plan  → PlannerAgent  → Ollama → WorkSpec follow-ups
  ├── work  → WorkAgent     → Ollama → delta result
  └── integrate → stub
  ↓
Scheduler applies result, grows DAG, re-checks for idle
  ↓
Global planner re-injected on idle → terminates when it returns empty follow_up
```

The scheduler has no knowledge of what handlers do. Handlers have no knowledge of the DAG.
