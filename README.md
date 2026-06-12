# Forge

Scheduler-centric agent framework.

Forge is an architecture for autonomous software construction built around one central claim: the scheduler should own the run.

Agents can propose plans, produce work, critique outputs, and request revisions, but they do not own graph consequences or artifact consequences. The scheduler owns the graph. The integrator owns artifact mutation. Each node is executed as a bounded game with typed outputs and explicit dispositions.

The result is an agentic system that is inspectable, replayable, and constrained by contracts instead of chat history.

## Introduction

Forge organizes work as a directed acyclic graph of nodes. Each node represents a bounded unit of agent work. Nodes are dispatched by the scheduler, executed through a producer -> critic -> referee loop, and reported back as typed responses.

The scheduler decides what those responses mean for the graph:

- accepted work can unlock dependents,
- rejected work can fail a node,
- revision feedback can cause another bounded attempt,
- decomposition can add new graph structure,
- already-complete work can be recorded without unnecessary mutation.

Nodes do not communicate directly. All feedback returns to the scheduler, and the scheduler is the only component that changes graph state.

## Core Principles

- **The scheduler owns the DAG.** Agents do not enqueue arbitrary work or mutate lifecycle state directly.
- **Every node is a bounded game.** Node execution has a request, a contract, a finite attempt budget, and a final disposition.
- **Review is explicit.** Producer output is checked by a critic and decided by a referee.
- **Outputs are typed.** Plans, work proposals, and failures cross boundaries as structured data.
- **Graph consequences are scheduler consequences.** A node result can affect dependencies, retries, failure propagation, and decomposition only through the scheduler.
- **Artifact consequences are integration consequences.** Workers propose changes; the integrator applies them.
- **Language behavior belongs to language plugins.** Test commands, language guidance, and artifact-specific mechanics stay outside the scheduler.
- **Telemetry is immutable.** Runs produce append-only trace data for observability and replay-oriented inspection.

## Architecture

Forge has two execution levels:

```text
Level 1: scheduler / graph loop

  Scheduler
    |
    +-- dispatch ready node -------------------+
    |                                          |
    v                                          |
  Typed node result <--------------------------+
    |
    +-- graph update
    |
    +-- integration, when accepted artifact changes need to be applied


Level 2: node / bounded game

  Node request
    |
    v
  Producer
    |
    v
  Critic
    |
    v
  Referee
    |
    v
  Typed node result
```

The scheduler level owns graph state and dispatch. The node level owns bounded execution of one request. The boundary between them is the typed node result: node execution produces it, and the scheduler interprets it. If the accepted result contains artifact changes, the integrator owns applying those changes to the artifact.

The important separation is ownership:

- The **scheduler** owns graph state.
- The **node executor** owns bounded attempt execution.
- The **producer** owns candidate output.
- The **critic** owns finding issues in that output.
- The **referee** owns the final disposition for the attempt.
- The **integrator** owns artifact mutation.
- The **language plugin** owns language-specific behavior.
- The **telemetry sink** owns append-only observability.

## Execution Model

A node executes as a bounded review game:

1. The scheduler dispatches a ready node.
2. The producer returns a typed output.
3. The critic evaluates the output against the node contract.
4. The referee decides what the system should do with the result.
5. The node returns a typed response to the scheduler.
6. The scheduler updates graph state.
7. Accepted artifact changes flow through integration.

The primary dispositions are:

- `ACCEPT`
- `REVISE`
- `REJECT`
- `DECOMPOSE`
- `ALREADY_DONE`

`REVISE` keeps feedback inside the bounded node game. `DECOMPOSE` returns a graph-level request to the scheduler. `ACCEPT`, `REJECT`, and `ALREADY_DONE` resolve the node.

## Scheduler and Nodes

The scheduler is responsible for:

- maintaining the DAG,
- tracking node lifecycle state,
- enforcing dependencies,
- dispatching ready work,
- handling failures,
- applying graph consequences from node results,
- deciding when the run is idle or complete.

A node is intentionally narrower. It receives a request and returns a response. It does not know the whole graph, does not mutate sibling nodes, and does not send messages to other nodes.

This keeps the graph legible. If a node causes more work to exist, that consequence is represented as scheduler-owned graph structure, not hidden agent state.

## Typed Outputs

Forge treats agent output as a boundary, not a suggestion.

Planner nodes return structured plan outputs. Worker nodes return structured artifact change proposals. Failures, revisions, and decomposition requests also travel through explicit response objects.

Typed outputs give the scheduler and integrator a stable contract:

- a plan can become graph structure,
- a delta can become an integration candidate,
- a rejection can become node failure,
- a decomposition request can become new planning work,
- an empty or already-complete result can be represented without pretending work changed.

The model call is allowed to be probabilistic. The runtime boundary is not.

## Language Plugins

Language-specific behavior belongs behind language plugins.

A language plugin can define how an artifact is initialized, how tests are run, and what guidance should be given to agents working in that language. This keeps the scheduler independent of Python, Rust, Zig, or any other language-specific workflow.

The scheduler should not need to know how to run a test suite. It should only need to know whether integration succeeded and what graph consequences follow.

## Telemetry and Replay

Forge writes immutable telemetry for each run under the workspace:

```text
workspaces/<workspace>/telemetry/runs/<run_id>/
  run.json
  events.jsonl
```

`run.json` records run metadata. `events.jsonl` records append-only framework events such as attempts, parsed producer responses, critic findings, referee decisions, revisions, exhaustion, decomposition, and node failures.

Telemetry is read-only from the perspective of agents. It is for humans and tooling, not for prompt context.

The purpose is observability:

- understand what happened in a run,
- inspect why a node revised or failed,
- compare attempts,
- review critic and referee decisions,
- debug graph behavior without reading raw JSON by hand.

## Trace Viewer

Forge includes a read-only trace viewer for telemetry.

```bash
uv run forge trace list
uv run forge trace latest
uv run forge trace <run_id>
uv run forge trace <run_id> --node <node_id>
```

The trace viewer summarizes runs, groups events by node, shows attempt timelines, and resolves short node prefixes when they are unambiguous.

Static HTML reports can also be generated for clickable inspection:

```bash
uv run forge trace html latest
uv run forge trace html <run_id>
```

Reports are written next to the run telemetry:

```text
workspaces/<workspace>/telemetry/runs/<run_id>/index.html
```

## Example Usage

Install dependencies:

```bash
uv sync
```

Start a run from a Forge config:

```bash
uv run forge start forge.yaml
```

Reset a workspace:

```bash
uv run forge reset forge.yaml
```

Inspect telemetry:

```bash
uv run forge trace list
uv run forge trace latest
```

A minimal config describes the northstar goal, workspace, concurrency, and artifacts:

```yaml
northstar: "build a web scraper in Python"
workspace: ./workspaces/scraper
concurrency: 1
artifacts:
  - name: codebase
    type: coding
    language: python
```

## Project Structure

```text
src/forge/
  agents/       Producer, critic, referee, planner, worker, and integrator logic
  adapters/     Artifact adapter definitions
  core/         Scheduler, models, config, persistence, telemetry, trace viewing
  languages/    Language plugin registry
  llm/          Provider transport adapters
  tools/        Tool registry and built-in tool schemas
  run.py        CLI entry point

adapters/       Built-in adapter specs
languages/      Built-in language plugin specs
tests/          Unit and integration tests for the framework contracts
workspaces/     Local run state, artifacts, and telemetry
```

The important boundary is not the directory layout. It is the ownership model: scheduler for graph state, integrator for artifacts, plugins for language behavior, telemetry for observation.

## Development

Run the test suite:

```bash
uv run pytest
```

Run static checks:

```bash
uv run pyright
uv run ruff check .
```

Forge development should preserve the core boundaries:

- do not let agents mutate graph state,
- do not let workers mutate artifacts directly,
- do not hide graph consequences in prompts,
- do not put language-specific behavior in the scheduler,
- do not expose telemetry as agent context,
- keep runtime boundaries typed.

The framework is easiest to reason about when every consequential state transition has one owner.
