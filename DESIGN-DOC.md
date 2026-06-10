# Forge Task Execution Model

## Core Principle

The first-class object in Forge is the **Task**.

The DAG is made of tasks, not agents.

Tasks may internally use any amount of complexity, but the scheduler sees only tasks and their lifecycle.

---

# Task Lifecycle

Every task progresses through:

```text
EMITTED
↓
DISPATCHED
↓
INTEGRATED
```

or, if something goes wrong:

```text
EMITTED
↓
DISPATCHED
↓
FAILED
```

or

```text
EMITTED
↓
DISPATCHED
↓
CANCELLED
```

These states represent:

### Emit

The task becomes known.

Nothing has been executed.

### Dispatch

Execution begins.

The task performs its internal reasoning.

### Integrate

The task result becomes reality.

Only integrated tasks affect the world.

---

# The Three Main Components

## Planner

The planner observes the current state and emits tasks.

Conceptually:

```text
State
↓
Planner
↓
TaskSpecs
```

The planner owns:

* decomposition;
* semantic dependencies;
* hierarchical planning.

The planner does not execute tasks.

The planner does not modify state.

---

## Scheduler

The scheduler owns the DAG and dispatches ready tasks.

It knows:

* which tasks exist;
* which tasks have been integrated;
* which tasks are ready.

The scheduler owns:

* readiness;
* concurrency;
* execution order.

The scheduler does not understand planning.

The scheduler does not understand internal task execution.

It simply dispatches ready tasks.

---

## Integrator

The integrator commits task results.

Only the integrator changes reality.

Conceptually:

```text
State + Result
↓
Updated State
```

The integrator owns:

* state mutation;
* validation;
* failure handling;
* rollback or rejection;
* committing successful work.

---

# Types of Tasks

A task may be one of two kinds.

## Planning Task

A planning task produces more tasks.

Conceptually:

```text
StateView
↓
Planning
↓
TaskSpecs
```

Integration adds the emitted tasks to the DAG.

Planning tasks do not directly modify state.

---

## Work Task

A work task produces a DeltaState.

Conceptually:

```text
StateView
↓
Execution
↓
DeltaState
```

Integration applies the DeltaState to state.

---

# Hierarchical Planning

Planning is naturally hierarchical.

Example:

```text
Plan Web App
├── Plan Database
├── Plan API
├── Plan Frontend
└── Plan Deployment
```

Subplans may themselves emit additional tasks:

```text
Plan API
├── Create Models
├── Create Routes
└── Create API Tests
```

Planning tasks and work tasks share the same lifecycle:

```text
EMITTED
↓
DISPATCHED
↓
INTEGRATED
```

The scheduler treats them identically.

---

# Dependencies

Dependencies exist between tasks.

Example:

```text
Task C depends on Task A and Task B
```

Task C may only be dispatched after A and B have been successfully integrated.

Dependencies represent:

> State availability

rather than:

> Proposal availability.

---

# Hidden Internal Execution

The scheduler does not know how a task performs its work.

A task may internally use:

* a single LLM call;
* a planner-worker-critic loop;
* multiple models;
* Claude Code subagents;
* retries;
* critics;
* any future mechanism.

These are implementation details.

The scheduler only observes:

```text
Task
↓
Result
```

---

# Main Loop

The system repeatedly performs:

```text
State
↓
Planner
↓
Emit tasks
↓
Scheduler
↓
Dispatch ready tasks
↓
Task execution
↓
Result
↓
Integrator
↓
Updated state
↓
Repeat
```

Thus Forge operates through:

```text
Emit
↓
Dispatch
↓
Integrate
↓
Repeat
```

with tasks serving as the primary abstraction.

---

# Ownership

Planner

* creates possibilities.

Scheduler

* executes possibilities.

Integrator

* commits possibilities to reality.

State

* remains the source of truth.

Tasks

* are the fundamental unit of execution.

Everything else is an implementation detail.

---

# Guiding Principle

Forge should be made as simple as possible, but not simpler.

Complexity should be hidden inside task execution.

The framework itself should expose only the essential concepts:

```text
Task
Emit
Dispatch
Integrate
State
```
