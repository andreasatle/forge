"""Functions for saving and loading scheduler state to/from the workspace."""

from pathlib import Path

from forge.core.models import NodeState, SchedulerState
from forge.core.workspace import Workspace


def save_run(state: SchedulerState, workspace: Workspace) -> Path:
    """Serialize state to JSON and write it to the workspace state path, returning that path."""
    path = workspace.state_path()
    path.write_text(state.model_dump_json(indent=2))
    return path


def load_run(workspace: Workspace) -> SchedulerState:
    """Deserialize state from the workspace state path, resetting any RUNNING nodes to PENDING."""
    state = SchedulerState.model_validate_json(workspace.state_path().read_text())
    recovered_dag = {
        nid: (node.with_state(NodeState.PENDING) if node.node_state == NodeState.RUNNING else node)
        for nid, node in state.dag.items()
    }
    return state.model_copy(update={"dag": recovered_dag})
