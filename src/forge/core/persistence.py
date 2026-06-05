from datetime import datetime
from pathlib import Path

from forge.core.models import NodeState, SchedulerState


def save_run(state: SchedulerState, runs_dir: Path = Path("runs")) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json"
    path = runs_dir / filename
    path.write_text(state.model_dump_json(indent=2))
    return path


def load_run(path: Path) -> SchedulerState:
    state = SchedulerState.model_validate_json(path.read_text())
    recovered_dag = {
        nid: (node.with_state(NodeState.PENDING) if node.node_state == NodeState.RUNNING else node)
        for nid, node in state.dag.items()
    }
    return state.model_copy(update={"dag": recovered_dag})
