from datetime import datetime
from pathlib import Path

from forge.core.models import SchedulerState


def save_run(state: SchedulerState, runs_dir: Path = Path("runs")) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json"
    path = runs_dir / filename
    path.write_text(state.model_dump_json(indent=2))
    return path
