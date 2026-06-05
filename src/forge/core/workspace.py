import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Workspace:
    path: Path

    def __post_init__(self) -> None:
        self.path = self.path.resolve()

    def state_path(self) -> Path:
        return self.path / "state.json"

    def blackboard_path(self) -> Path:
        return self.path / "blackboard.json"

    def outputs_dir(self) -> Path:
        return self.path / "outputs"

    def logs_dir(self) -> Path:
        return self.path / "logs"

    def init(self) -> None:
        if self.path.exists() and not self.path.is_dir():
            raise NotADirectoryError(f"{self.path} exists but is not a directory")
        self.path.mkdir(parents=True, exist_ok=True)
        self.outputs_dir().mkdir(exist_ok=True)
        self.logs_dir().mkdir(exist_ok=True)

    def reset(self) -> None:
        self.state_path().unlink(missing_ok=True)
        self.blackboard_path().unlink(missing_ok=True)
        for d in (self.outputs_dir(), self.logs_dir()):
            for item in d.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
