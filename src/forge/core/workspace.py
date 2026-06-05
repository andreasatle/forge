"""Workspace dataclass for managing the on-disk layout of a forge run."""

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Workspace:
    """On-disk workspace rooted at a given path with well-known subdirectories."""

    path: Path

    def __post_init__(self) -> None:
        self.path = self.path.resolve()

    def state_path(self) -> Path:
        """Return the path to the scheduler state JSON file."""
        return self.path / "state.json"

    def blackboard_path(self) -> Path:
        """Return the path to the shared blackboard JSON file."""
        return self.path / "blackboard.json"

    def artifact_dir(self, name: str) -> Path:
        """Return the root directory for the named artifact directly under the workspace."""
        return self.path / name

    def logs_dir(self) -> Path:
        """Return the path to the run logs directory."""
        return self.path / "logs"

    def init(self) -> None:
        """Create the workspace directory tree, raising NotADirectoryError if path is a file."""
        if self.path.exists() and not self.path.is_dir():
            raise NotADirectoryError(f"{self.path} exists but is not a directory")
        self.path.mkdir(parents=True, exist_ok=True)
        self.logs_dir().mkdir(exist_ok=True)

    def init_artifact(self, name: str) -> None:
        """Create the artifact root directory if it does not exist."""
        self.artifact_dir(name).mkdir(parents=True, exist_ok=True)

    def reset(self, artifact_names: list[str]) -> None:
        """Delete state, blackboard, and all contents of artifact directories."""
        self.state_path().unlink(missing_ok=True)
        self.blackboard_path().unlink(missing_ok=True)
        for name in artifact_names:
            d = self.artifact_dir(name)
            if d.exists():
                for item in d.iterdir():
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
