"""Workspace dataclass for managing the on-disk layout of a forge run."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from forge.languages.registry import LanguagePlugin


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

    def telemetry_dir(self) -> Path:
        """Return the path to the framework-owned telemetry directory."""
        return self.path / "telemetry"

    def init(self) -> None:
        """Create the workspace directory tree, raising NotADirectoryError if path is a file."""
        if self.path.exists() and not self.path.is_dir():
            raise NotADirectoryError(f"{self.path} exists but is not a directory")
        self.path.mkdir(parents=True, exist_ok=True)
        self.logs_dir().mkdir(exist_ok=True)
        self.telemetry_dir().mkdir(exist_ok=True)

    def init_artifact(self, name: str, plugin: LanguagePlugin | None = None) -> None:
        """Create the artifact root directory. For language-backed artifacts, also run init and sync if the directory is new."""
        artifact_dir = self.artifact_dir(name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if plugin is None or any(artifact_dir.iterdir()):
            return
        if not shutil.which("git"):
            raise RuntimeError(
                "git is required for language-backed artifacts but not found in PATH"
            )
        cmd = plugin.init_command.format(artifact_name=name)
        subprocess.run(cmd, shell=True, cwd=artifact_dir, check=True)
        subprocess.run(plugin.sync_command, shell=True, cwd=artifact_dir, check=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=artifact_dir, check=True)
        subprocess.run(["git", "add", "-A"], cwd=artifact_dir, check=True)
        subprocess.run(["git", "commit", "-m", f"init: {name}"], cwd=artifact_dir, check=True)

    def reset(self, artifact_names: list[str]) -> None:
        """Delete state, blackboard, and all contents of artifact directories."""
        self.state_path().unlink(missing_ok=True)
        self.blackboard_path().unlink(missing_ok=True)
        for name in artifact_names:
            d = self.artifact_dir(name)
            if d.exists():
                for item in d.iterdir():
                    shutil.rmtree(item) if item.is_dir() else item.unlink()

    def create_worktree(self, artifact_name: str, node_id: str) -> Path:
        """Create a git worktree for a work node, branched from main.
        Returns the worktree path."""
        artifact_dir = self.artifact_dir(artifact_name)
        worktree_path = self.path / f"{artifact_name}-work-{node_id}"
        branch_name = f"work/{node_id}"
        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "main"],
            cwd=artifact_dir,
            check=True,
        )
        return worktree_path

    def remove_worktree(self, artifact_name: str, node_id: str) -> None:
        """Remove a git worktree and its branch after integration."""
        artifact_dir = self.artifact_dir(artifact_name)
        worktree_path = self.path / f"{artifact_name}-work-{node_id}"
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=artifact_dir,
            check=True,
        )
        subprocess.run(
            ["git", "branch", "-D", f"work/{node_id}"],
            cwd=artifact_dir,
            check=True,
        )

    def get_current_sha(self, artifact_name: str) -> str:
        """Return the current HEAD commit SHA of the artifact main branch."""
        artifact_dir = self.artifact_dir(artifact_name)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=artifact_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
