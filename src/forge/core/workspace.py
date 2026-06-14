"""Workspace dataclass for managing the on-disk layout of a forge run."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, cast

from forge.languages.registry import LanguagePlugin

_GIT_LOCAL_ENV_VARS = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)

_PYTHON_GITIGNORE_LINES = (
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
)


def git_subprocess_env() -> dict[str, str]:
    """Return an environment where parent-repository Git variables are removed."""
    env = os.environ.copy()
    for key in _GIT_LOCAL_ENV_VARS:
        env.pop(key, None)
    return env


def run_git(
    args: list[str], cwd: str | PathLike[str], **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Run a git command scoped by cwd, independent of inherited hook Git env."""
    return cast(
        "subprocess.CompletedProcess[Any]",
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            env=git_subprocess_env(),
            **kwargs,
        ),
    )


def _ensure_python_gitignore(artifact_dir: Path) -> None:
    gitignore = artifact_dir / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    missing = [line for line in _PYTHON_GITIGNORE_LINES if line not in existing]
    if not missing:
        return
    prefix = "\n" if existing and existing[-1] != "" else ""
    gitignore.write_text(
        "\n".join(existing) + prefix + "\n".join(missing) + "\n",
        encoding="utf-8",
    )


@dataclass
class Workspace:
    """On-disk workspace rooted at a given path with well-known subdirectories."""

    path: Path

    def __post_init__(self) -> None:
        self.path = self.path.resolve()

    def state_path(self) -> Path:
        """Return the path to the scheduler state JSON file."""
        return self.path / "state.json"

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
        was_empty = not any(artifact_dir.iterdir())
        if plugin is not None and was_empty:
            cmd = plugin.init_command.format(artifact_name=name)
            subprocess.run(cmd, shell=True, cwd=artifact_dir, check=True)
            subprocess.run(plugin.sync_command, shell=True, cwd=artifact_dir, check=True)
        if plugin is not None and plugin.name == "python":
            _ensure_python_gitignore(artifact_dir)
        if (artifact_dir / ".git").exists():
            return
        if not shutil.which("git"):
            raise RuntimeError("git is required for artifacts but not found in PATH")
        run_git(["init", "-b", "main"], cwd=artifact_dir)
        run_git(["add", "-A"], cwd=artifact_dir)
        run_git(
            ["commit", "--allow-empty", "-m", f"init: {name}"],
            cwd=artifact_dir,
        )

    def reset(self, artifact_names: list[str]) -> None:
        """Delete state and all contents of artifact directories."""
        self.state_path().unlink(missing_ok=True)
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
        run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), "main"],
            cwd=artifact_dir,
        )
        return worktree_path

    def worktree_path(self, artifact_name: str, node_id: str) -> Path:
        """Return the expected worktree path for a work node."""
        return self.path / f"{artifact_name}-work-{node_id}"

    def remove_worktree(self, artifact_name: str, node_id: str) -> None:
        """Remove a git worktree and its branch after integration."""
        artifact_dir = self.artifact_dir(artifact_name)
        worktree_path = self.path / f"{artifact_name}-work-{node_id}"
        run_git(
            ["worktree", "remove", "--force", str(worktree_path)],
            cwd=artifact_dir,
        )
        run_git(
            ["branch", "-D", f"work/{node_id}"],
            cwd=artifact_dir,
        )

    def get_current_sha(self, artifact_name: str) -> str:
        """Return the current HEAD commit SHA of the artifact main branch."""
        artifact_dir = self.artifact_dir(artifact_name)
        result = run_git(
            ["rev-parse", "HEAD"],
            cwd=artifact_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
