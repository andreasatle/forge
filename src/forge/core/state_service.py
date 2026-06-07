"""StateService: single mutation boundary for artifact state."""

import re
import subprocess
import tomllib
from pathlib import Path

from forge.core.models import DeltaState, StateView
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin


def _read_python_deps(artifact_dir: Path) -> list[str]:
    manifest = artifact_dir / "pyproject.toml"
    if not manifest.exists():
        return []
    with manifest.open("rb") as f:
        data = tomllib.load(f)
    return list(data.get("project", {}).get("dependencies", []))


def _read_rust_deps(artifact_dir: Path) -> list[str]:
    manifest = artifact_dir / "Cargo.toml"
    if not manifest.exists():
        return []
    with manifest.open("rb") as f:
        data = tomllib.load(f)
    return list(data.get("dependencies", {}).keys())


_DEPS_READERS = {
    "python": _read_python_deps,
    "rust": _read_rust_deps,
}


class StateService:
    """Single mutation boundary for artifact state — builds StateView and applies DeltaState."""

    def __init__(self, workspace: Workspace, artifact_name: str, plugin: LanguagePlugin | None = None) -> None:
        self._workspace = workspace
        self._artifact_name = artifact_name
        self._plugin = plugin

    def build_state_view(self) -> StateView:
        """Build a StateView from the current artifact directory."""
        artifact_dir = self._workspace.artifact_dir(self._artifact_name)
        language = self._plugin.name if self._plugin else None

        if not artifact_dir.exists():
            return StateView(artifact_name=self._artifact_name, language=language, files=[], dependencies=[])

        files = sorted(
            str(f.relative_to(artifact_dir))
            for f in artifact_dir.rglob("*")
            if f.is_file()
        )

        reader = _DEPS_READERS.get(self._plugin.name) if self._plugin else None
        deps = reader(artifact_dir) if reader else []

        return StateView(artifact_name=self._artifact_name, language=language, files=files, dependencies=deps)

    def apply_delta(self, delta: DeltaState) -> None:
        """Apply a DeltaState to the artifact directory — writes files and applies edits."""
        artifact_dir = self._workspace.artifact_dir(self._artifact_name)

        for fw in delta.new_files:
            target = artifact_dir / fw.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fw.content, encoding="utf-8")

        for edit in delta.edits:
            target = artifact_dir / edit.path
            if not target.exists():
                raise FileNotFoundError(f"file not found: {edit.path}")
            content = target.read_text(encoding="utf-8")
            count = len(re.findall(re.escape(edit.old), content))
            if count == 0:
                raise ValueError(f"old string not found in {edit.path}")
            if count > 1:
                raise ValueError(f"old string not unique in {edit.path} — found {count} occurrences")
            target.write_text(content.replace(edit.old, edit.new, 1), encoding="utf-8")

        if self._plugin and delta.dependencies:
            for dep in delta.dependencies:
                cmd = self._plugin.add_dependency_command.format(package=dep)
                subprocess.run(cmd, shell=True, cwd=artifact_dir, check=True)
