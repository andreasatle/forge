"""StateService: single mutation boundary for artifact state."""

import re
import subprocess
import uuid
from pathlib import Path

from forge.core.models import DeltaState, FileView, RunResult, StateView, WorkOutput
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin

_EXCLUDED_DIR_NAMES = frozenset({"__pycache__", "node_modules", "dist", "build"})
_EXCLUDED_FILE_NAMES = frozenset({"CACHEDIR.TAG", "pyvenv.cfg"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".lock", ".egg-info"})

_TEST_TIMEOUT = 60


def _is_noise(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return (
        any(p.startswith(".") for p in parts)
        or any(p in _EXCLUDED_DIR_NAMES for p in parts)
        or path.name in _EXCLUDED_FILE_NAMES
        or path.suffix in _EXCLUDED_SUFFIXES
    )


def _parse_test_result(raw: str, returncode: int = 0) -> RunResult:
    if "timed out" in raw:
        return RunResult(
            passed=False,
            failures=["timed out"],
            summary=raw.strip(),
            output=raw,
        )
    lines = raw.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    summary = non_empty[-1] if non_empty else raw.strip()
    passed = returncode == 0
    failures = [] if passed else [summary]
    return RunResult(passed=passed, failures=failures, summary=summary, output=raw)


class StateService:
    """Single mutation boundary for artifact state — builds StateView and applies DeltaState."""

    def __init__(
        self, workspace: Workspace, artifact_name: str, plugin: LanguagePlugin | None = None
    ) -> None:
        self._workspace = workspace
        self._artifact_name = artifact_name
        self._plugin = plugin
        self._version: int = 0

    @property
    def current_version(self) -> int:
        """Return the current monotonic version counter for this artifact."""
        return self._version

    def build_state_view(self) -> StateView:
        """Build a StateView from the current artifact directory."""
        artifact_dir = self._workspace.artifact_dir(self._artifact_name)
        language = self._plugin.name if self._plugin else None

        if not artifact_dir.exists():
            return StateView(
                artifact_name=self._artifact_name,
                language=language,
                files=[],
                dependencies=[],
                version=self._version,
            )

        version_sha = ""
        if self._plugin:
            version_sha = self._workspace.get_current_sha(self._artifact_name)

        file_views: list[FileView] = []
        for f in sorted(artifact_dir.rglob("*")):
            if not f.is_file() or _is_noise(f, artifact_dir):
                continue
            try:
                content = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            file_views.append(FileView(path=str(f.relative_to(artifact_dir)), content=content))

        return StateView(
            artifact_name=self._artifact_name,
            language=language,
            files=file_views,
            dependencies=[],
            version=self._version,
            version_sha=version_sha,
        )

    def apply_delta(self, delta: DeltaState) -> None:
        """Apply a DeltaState to the artifact directory — writes files, applies edits, and tests.

        When a language plugin is configured the artifact has a git repo. This method uses git
        stash as a transaction: stash before applying, commit on success, pop on test failure.
        """
        artifact_dir = self._workspace.artifact_dir(self._artifact_name)

        stashed = False
        if self._plugin:
            subprocess.run(["git", "-C", str(artifact_dir), "add", "-A"], check=True)
            stash_result = subprocess.run(
                ["git", "-C", str(artifact_dir), "stash"],
                capture_output=True,
                text=True,
                check=True,
            )
            stashed = "No local changes to save" not in stash_result.stdout

        for fw in delta.new_files:
            target = artifact_dir / fw.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fw.content, encoding="utf-8")

        for edit in delta.edits:
            if not edit.old or not edit.old.strip():
                raise ValueError(
                    f"Edit for {edit.path} has empty 'old' string — "
                    "use new_files to create content, not edits"
                )
            target = artifact_dir / edit.path
            if not target.exists():
                raise FileNotFoundError(f"file not found: {edit.path}")
            content = target.read_text(encoding="utf-8")
            count = len(re.findall(re.escape(edit.old), content))
            if count == 0:
                raise ValueError(f"old string not found in {edit.path}")
            if count > 1:
                raise ValueError(
                    f"old string not unique in {edit.path} — found {count} occurrences"
                )
            target.write_text(content.replace(edit.old, edit.new, 1), encoding="utf-8")

        if self._plugin and delta.dependencies:
            for dep in delta.dependencies:
                cmd = self._plugin.add_dependency_command.format(package=dep)
                subprocess.run(cmd, shell=True, cwd=artifact_dir, check=True)

        if self._plugin:
            test_result = self.run_tests()
            if test_result.passed:
                if stashed:
                    subprocess.run(["git", "-C", str(artifact_dir), "stash", "drop"], check=True)
                subprocess.run(["git", "-C", str(artifact_dir), "add", "-A"], check=True)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(artifact_dir),
                        "commit",
                        "-m",
                        f"integrated: {uuid.uuid4()}",
                    ],
                    check=True,
                )
                self._version += 1
            else:
                if stashed:
                    subprocess.run(["git", "-C", str(artifact_dir), "stash", "pop"], check=True)
                raise RuntimeError(
                    f"tests failed after delta: {test_result.summary}\n{test_result.output}"
                )
        else:
            self._version += 1

    async def apply_work_output(self, output: WorkOutput, node_id: str) -> None:
        """Apply WorkOutput via git worktree — write files, merge to main, run tests,
        commit on pass or rollback on fail."""
        if output.base_version != "":
            current_sha = self._workspace.get_current_sha(self._artifact_name)
            if output.base_version != current_sha:
                raise RuntimeError(
                    f"stale base_version: output based on {output.base_version!r} "
                    f"but HEAD is {current_sha!r}"
                )

        worktree_path = self._workspace.create_worktree(self._artifact_name, node_id)
        try:
            for file in output.files:
                dest = worktree_path / file.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(file.content)

            subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", f"work: {node_id}"],
                cwd=worktree_path,
                check=True,
            )

            artifact_dir = self._workspace.artifact_dir(self._artifact_name)
            subprocess.run(
                ["git", "merge", "--no-ff", f"work/{node_id}", "-m", f"integrated: {node_id}"],
                cwd=artifact_dir,
                check=True,
            )

            result = self.run_tests()
            if not result.passed:
                subprocess.run(
                    ["git", "reset", "--hard", "HEAD~1"],
                    cwd=artifact_dir,
                    check=True,
                )
                raise RuntimeError(f"tests failed after delta: {result.output}")

            self._version += 1

        finally:
            self._workspace.remove_worktree(self._artifact_name, node_id)

    def run_tests(self) -> RunResult:
        """Run the language plugin test command and return structured result."""
        if not self._plugin:
            return RunResult(passed=True)
        artifact_dir = self._workspace.artifact_dir(self._artifact_name)
        try:
            result = subprocess.run(
                self._plugin.test_command,
                shell=True,
                cwd=artifact_dir,
                capture_output=True,
                text=True,
                timeout=_TEST_TIMEOUT,
            )
            raw = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return RunResult(
                passed=False,
                failures=["timed out"],
                summary="test command timed out",
                output="test command timed out",
            )
        return _parse_test_result(raw, result.returncode)
