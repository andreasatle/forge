"""StateService: single mutation boundary for artifact state."""

import subprocess
from pathlib import Path

from forge.core.file_filters import (
    EXCLUDED_FILE_NAMES,
    GENERATED_ARTIFACT_DIRS,
    GENERATED_ARTIFACT_SUFFIXES,
    STATE_VIEW_EXCLUDED_DIRS,
    STATE_VIEW_EXCLUDED_SUFFIXES,
)
from forge.core.models import FileView, RunResult, StateView, WorkOutput
from forge.core.workspace import Workspace, run_git
from forge.languages.registry import LanguagePlugin

_TEST_TIMEOUT = 60
MAX_INTEGRATION_TEST_OUTPUT_CHARS = 4000


class IntegrationTestFailure(RuntimeError):
    """Raised when post-merge integration tests fail and rollback succeeds."""

    def __init__(self, *, summary: str, output: str, rollback_sha: str) -> None:
        self.summary = summary
        self.output_excerpt = _bounded_test_output(output)
        self.rollback_sha = rollback_sha
        super().__init__(f"tests failed after work output: {self.output_excerpt}")


def _bounded_test_output(output: str) -> str:
    if len(output) <= MAX_INTEGRATION_TEST_OUTPUT_CHARS:
        return output
    omitted = len(output) - MAX_INTEGRATION_TEST_OUTPUT_CHARS
    return output[:MAX_INTEGRATION_TEST_OUTPUT_CHARS].rstrip() + (
        f"\n...[truncated {omitted} chars]"
    )


def _is_noise(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return (
        any(p.startswith(".") for p in parts)
        or any(p in STATE_VIEW_EXCLUDED_DIRS for p in parts)
        or path.name in EXCLUDED_FILE_NAMES
        or path.suffix in STATE_VIEW_EXCLUDED_SUFFIXES
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


def _is_generated_artifact_path(path: str) -> bool:
    parts = Path(path).parts
    return (
        any(part in GENERATED_ARTIFACT_DIRS for part in parts)
        or Path(path).suffix in GENERATED_ARTIFACT_SUFFIXES
    )


def _clean_ignored_files(cwd: Path) -> None:
    run_git(["clean", "-fdX"], cwd=cwd)


def _status_lines(cwd: Path) -> list[str]:
    result = run_git(
        ["status", "--porcelain", "--untracked-files=normal"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _status_path(line: str) -> str:
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _restore_tracked_generated_changes(cwd: Path) -> None:
    paths = [
        _status_path(line)
        for line in _status_lines(cwd)
        if not line.startswith("?? ") and _is_generated_artifact_path(_status_path(line))
    ]
    if paths:
        run_git(["checkout", "--", *paths], cwd=cwd)


def _ensure_clean_for_merge(cwd: Path) -> None:
    _clean_ignored_files(cwd)
    _restore_tracked_generated_changes(cwd)
    remaining = _status_lines(cwd)
    if remaining:
        raise RuntimeError(
            "artifact worktree has uncommitted changes before merge:\n" + "\n".join(remaining)
        )


def _format_git_error(error: subprocess.CalledProcessError) -> str:
    parts = [f"git command failed with exit code {error.returncode}: {' '.join(error.cmd)}"]
    stdout = getattr(error, "stdout", None)
    stderr = getattr(error, "stderr", None)
    if stdout:
        parts.append(str(stdout).strip())
    if stderr:
        parts.append(str(stderr).strip())
    return "\n".join(part for part in parts if part)


class StateService:
    """Single mutation boundary for artifact state."""

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
                version=self._version,
            )

        version_sha = ""
        if (artifact_dir / ".git").exists():
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
            version=self._version,
            version_sha=version_sha,
        )

    async def apply_work_output(
        self, output: WorkOutput, node_id: str, dispatch_sha: str = ""
    ) -> None:
        """Apply git-native worktree changes — commit, merge to main, run tests,
        commit on pass or rollback on fail."""
        if dispatch_sha != "":
            current_sha = self._workspace.get_current_sha(self._artifact_name)
            if dispatch_sha != current_sha:
                raise RuntimeError(
                    f"stale base_version: output based on {dispatch_sha!r} "
                    f"but HEAD is {current_sha!r}"
                )

        worktree_path = self._workspace.worktree_path(self._artifact_name, node_id)
        if not worktree_path.exists():
            raise RuntimeError(f"worktree not found for node {node_id}: {worktree_path}")
        artifact_dir = self._workspace.artifact_dir(self._artifact_name)
        merge_completed = False
        try:
            _clean_ignored_files(worktree_path)
            _restore_tracked_generated_changes(worktree_path)
            if not _status_lines(worktree_path):
                raise RuntimeError("no worktree changes produced")

            run_git(["add", "-A", "--", "."], cwd=worktree_path, capture_output=True, text=True)
            run_git(
                ["commit", "-m", f"work: {node_id}"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )

            _ensure_clean_for_merge(artifact_dir)
            pre_merge_sha = self._workspace.get_current_sha(self._artifact_name)
            run_git(
                ["merge", "--no-ff", f"work/{node_id}", "-m", f"integrated: {node_id}"],
                cwd=artifact_dir,
                capture_output=True,
                text=True,
            )
            merge_completed = True

            result = self.run_tests()
            if not result.passed:
                run_git(
                    ["reset", "--hard", pre_merge_sha],
                    cwd=artifact_dir,
                    capture_output=True,
                    text=True,
                )
                raise IntegrationTestFailure(
                    summary=result.summary,
                    output=result.output,
                    rollback_sha=pre_merge_sha,
                )

            self._version += 1

        except subprocess.CalledProcessError as e:
            if not merge_completed:
                try:
                    run_git(["merge", "--abort"], cwd=artifact_dir)
                except subprocess.CalledProcessError:
                    pass
            raise RuntimeError(_format_git_error(e)) from e

    def remove_worktree(self, node_id: str) -> None:
        """Remove the git worktree for a work node after integration."""
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
