"""Tests for forge.core.file_filters — file filter constant contracts."""

from forge.core.file_filters import (
    CRITIC_EVIDENCE_NOISE_DIRS,
    CRITIC_EVIDENCE_NOISE_SUFFIXES,
    GENERATED_ARTIFACT_DIRS,
    GENERATED_ARTIFACT_SUFFIXES,
    NOISE_FILE_NAMES,
    STATE_VIEW_NOISE_DIRS,
    STATE_VIEW_NOISE_SUFFIXES,
)

# ── NOISE_FILE_NAMES ──────────────────────────────────────────────────────────


def test_noise_file_names_contains_cachedir_tag() -> None:
    """NOISE_FILE_NAMES covers the pip/setuptools cache sentinel file."""
    assert "CACHEDIR.TAG" in NOISE_FILE_NAMES


def test_noise_file_names_contains_pyvenv_cfg() -> None:
    """NOISE_FILE_NAMES covers the venv config file."""
    assert "pyvenv.cfg" in NOISE_FILE_NAMES


# ── STATE_VIEW_NOISE_* ────────────────────────────────────────────────────────


def test_state_view_noise_dirs_excludes_pycache() -> None:
    """STATE_VIEW_NOISE_DIRS covers Python bytecode cache directories."""
    assert "__pycache__" in STATE_VIEW_NOISE_DIRS


def test_state_view_noise_dirs_excludes_node_modules() -> None:
    """STATE_VIEW_NOISE_DIRS covers JavaScript dependency directories."""
    assert "node_modules" in STATE_VIEW_NOISE_DIRS


def test_state_view_noise_dirs_excludes_build_outputs() -> None:
    """StateView noise includes legacy build output dirs absent from generated-artifact set."""
    assert "dist" in STATE_VIEW_NOISE_DIRS
    assert "build" in STATE_VIEW_NOISE_DIRS


def test_state_view_noise_suffixes_excludes_compiled_python() -> None:
    """STATE_VIEW_NOISE_SUFFIXES covers compiled Python bytecode extensions."""
    assert ".pyc" in STATE_VIEW_NOISE_SUFFIXES
    assert ".pyo" in STATE_VIEW_NOISE_SUFFIXES


def test_state_view_noise_suffixes_excludes_lock_files() -> None:
    """STATE_VIEW_NOISE_SUFFIXES covers package lock files."""
    assert ".lock" in STATE_VIEW_NOISE_SUFFIXES


# ── GENERATED_ARTIFACT_* — narrower than StateView noise ─────────────────────


def test_generated_artifact_dirs_excludes_tool_caches() -> None:
    """GENERATED_ARTIFACT_DIRS covers volatile cache and virtualenv directories."""
    assert ".pytest_cache" in GENERATED_ARTIFACT_DIRS
    assert ".ruff_cache" in GENERATED_ARTIFACT_DIRS
    assert ".venv" in GENERATED_ARTIFACT_DIRS
    assert "__pycache__" in GENERATED_ARTIFACT_DIRS


def test_generated_artifact_dirs_does_not_include_build_outputs() -> None:
    """dist and build are StateView noise only — not generated artifact dirs."""
    assert "dist" not in GENERATED_ARTIFACT_DIRS
    assert "build" not in GENERATED_ARTIFACT_DIRS


def test_generated_artifact_suffixes_excludes_compiled_python() -> None:
    """GENERATED_ARTIFACT_SUFFIXES covers all compiled Python extension variants."""
    assert ".pyc" in GENERATED_ARTIFACT_SUFFIXES
    assert ".pyo" in GENERATED_ARTIFACT_SUFFIXES
    assert ".pyd" in GENERATED_ARTIFACT_SUFFIXES


def test_generated_artifact_suffixes_does_not_include_lock() -> None:
    """.lock is StateView noise only — not a generated artifact suffix."""
    assert ".lock" not in GENERATED_ARTIFACT_SUFFIXES


# ── CRITIC_EVIDENCE_NOISE_* ───────────────────────────────────────────────────


def test_critic_evidence_noise_dirs_includes_git_internals() -> None:
    """.git is critic-evidence noise only — not in StateView or generated-artifact sets."""
    assert ".git" in CRITIC_EVIDENCE_NOISE_DIRS
    assert ".git" not in STATE_VIEW_NOISE_DIRS
    assert ".git" not in GENERATED_ARTIFACT_DIRS


def test_critic_evidence_noise_dirs_includes_tool_caches() -> None:
    """CRITIC_EVIDENCE_NOISE_DIRS covers all cache, venv, and dependency directories."""
    assert ".pytest_cache" in CRITIC_EVIDENCE_NOISE_DIRS
    assert ".ruff_cache" in CRITIC_EVIDENCE_NOISE_DIRS
    assert ".venv" in CRITIC_EVIDENCE_NOISE_DIRS
    assert "__pycache__" in CRITIC_EVIDENCE_NOISE_DIRS
    assert "node_modules" in CRITIC_EVIDENCE_NOISE_DIRS


def test_critic_evidence_noise_suffixes_excludes_compiled_and_lock() -> None:
    """CRITIC_EVIDENCE_NOISE_SUFFIXES covers compiled Python and lock file extensions."""
    assert ".pyc" in CRITIC_EVIDENCE_NOISE_SUFFIXES
    assert ".pyo" in CRITIC_EVIDENCE_NOISE_SUFFIXES
    assert ".pyd" in CRITIC_EVIDENCE_NOISE_SUFFIXES
    assert ".lock" in CRITIC_EVIDENCE_NOISE_SUFFIXES
