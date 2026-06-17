"""Tests for forge.core.file_filters — file filter constant contracts."""

from forge.core.file_filters import (
    CRITIC_EVIDENCE_EXCLUDED_DIRS,
    CRITIC_EVIDENCE_EXCLUDED_SUFFIXES,
    EXCLUDED_FILE_NAMES,
    GENERATED_ARTIFACT_DIRS,
    GENERATED_ARTIFACT_SUFFIXES,
    STATE_VIEW_EXCLUDED_DIRS,
    STATE_VIEW_EXCLUDED_SUFFIXES,
)

# ── EXCLUDED_FILE_NAMES ───────────────────────────────────────────────────────


def test_excluded_file_names_contains_cachedir_tag() -> None:
    """EXCLUDED_FILE_NAMES covers the pip/setuptools cache sentinel file."""
    assert "CACHEDIR.TAG" in EXCLUDED_FILE_NAMES


def test_excluded_file_names_contains_pyvenv_cfg() -> None:
    """EXCLUDED_FILE_NAMES covers the venv config file."""
    assert "pyvenv.cfg" in EXCLUDED_FILE_NAMES


# ── STATE_VIEW_EXCLUDED_* ─────────────────────────────────────────────────────


def test_state_view_excluded_dirs_excludes_pycache() -> None:
    """STATE_VIEW_EXCLUDED_DIRS covers Python bytecode cache directories."""
    assert "__pycache__" in STATE_VIEW_EXCLUDED_DIRS


def test_state_view_excluded_dirs_excludes_node_modules() -> None:
    """STATE_VIEW_EXCLUDED_DIRS covers JavaScript dependency directories."""
    assert "node_modules" in STATE_VIEW_EXCLUDED_DIRS


def test_state_view_excluded_dirs_excludes_build_outputs() -> None:
    """STATE_VIEW_EXCLUDED_DIRS includes legacy build output dirs absent from generated-artifact set."""
    assert "dist" in STATE_VIEW_EXCLUDED_DIRS
    assert "build" in STATE_VIEW_EXCLUDED_DIRS


def test_state_view_excluded_suffixes_excludes_compiled_python() -> None:
    """STATE_VIEW_EXCLUDED_SUFFIXES covers compiled Python bytecode extensions."""
    assert ".pyc" in STATE_VIEW_EXCLUDED_SUFFIXES
    assert ".pyo" in STATE_VIEW_EXCLUDED_SUFFIXES


def test_state_view_excluded_suffixes_excludes_lock_files() -> None:
    """STATE_VIEW_EXCLUDED_SUFFIXES covers package lock files."""
    assert ".lock" in STATE_VIEW_EXCLUDED_SUFFIXES


# ── GENERATED_ARTIFACT_* — narrower than StateView exclusions ────────────────


def test_generated_artifact_dirs_excludes_tool_caches() -> None:
    """GENERATED_ARTIFACT_DIRS covers volatile cache and virtualenv directories."""
    assert ".pytest_cache" in GENERATED_ARTIFACT_DIRS
    assert ".ruff_cache" in GENERATED_ARTIFACT_DIRS
    assert ".venv" in GENERATED_ARTIFACT_DIRS
    assert "__pycache__" in GENERATED_ARTIFACT_DIRS


def test_generated_artifact_dirs_does_not_include_build_outputs() -> None:
    """dist and build are StateView exclusions only — not generated artifact dirs."""
    assert "dist" not in GENERATED_ARTIFACT_DIRS
    assert "build" not in GENERATED_ARTIFACT_DIRS


def test_generated_artifact_suffixes_excludes_compiled_python() -> None:
    """GENERATED_ARTIFACT_SUFFIXES covers all compiled Python extension variants."""
    assert ".pyc" in GENERATED_ARTIFACT_SUFFIXES
    assert ".pyo" in GENERATED_ARTIFACT_SUFFIXES
    assert ".pyd" in GENERATED_ARTIFACT_SUFFIXES


def test_generated_artifact_suffixes_does_not_include_lock() -> None:
    """.lock is a StateView exclusion only — not a generated artifact suffix."""
    assert ".lock" not in GENERATED_ARTIFACT_SUFFIXES


# ── CRITIC_EVIDENCE_EXCLUDED_* ────────────────────────────────────────────────


def test_critic_evidence_excluded_dirs_includes_git_internals() -> None:
    """.git is a critic-evidence exclusion only — not in StateView or generated-artifact sets."""
    assert ".git" in CRITIC_EVIDENCE_EXCLUDED_DIRS
    assert ".git" not in STATE_VIEW_EXCLUDED_DIRS
    assert ".git" not in GENERATED_ARTIFACT_DIRS


def test_critic_evidence_excluded_dirs_includes_tool_caches() -> None:
    """CRITIC_EVIDENCE_EXCLUDED_DIRS covers all cache, venv, and dependency directories."""
    assert ".pytest_cache" in CRITIC_EVIDENCE_EXCLUDED_DIRS
    assert ".ruff_cache" in CRITIC_EVIDENCE_EXCLUDED_DIRS
    assert ".venv" in CRITIC_EVIDENCE_EXCLUDED_DIRS
    assert "__pycache__" in CRITIC_EVIDENCE_EXCLUDED_DIRS
    assert "node_modules" in CRITIC_EVIDENCE_EXCLUDED_DIRS


def test_critic_evidence_excluded_suffixes_excludes_compiled_python() -> None:
    """CRITIC_EVIDENCE_EXCLUDED_SUFFIXES covers compiled Python extensions but not lockfiles."""
    assert ".pyc" in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
    assert ".pyo" in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
    assert ".pyd" in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES


def test_critic_evidence_excluded_suffixes_does_not_exclude_lock_files() -> None:
    """.lock is NOT in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES — lockfiles are review-relevant evidence."""
    assert ".lock" not in CRITIC_EVIDENCE_EXCLUDED_SUFFIXES
