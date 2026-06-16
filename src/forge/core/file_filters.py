"""File filter constants for Forge.

Three semantically distinct groups — do not merge them:

  STATE_VIEW_*          — files excluded from the artifact StateView snapshot.
  GENERATED_ARTIFACT_*  — tracked files restored/cleaned before branch merge.
  CRITIC_EVIDENCE_*     — untracked files excluded from critic review evidence.

EXCLUDED_FILE_NAMES is shared: both StateView and critic evidence exclude the same
well-known cache/config file names.
"""

# Shared by StateView exclusion and critic evidence exclusion.
EXCLUDED_FILE_NAMES: frozenset[str] = frozenset({"CACHEDIR.TAG", "pyvenv.cfg"})

# StateView snapshot exclusion — broad: includes legacy build outputs (dist,
# build) that should not appear in the agent's view of the project.
STATE_VIEW_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {"__pycache__", "node_modules", "dist", "build"}
)
STATE_VIEW_EXCLUDED_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".lock", ".egg-info"})

# Generated artifact exclusion — which tracked files are restored before a
# worker branch is merged into main. Narrower than StateView exclusions: only
# volatile cache directories that must not accumulate across merges.
GENERATED_ARTIFACT_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
)
GENERATED_ARTIFACT_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".pyd"})

# Critic evidence exclusion — untracked files excluded from the critic's
# worktree evidence window. Superset of generated artifact dirs, plus .git.
CRITIC_EVIDENCE_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".venv"}
)
CRITIC_EVIDENCE_EXCLUDED_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".pyd", ".lock"})
