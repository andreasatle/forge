import re
from pathlib import Path

import pytest

from forge.core.models import SchedulerState
from forge.core.persistence import save_run


@pytest.fixture()
def state() -> SchedulerState:
    return SchedulerState(northstar="test goal")


def test_save_run_creates_directory(state: SchedulerState, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    save_run(state, runs_dir)
    assert runs_dir.is_dir()


def test_save_run_writes_valid_json(state: SchedulerState, tmp_path: Path) -> None:
    path = save_run(state, tmp_path / "runs")
    loaded = SchedulerState.model_validate_json(path.read_text())
    assert loaded.northstar == state.northstar


def test_save_run_filename_matches_format(state: SchedulerState, tmp_path: Path) -> None:
    path = save_run(state, tmp_path / "runs")
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}\.json", path.name)


def test_save_run_returns_correct_path(state: SchedulerState, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    path = save_run(state, runs_dir)
    assert path.parent == runs_dir
    assert path.exists()


def test_save_run_multiple_calls_produce_separate_files(state: SchedulerState, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    path1 = save_run(state, runs_dir)
    path2 = save_run(state, runs_dir)
    assert path1 != path2
    assert len(list(runs_dir.iterdir())) == 2
