"""Tests for parse_plan: JSON parsing, artifact validation, dependency wiring, and adapter defaulting."""

# pyright: reportPrivateUsage=false

import json

import pytest

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.core.models import WorkSpec
from forge.parsers.plan import parse_plan


def _make_registry(*adapter_names: str) -> AdapterRegistry:
    registry = AdapterRegistry()
    for name in adapter_names:
        registry._adapters[name] = AdapterSpec(
            name=name,
            description=f"{name} adapter",
            tools=[],
            prompt_template="do: {objective}",
        )
    return registry


def _plan_json(tasks: list[dict[str, object]]) -> str:
    return json.dumps({"tasks": tasks})


_REGISTRY = _make_registry("coding", "document")


def test_parse_plan_parses_artifact_field() -> None:
    """parse_plan() sets the artifact field on the resulting WorkSpec."""
    raw = _plan_json([{
        "objective": "write parser",
        "success_condition": "parser passes tests",
        "adapter": "coding",
        "artifact": "codebase",
        "depends_on": [],
    }])
    requests = parse_plan(raw, _REGISTRY)
    assert len(requests) == 1
    spec = requests[0].spec
    assert isinstance(spec, WorkSpec)
    assert spec.artifact == "codebase"


def test_parse_plan_raises_on_missing_artifact() -> None:
    """parse_plan() raises ValueError with a clear message when a task has no artifact field."""
    raw = _plan_json([{
        "objective": "write parser",
        "success_condition": "done",
        "adapter": "coding",
        "depends_on": [],
    }])
    with pytest.raises(ValueError, match="artifact"):
        parse_plan(raw, _REGISTRY)


def test_parse_plan_wires_dependencies_correctly() -> None:
    """parse_plan() sets dependency IDs on tasks according to depends_on indices."""
    raw = _plan_json([
        {
            "objective": "task A",
            "success_condition": "A done",
            "adapter": "coding",
            "artifact": "codebase",
            "depends_on": [],
        },
        {
            "objective": "task B",
            "success_condition": "B done",
            "adapter": "coding",
            "artifact": "codebase",
            "depends_on": [0],
        },
    ])
    requests = parse_plan(raw, _REGISTRY)
    a, b = requests
    assert not a.dependencies
    assert a.id in b.dependencies


def test_parse_plan_handles_markdown_fenced_json() -> None:
    """parse_plan() correctly parses a JSON plan wrapped in a markdown code fence."""
    inner = _plan_json([{
        "objective": "write docs",
        "success_condition": "docs exist",
        "adapter": "document",
        "artifact": "docs",
        "depends_on": [],
    }])
    raw = f"```json\n{inner}\n```"
    requests = parse_plan(raw, _REGISTRY)
    assert len(requests) == 1
    spec = requests[0].spec
    assert isinstance(spec, WorkSpec)
    assert spec.adapter == "document"
    assert spec.artifact == "docs"


def test_parse_plan_defaults_unknown_adapter_to_coding() -> None:
    """parse_plan() substitutes 'coding' when the task adapter is not in the registry."""
    raw = _plan_json([{
        "objective": "do something",
        "success_condition": "done",
        "adapter": "nonexistent",
        "artifact": "codebase",
        "depends_on": [],
    }])
    requests = parse_plan(raw, _REGISTRY)
    spec = requests[0].spec
    assert isinstance(spec, WorkSpec)
    assert spec.adapter == "coding"
