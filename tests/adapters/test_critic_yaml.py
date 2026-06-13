"""Tests verifying critic and referee YAML contain contract-consistency rules."""

from pathlib import Path

from forge.adapters.registry import AdapterRegistry

ADAPTERS_DIR = Path(__file__).parents[2] / "adapters"


def _load_prompt(adapter_name: str) -> str:
    registry = AdapterRegistry()
    registry.load(ADAPTERS_DIR)
    return registry.get(adapter_name).prompt_template


def test_critic_prompt_contains_contract_consistency_rule() -> None:
    """Critic prompt must instruct checking revision items against contract constraints."""
    prompt = _load_prompt("critic")
    assert "check it against the language plugin guidance and constraints" in prompt


def test_critic_prompt_forbids_contradicting_contract_constraints() -> None:
    """Critic prompt must forbid requesting changes that contradict contract constraints."""
    prompt = _load_prompt("critic")
    assert "Never request a change that contradicts the contract constraints" in prompt


def test_referee_prompt_must_override_critic_contradicting_contract() -> None:
    """Referee prompt must mandate overriding a critic that contradicts the contract."""
    prompt = _load_prompt("referee")
    assert "A critic that contradicts the contract is always wrong" in prompt


def test_referee_prompt_specifies_override_action() -> None:
    """Referee prompt must specify the corrective action when critic contradicts contract."""
    prompt = _load_prompt("referee")
    assert "MUST override the critic" in prompt
