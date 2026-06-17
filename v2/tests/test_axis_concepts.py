from __future__ import annotations

import json
from pathlib import Path

import pytest

from v2.coworld import game


def test_bundled_concept_axes_are_valid() -> None:
    axes = game.load_concept_axes(None)

    assert len(axes) == 15
    assert len(axes["emotion"]) >= 10
    assert len(axes["object"]) >= 30
    assert len(axes["place"]) >= 30


def test_axis_combo_concept_samples_distinct_axes_with_seed(tmp_path: Path) -> None:
    axes_path = tmp_path / "axes"
    axes_path.mkdir()
    (axes_path / "persona.json").write_text(json.dumps(["noir detective", "field biologist"]), encoding="utf-8")
    (axes_path / "object.json").write_text(json.dumps(["broken clock", "old map"]), encoding="utf-8")
    (axes_path / "register.json").write_text(json.dumps(["terse", "poetic"]), encoding="utf-8")

    config = {
        "concept_type": "axis_combo",
        "concept_axes_path": str(axes_path),
        "concept_axis_names": ["persona", "object", "register"],
        "concept_axis_count": 2,
        "concept_seed": "same-round",
    }

    first = game.select_concept(config)
    second = game.select_concept(config)

    assert first == second
    assert first["type"] == "text"
    assert len(first["components"]) == 2
    assert len({component["axis"] for component in first["components"]}) == 2
    assert first["text"] == "; ".join(component["value"] for component in first["components"])


def test_hidden_judge_prompt_renders_axis_components_as_private_traits() -> None:
    prompt = game.hidden_judge_system_prompt(
        {
            "type": "text",
            "text": "1890s Vienna; brass compass",
            "components": [
                {"axis": "time", "value": "1890s Vienna"},
                {"axis": "object", "value": "brass compass"},
                {"axis": "register", "value": "formal and observant"},
            ],
        }
    )

    assert "Hidden traits:" in prompt
    assert "- Time period: 1890s Vienna" in prompt
    assert "- Favorite object: brass compass" in prompt
    assert "- Speaking style: formal and observant" in prompt
    assert "must not be revealed" in prompt


def test_forced_choice_prompt_prefers_persona_consistency_over_generic_quality() -> None:
    prompt = game.forced_choice_prompt(
        "Reference material:",
        "What would you carry?",
        "a brass compass",
        "a smartphone",
        {
            "type": "text",
            "components": [{"axis": "object", "value": "brass compass"}],
        },
    )

    assert "this hidden person would more naturally give" in prompt
    assert "more consistent with the hidden traits" in prompt
    assert "Do not choose based on writing quality alone" in prompt
    assert "Candidate A: a brass compass" in prompt
    assert "Candidate B: a smartphone" in prompt


def test_axis_combo_rejects_unknown_axis(tmp_path: Path) -> None:
    axes_path = tmp_path / "axes"
    axes_path.mkdir()
    (axes_path / "persona.json").write_text(json.dumps(["noir detective"]), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown concept axes"):
        game.select_concept(
            {
                "concept_type": "axis_combo",
                "concept_axes_path": str(axes_path),
                "concept_axis_names": ["persona", "missing"],
            }
        )
