from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from v2.coworld import game


def test_bundled_concept_axes_are_valid() -> None:
    axes = game.load_concept_axes(None)

    assert len(axes) == 15
    assert len(axes["emotion"]) >= 10
    assert len(axes["object"]) >= 30
    assert len(axes["place"]) >= 30


def test_bundled_concept_axes_do_not_repeat_nontrivial_words() -> None:
    ignore = {"a", "all", "above", "an", "and", "of", "the", "to", "until"}

    for name, values in game.load_concept_axes(None).items():
        counts: Counter[str] = Counter()
        for value in values:
            counts.update(word for word in re.findall(r"[a-z]+", value.lower()) if word not in ignore and not word.endswith("s"))

        repeated = {word: count for word, count in counts.items() if count > 1}
        assert repeated == {}, f"{name} repeats words across axis items: {repeated}"


def test_axis_combo_concept_samples_distinct_axes_with_seed(tmp_path: Path) -> None:
    axes_path = tmp_path / "axes"
    axes_path.mkdir()
    (axes_path / "persona.json").write_text(json.dumps(["noir detective", "field biologist"]), encoding="utf-8")
    (axes_path / "object_motif.json").write_text(json.dumps(["broken clock", "old map"]), encoding="utf-8")
    (axes_path / "register.json").write_text(json.dumps(["terse", "poetic"]), encoding="utf-8")

    config = {
        "concept_type": "axis_combo",
        "concept_axes_path": str(axes_path),
        "concept_axis_names": ["persona", "object_motif", "register"],
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
