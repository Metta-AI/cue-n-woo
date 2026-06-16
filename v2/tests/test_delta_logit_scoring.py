from __future__ import annotations

import asyncio
import math
from typing import Any

from v2.coworld import game


class FakeScoringWorker:
    def __init__(self, *, steered: dict[str, float], unsteered: dict[str, float]) -> None:
        self.steered = steered
        self.unsteered = unsteered
        self.requests: list[dict[str, Any]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/choice-logprobs"
        self.requests.extend(payload["requests"])
        results = []
        for request in payload["requests"]:
            choices = request["choices"]
            if request["id"] == "steered":
                results.append({"id": "steered", "probabilities": [self.steered[choice] for choice in choices], "orderings": []})
            elif request["id"] == "unsteered":
                results.append({"id": "unsteered", "probabilities": [self.unsteered[choice] for choice in choices], "orderings": []})
            else:
                raise AssertionError(f"unexpected request id: {request['id']}")
        return {"results": results}


def test_delta_logit_scoring_renormalizes_base_points(monkeypatch) -> None:
    worker = FakeScoringWorker(
        steered={"alpha": 0.8, "bravo": 0.2},
        unsteered={"alpha": 0.5, "bravo": 0.5},
    )
    monkeypatch.setattr(game.state, "worker", worker)

    score = asyncio.run(game.answer_score("context", "question?", "alpha", "bravo", {"type": "text", "text": "x"}))

    assert math.isclose(score["secret_base_points"], 80.0)
    assert math.isclose(score["opponent_base_points"], 20.0)
    assert math.isclose(score["secret_base_points"] + score["opponent_base_points"], game.SCORE_SCALE)
    assert score["secret_bonus_points"] == game.BEAT_BONUS_POINTS
    assert score["opponent_bonus_points"] == 0.0
    assert [request["flas"]["flowtime"] for request in worker.requests] == [2.0, 0.0, 2.0, 0.0]


def test_delta_logit_scoring_discounts_unsteered_preference(monkeypatch) -> None:
    worker = FakeScoringWorker(
        steered={"alpha": 0.8, "bravo": 0.2},
        unsteered={"alpha": 0.8, "bravo": 0.2},
    )
    monkeypatch.setattr(game.state, "worker", worker)

    score = asyncio.run(game.answer_score("context", "question?", "alpha", "bravo", {"type": "text", "text": "x"}))

    assert math.isclose(score["average_secret_probability"], 0.5)
    assert math.isclose(score["average_opponent_probability"], 0.5)
    assert math.isclose(score["secret_base_points"], 50.0)
    assert math.isclose(score["opponent_base_points"], 50.0)
    assert score["secret_bonus_points"] == 0.0
    assert score["opponent_bonus_points"] == 0.0
