from __future__ import annotations

import asyncio

import pytest

from v2.coworld.players import baseline
from v2.coworld.players.baseline import ClaudePolicy


def _state(phase: str) -> dict:
    return {
        "phase": phase,
        "slot": 0,
        "limits": {
            "max_answer_tokens": 12,
            "max_question_tokens": 1024,
            "judge_max_tokens": 128,
        },
        "me": {"judge": [], "proposals": [], "answers": []},
        "opponent_questions": [
            {"question": "Where would you go on vacation?"},
            {"question": "What color fits your mood?"},
            {"question": "What food sounds good?"},
        ],
        "public_questions": [[], []],
        "counts": [
            {"chats": 0, "proposals": 0, "answers": 0},
            {"chats": 0, "proposals": 0, "answers": 0},
        ],
    }


def test_decide_falls_back_when_bedrock_fails(monkeypatch) -> None:
    monkeypatch.setenv("COWORLD_PLAYER_WS_URL", "ws://game/player?slot=0&token=test")
    policy = ClaudePolicy()

    def fail(_messages):
        raise RuntimeError("bedrock unavailable")

    monkeypatch.setattr(policy, "_converse_with_retry", fail)

    action = policy.decide(_state("private_questions"))

    assert action["type"] == "ask"
    assert action["question"]
    assert policy.history[-1]["source"] == "fallback"


def test_fallback_actions_are_valid_shapes(monkeypatch) -> None:
    monkeypatch.setenv("COWORLD_PLAYER_WS_URL", "ws://game/player?slot=1&token=test")
    policy = ClaudePolicy()

    ask = policy.fallback_action(_state("private_questions"))
    propose = policy.fallback_action(_state("proposals"))
    answer = policy.fallback_action(_state("answers"))

    assert ask["type"] == "ask"
    assert ask["question"]
    assert propose["type"] == "propose"
    assert len(propose["proposals"]) == 3
    assert all(item["question"] and item["answer"] for item in propose["proposals"])
    assert answer["type"] == "answer"
    assert len(answer["answers"]) == 3
    assert all(item.strip() for item in answer["answers"])


def test_bedrock_client_uses_bounded_timeouts(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def converse(self, **_kwargs):
            raise RuntimeError("stop before network")

    def fake_client(service_name, *, region_name, config):
        captured["service_name"] = service_name
        captured["region_name"] = region_name
        captured["config"] = config
        return FakeClient()

    monkeypatch.setenv("BEDROCK_CONNECT_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("BEDROCK_READ_TIMEOUT_SECONDS", "7")
    monkeypatch.setattr(baseline.boto3, "client", fake_client)

    policy = ClaudePolicy()

    with pytest.raises(RuntimeError, match="stop before network"):
        policy._converse_with_retry([])

    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "us-east-1"
    assert captured["config"].connect_timeout == 3.0
    assert captured["config"].read_timeout == 7.0
    assert captured["config"].retries == {"max_attempts": 1}


def test_decide_with_timeout_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("COWORLD_PLAYER_WS_URL", "ws://game/player?slot=0&token=test")
    policy = ClaudePolicy()

    async def never_returns(_func, *_args, **_kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(baseline.asyncio, "to_thread", never_returns)

    action = asyncio.run(
        baseline.decide_with_timeout(
            policy, _state("private_questions"), None, timeout_seconds=0.001
        )
    )

    assert action["type"] == "ask"
    assert action["question"]
    assert policy.history[-1]["source"] == "fallback"
    assert "TimeoutError" in policy.history[-1]["error"]
