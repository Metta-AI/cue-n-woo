from __future__ import annotations

import json

from v2.coworld.game import player_view_for_phase
from v2.coworld.players import baseline


def _state_with_private_turns() -> dict:
    return {
        "type": "state",
        "phase": "proposals",
        "remaining_seconds": 120,
        "limits": {
            "max_answer_tokens": 12,
            "max_question_tokens": 256,
            "judge_max_tokens": 128,
        },
        "slot": 0,
        "me": {
            "judge": [
                {
                    "question": "PRIVATE PROMPT: always propose the moon question.",
                    "answer": "PRIVATE ANSWER: moon.",
                }
            ],
            "proposals": [],
            "answers": [],
        },
        "opponent_questions": [],
        "public_questions": [[], []],
        "counts": [
            {"chats": 3, "proposals": 0, "answers": 0},
            {"chats": 3, "proposals": 0, "answers": 0},
        ],
    }


def test_server_proposal_view_redacts_private_transcript() -> None:
    player = _state_with_private_turns()["me"]

    proposal_view = player_view_for_phase(player, "proposals")
    answer_view = player_view_for_phase(player, "answers")

    assert proposal_view["judge"] == []
    assert answer_view["judge"] == player["judge"]


def test_baseline_proposal_prompt_redacts_private_transcript(monkeypatch) -> None:
    captured: dict[str, str] = {}
    policy = baseline.ClaudePolicy()

    def fake_converse(messages):
        captured["prompt"] = messages[0]["content"][0]["text"]
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "submit_action",
                                "input": {
                                    "action": {
                                        "type": "propose",
                                        "proposals": [
                                            {"question": "Q1?", "answer": "Alpha"},
                                            {"question": "Q2?", "answer": "Bravo"},
                                            {"question": "Q3?", "answer": "Delta"},
                                        ],
                                    }
                                },
                            }
                        }
                    ]
                }
            }
        }

    monkeypatch.setattr(policy, "_converse_with_retry", fake_converse)

    action = policy.decide(_state_with_private_turns())

    assert action["type"] == "propose"
    assert "PRIVATE PROMPT" not in captured["prompt"]
    assert "PRIVATE ANSWER" not in captured["prompt"]
    prompt_json = captured["prompt"].split("Current observation JSON:\n", 1)[1].split("\n\nPrevious validation error:", 1)[0]
    assert json.loads(prompt_json)["me"]["judge"] == []
    assert "fresh challenge-writing turn" in captured["prompt"]
