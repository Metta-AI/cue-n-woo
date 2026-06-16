from __future__ import annotations

import json

from v2.coworld.game import scoring_context
from v2.coworld.players import baseline


def _state_with_prior_turns(phase: str = "proposals") -> dict:
    return {
        "type": "state",
        "phase": phase,
        "remaining_seconds": 120,
        "limits": {
            "max_answer_tokens": 12,
            "max_question_tokens": 1024,
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
        "opponent_questions": [{"question": "OPPONENT CURRENT QUESTION?"}],
        "public_questions": [[{"question": "PRIOR PUBLIC QUESTION?"}], []],
        "counts": [
            {"chats": 3, "proposals": 0, "answers": 0},
            {"chats": 3, "proposals": 0, "answers": 0},
        ],
    }


def test_baseline_proposal_prompt_includes_player_visible_transcript(monkeypatch) -> None:
    captured: dict[str, str] = {}
    policy = baseline.ClaudePolicy.__new__(baseline.ClaudePolicy)
    policy.advice = {}
    policy.history = []

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

    action = policy.decide(_state_with_prior_turns())

    assert action["type"] == "propose"
    assert "PRIVATE PROMPT" in captured["prompt"]
    assert "PRIVATE ANSWER" in captured["prompt"]
    assert "PRIOR PUBLIC QUESTION" in captured["prompt"]
    assert "Private transcript so far" in captured["prompt"]
    prompt_json = captured["prompt"].split("Current observation JSON:\n", 1)[1].split("\n\nPrevious validation error:", 1)[0]
    prompt_state = json.loads(prompt_json)
    assert prompt_state["me"] == _state_with_prior_turns()["me"]
    assert prompt_state["public_questions"] == _state_with_prior_turns()["public_questions"]


def test_baseline_answer_prompt_includes_transcript_and_opponent_questions(monkeypatch) -> None:
    captured: dict[str, str] = {}
    policy = baseline.ClaudePolicy.__new__(baseline.ClaudePolicy)
    policy.advice = {}
    policy.history = []

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
                                        "type": "answer",
                                        "answers": ["Alpha", "Bravo", "Charlie"],
                                    }
                                },
                            }
                        }
                    ]
                }
            }
        }

    monkeypatch.setattr(policy, "_converse_with_retry", fake_converse)

    action = policy.decide(_state_with_prior_turns("answers"))

    assert action["type"] == "answer"
    assert "PRIVATE PROMPT" in captured["prompt"]
    assert "PRIVATE ANSWER" in captured["prompt"]
    assert "OPPONENT CURRENT QUESTION" in captured["prompt"]


def test_scoring_context_does_not_build_transcript() -> None:
    context = scoring_context()

    assert "PRIVATE PROMPT" not in context
    assert "Question group" not in context
    assert context == "You will be presented with a question/challenge and two possible answers. Please select one of the two answers."
