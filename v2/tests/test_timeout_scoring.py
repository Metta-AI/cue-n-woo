from __future__ import annotations

from v2.coworld import game


def _player(*, judge: int = 0, proposals: int = 0, answers: int = 0) -> dict:
    return {
        "judge": [{"question": f"Q{i}?", "answer": f"A{i}"} for i in range(judge)],
        "proposals": [{"question": f"C{i}?", "answer": f"P{i}"} for i in range(proposals)],
        "answers": [f"R{i}" for i in range(answers)],
    }


def test_timeout_penalizes_only_player_blocking_private_questions(monkeypatch) -> None:
    monkeypatch.setitem(game.CONFIG, "private_questions_per_player", 3)
    players = [_player(judge=3), _player(judge=1)]

    scores, rows, penalties = game.timeout_scores(players, "private_questions")

    assert scores == [0.0, game.INACTIVE_TIMEOUT_PENALTY]
    assert rows == []
    assert penalties["inactive_slots"] == [1]
    assert penalties["neutral_slots"] == [0]
    assert penalties["phase"] == "private_questions"


def test_timeout_penalizes_only_player_blocking_proposals(monkeypatch) -> None:
    monkeypatch.setitem(game.CONFIG, "challenge_questions_per_player", 3)
    players = [_player(judge=3, proposals=3), _player(judge=3, proposals=0)]

    scores, rows, penalties = game.timeout_scores(players, "proposals")

    assert scores == [0.0, game.INACTIVE_TIMEOUT_PENALTY]
    assert rows == []
    assert penalties["inactive_slots"] == [1]
    assert penalties["neutral_slots"] == [0]
    assert penalties["phase"] == "proposals"


def test_timeout_penalizes_only_player_blocking_answers(monkeypatch) -> None:
    monkeypatch.setitem(game.CONFIG, "challenge_questions_per_player", 3)
    players = [
        _player(judge=3, proposals=3, answers=3),
        _player(judge=3, proposals=3, answers=1),
    ]

    scores, rows, penalties = game.timeout_scores(players, "answers")

    assert scores == [0.0, game.INACTIVE_TIMEOUT_PENALTY]
    assert rows == []
    assert penalties["inactive_slots"] == [1]
    assert penalties["neutral_slots"] == [0]
    assert penalties["phase"] == "answers"


def test_timeout_penalizes_all_players_when_no_one_finishes_blocking_phase(monkeypatch) -> None:
    monkeypatch.setitem(game.CONFIG, "challenge_questions_per_player", 3)
    players = [_player(judge=3, proposals=1), _player(judge=3, proposals=2)]

    scores, rows, penalties = game.timeout_scores(players, "proposals")

    assert scores == [game.INACTIVE_TIMEOUT_PENALTY, game.INACTIVE_TIMEOUT_PENALTY]
    assert rows == []
    assert penalties["inactive_slots"] == [0, 1]
    assert penalties["neutral_slots"] == []


def test_timeout_penalty_is_not_applied_after_all_actions_are_ready() -> None:
    players = [
        _player(judge=3, proposals=3, answers=3),
        _player(judge=3, proposals=3, answers=3),
    ]

    scores, rows, penalties = game.timeout_scores(players, "ready_to_score")

    assert scores == [0.0, 0.0]
    assert rows == []
    assert penalties["inactive_slots"] == []
    assert penalties["neutral_slots"] == [0, 1]
