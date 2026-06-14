from __future__ import annotations

from types import SimpleNamespace

from v2.coworld.commissioner.scheduling import (
    CommissionerMatchmakingConfig,
    leaderboard_neighbor_pairings,
    leaderboard_neighbors,
    leaderboard_ordered_policy_ids,
)


def _result(policy_id: str, *, score: float, rank: int, round_number: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        policy_version_id=policy_id,
        score=score,
        rank=rank,
        round_number=round_number,
    )


def test_leaderboard_neighbor_pairings_schedule_minimum_from_nearby_ranks() -> None:
    policy_ids = ["p0", "p1", "p2", "p3", "p4", "p5"]
    recent_results = [
        _result("p0", score=60, rank=1),
        _result("p1", score=50, rank=2),
        _result("p2", score=40, rank=3),
        _result("p3", score=30, rank=4),
        _result("p4", score=20, rank=5),
        _result("p5", score=10, rank=6),
    ]

    pairings = leaderboard_neighbor_pairings(
        policy_ids,
        recent_results,
        min_episodes_per_champion=4,
    )

    assert pairings[:4] == [
        ("p0", "p1"),
        ("p0", "p2"),
        ("p0", "p3"),
        ("p0", "p4"),
    ]
    assert pairings[8:12] == [
        ("p2", "p3"),
        ("p2", "p4"),
        ("p2", "p1"),
        ("p2", "p0"),
    ]
    assert pairings[-4:] == [
        ("p5", "p4"),
        ("p5", "p3"),
        ("p5", "p2"),
        ("p5", "p1"),
    ]

    anchored_counts = {policy_id: 0 for policy_id in policy_ids}
    for champion_id, _opponent_id in pairings:
        anchored_counts[champion_id] += 1
    assert anchored_counts == {policy_id: 4 for policy_id in policy_ids}


def test_leaderboard_neighbors_fill_missing_side_from_available_direction() -> None:
    ordered = ["p0", "p1", "p2", "p3", "p4"]

    assert leaderboard_neighbors(ordered, 0, 3) == ["p1", "p2", "p3"]
    assert leaderboard_neighbors(ordered, 4, 3) == ["p3", "p2", "p1"]
    assert leaderboard_neighbors(ordered, 2, 3) == ["p3", "p4", "p1"]


def test_leaderboard_order_uses_mean_score_then_unscored_stable_order() -> None:
    policy_ids = ["new-a", "scored-low", "scored-high", "new-b"]
    recent_results = [
        _result("scored-low", score=10, rank=2, round_number=1),
        _result("scored-high", score=20, rank=1, round_number=1),
        _result("scored-high", score=30, rank=1, round_number=2),
    ]

    assert leaderboard_ordered_policy_ids(policy_ids, recent_results) == [
        "scored-high",
        "scored-low",
        "new-a",
        "new-b",
    ]


def test_leaderboard_pairings_dedupe_duplicate_active_policy_ids() -> None:
    pairings = leaderboard_neighbor_pairings(
        ["p0", "p1", "p1", "p2"],
        [
            _result("p0", score=3, rank=1),
            _result("p1", score=2, rank=2),
            _result("p2", score=1, rank=3),
        ],
        min_episodes_per_champion=2,
    )

    assert pairings == [
        ("p0", "p1"),
        ("p0", "p2"),
        ("p1", "p2"),
        ("p1", "p0"),
        ("p2", "p1"),
        ("p2", "p0"),
    ]


def test_matchmaking_config_accepts_min_episodes_alias() -> None:
    cfg = CommissionerMatchmakingConfig.from_config({"minimum_episodes_per_champion": 6})

    assert cfg.min_episodes_per_champion == 6


def test_matchmaking_config_defaults_to_four_episodes_per_champion() -> None:
    cfg = CommissionerMatchmakingConfig.from_config({})

    assert cfg.min_episodes_per_champion == 4
