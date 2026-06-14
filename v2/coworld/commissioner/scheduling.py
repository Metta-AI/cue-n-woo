"""Load-aware scheduling math for the cue-n-woo commissioner.

The cue-n-woo game is served by a single GPU FLAS/Gemma worker. A naive league
would launch every episode at once and swamp the worker, so this module decides
*how many* games may be in flight and *how far apart* to start them so the worker
stays near a target utilization with headroom for other traffic.

All knobs come from ``commissioner_config`` (see ``CommissionerSchedulingConfig``)
so they can be tuned per-deployment without rebuilding the image. Defaults are
derived from measured single-L40S throughput; recalibrate with the load test in
``v2/coworld/commissioner/loadtest.py`` if the worker hardware changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommissionerSchedulingConfig:
    # Wall-clock a single game may run before its hard timeout (game_config
    # round_timeout_seconds). Policies do not fire worker requests instantly, so
    # we reserve a startup buffer and only count the remaining window as usable.
    game_timeout_seconds: float = 300.0
    startup_buffer_seconds: float = 60.0

    # Fraction of the worker we let the tournament consume, leaving headroom for
    # other (e.g. public) traffic and burst slack.
    target_gpu_load: float = 0.80

    # Measured worker-busy seconds a single game costs over its lifetime
    # (generation + scoring), on the target GPU. This is the number to recalibrate
    # from loadtest.py when hardware changes.
    worker_seconds_per_game: float = 25.0

    # Hard floor/ceiling on simultaneous in-flight games, regardless of the math.
    min_in_flight: int = 1
    max_in_flight: int = 64

    @staticmethod
    def from_config(
        config: dict[str, Any] | None,
        *,
        game_timeout_seconds: float | None = None,
    ) -> "CommissionerSchedulingConfig":
        config = config or {}
        defaults = CommissionerSchedulingConfig()
        return CommissionerSchedulingConfig(
            game_timeout_seconds=float(
                game_timeout_seconds
                if game_timeout_seconds is not None
                else config.get("game_timeout_seconds", defaults.game_timeout_seconds)
            ),
            startup_buffer_seconds=float(
                config.get("startup_buffer_seconds", defaults.startup_buffer_seconds)
            ),
            target_gpu_load=float(config.get("target_gpu_load", defaults.target_gpu_load)),
            worker_seconds_per_game=float(
                config.get("worker_seconds_per_game", defaults.worker_seconds_per_game)
            ),
            min_in_flight=int(config.get("min_in_flight", defaults.min_in_flight)),
            max_in_flight=int(config.get("max_in_flight", defaults.max_in_flight)),
        )

    @property
    def usable_window_seconds(self) -> float:
        """Wall-clock window we actually schedule worker load into, per game."""
        return max(1.0, self.game_timeout_seconds - self.startup_buffer_seconds)

    def max_concurrent_games(self) -> int:
        """Most games allowed in flight to keep worker load near target.

        A single game keeps the worker busy ``worker_seconds_per_game`` out of its
        ``usable_window_seconds`` of wall time -> a per-game duty cycle. To hold the
        worker at ``target_gpu_load`` we allow target_load / duty_cycle games at
        once. (Duty cycle < target means even one game is under target, so we still
        allow at least min_in_flight.)
        """
        duty_cycle = self.worker_seconds_per_game / self.usable_window_seconds
        if duty_cycle <= 0:
            allowed = self.max_in_flight
        else:
            allowed = math.floor(self.target_gpu_load / duty_cycle)
        return max(self.min_in_flight, min(self.max_in_flight, allowed))

    def stagger_seconds(self) -> float:
        """Delay between consecutive episode starts.

        Spreading starts avoids N games hitting the same phase (and thus the worker)
        in lockstep. Spacing the in-flight set evenly across a game's usable window
        spreads their bursts: window / concurrency.
        """
        return self.usable_window_seconds / self.max_concurrent_games()


@dataclass(frozen=True)
class CommissionerMatchmakingConfig:
    # Each champion anchors this many two-player episodes against nearby
    # leaderboard neighbors. The policy may appear in additional games as another
    # champion's neighbor.
    min_episodes_per_champion: int = 4

    @staticmethod
    def from_config(config: dict[str, Any] | None) -> "CommissionerMatchmakingConfig":
        config = config or {}
        defaults = CommissionerMatchmakingConfig()
        return CommissionerMatchmakingConfig(
            min_episodes_per_champion=int(
                config.get(
                    "min_episodes_per_champion",
                    config.get(
                        "minimum_episodes_per_champion",
                        defaults.min_episodes_per_champion,
                    ),
                )
            ),
        )

    def __post_init__(self) -> None:
        if self.min_episodes_per_champion < 1:
            raise ValueError("min_episodes_per_champion must be at least 1")


def leaderboard_neighbor_pairings(
    champion_ids: list[Any],
    recent_results: list[Any],
    *,
    min_episodes_per_champion: int,
) -> list[tuple[Any, Any]]:
    """Pair every champion with nearby policies on the current leaderboard.

    For a middle leaderboard entry and N requested episodes, schedule against
    roughly N/2 policies directly below and N/2 directly above it. Near the top or
    bottom, fill the missing side from the available direction, which naturally
    overschedules some policies as neighbors.
    """
    n = len(champion_ids)
    if n < 2:
        return []
    ordered_ids = leaderboard_ordered_policy_ids(champion_ids, recent_results)
    pairings: list[tuple[Any, Any]] = []
    for champion_index, champion_id in enumerate(ordered_ids):
        opponents = leaderboard_neighbors(
            ordered_ids,
            champion_index,
            min_episodes_per_champion,
        )
        pairings.extend((champion_id, opponent_id) for opponent_id in opponents)
    return pairings


def leaderboard_ordered_policy_ids(policy_ids: list[Any], recent_results: list[Any]) -> list[Any]:
    """Order active policies by the commissioner leaderboard rule.

    Cue-n-woo ranks by mean score. The round-start message includes recent round
    results rather than a materialized leaderboard, so this reconstructs a stable
    ordering from recent mean score and uses mean rank as a tie-breaker. Policies
    without recent results stay eligible but sort after scored policies.
    """
    unique_policy_ids = _dedupe(policy_ids)
    original_index = {policy_id: index for index, policy_id in enumerate(unique_policy_ids)}
    scores: dict[Any, list[float]] = {policy_id: [] for policy_id in unique_policy_ids}
    ranks: dict[Any, list[float]] = {policy_id: [] for policy_id in unique_policy_ids}
    active_ids = set(unique_policy_ids)

    for result in recent_results:
        policy_id = getattr(result, "policy_version_id", None)
        if policy_id not in active_ids:
            continue
        scores[policy_id].append(float(getattr(result, "score")))
        ranks[policy_id].append(float(getattr(result, "rank")))

    def sort_key(policy_id: Any) -> tuple[int, float, float, int]:
        policy_scores = scores[policy_id]
        if not policy_scores:
            return (1, 0.0, math.inf, original_index[policy_id])
        mean_score = sum(policy_scores) / len(policy_scores)
        mean_rank = sum(ranks[policy_id]) / max(1, len(ranks[policy_id]))
        return (0, -mean_score, mean_rank, original_index[policy_id])

    return sorted(unique_policy_ids, key=sort_key)


def leaderboard_neighbors(
    ordered_policy_ids: list[Any],
    champion_index: int,
    count: int,
) -> list[Any]:
    if count < 1:
        raise ValueError("neighbor count must be at least 1")
    if not 0 <= champion_index < len(ordered_policy_ids):
        raise IndexError("champion index out of range")

    max_count = min(count, len(ordered_policy_ids) - 1)
    below_target = (count + 1) // 2
    above_target = count // 2

    selected: list[Any] = []
    selected.extend(_below(ordered_policy_ids, champion_index, start=1, count=below_target))
    selected.extend(_above(ordered_policy_ids, champion_index, start=1, count=above_target))

    if len(selected) < max_count:
        selected.extend(
            _below(ordered_policy_ids, champion_index, start=below_target + 1, count=max_count)
        )
    if len(selected) < max_count:
        selected.extend(
            _above(ordered_policy_ids, champion_index, start=above_target + 1, count=max_count)
        )
    return _dedupe(selected)[:max_count]


def _below(
    ordered_policy_ids: list[Any],
    champion_index: int,
    *,
    start: int,
    count: int,
) -> list[Any]:
    if count <= 0:
        return []
    first = champion_index + start
    return ordered_policy_ids[first : first + count]


def _above(
    ordered_policy_ids: list[Any],
    champion_index: int,
    *,
    start: int,
    count: int,
) -> list[Any]:
    if count <= 0:
        return []
    first = champion_index - start
    last = max(-1, first - count)
    return [ordered_policy_ids[index] for index in range(first, last, -1) if index >= 0]


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
