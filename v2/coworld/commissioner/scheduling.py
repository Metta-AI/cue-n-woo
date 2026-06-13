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
    def from_config(config: dict[str, Any] | None, *, game_timeout_seconds: float | None = None) -> "CommissionerSchedulingConfig":
        config = config or {}
        defaults = CommissionerSchedulingConfig()
        return CommissionerSchedulingConfig(
            game_timeout_seconds=float(game_timeout_seconds if game_timeout_seconds is not None else config.get("game_timeout_seconds", defaults.game_timeout_seconds)),
            startup_buffer_seconds=float(config.get("startup_buffer_seconds", defaults.startup_buffer_seconds)),
            target_gpu_load=float(config.get("target_gpu_load", defaults.target_gpu_load)),
            worker_seconds_per_game=float(config.get("worker_seconds_per_game", defaults.worker_seconds_per_game)),
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


def round_robin_pairings(entrant_ids: list[Any]) -> list[tuple[Any, Any]]:
    """Every entrant paired at least once.

    For the 2-player game we pair (i, i+1) around the ring so every entrant appears
    in >=1 game and no entrant is left out, with an even spread of partners. With an
    odd count the last entrant wraps to the first (so it still gets a game).
    """
    n = len(entrant_ids)
    if n < 2:
        return []
    pairings = []
    for i in range(n):
        a = entrant_ids[i]
        b = entrant_ids[(i + 1) % n]
        pairings.append((a, b))
    # With n==2 the ring would duplicate the same pair twice; collapse it.
    if n == 2:
        return [pairings[0]]
    return pairings
