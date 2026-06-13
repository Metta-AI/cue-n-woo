#!/usr/bin/env python3
"""Cue-n-woo league commissioner: a concurrency-throttling WebSocket server.

The hosted platform connects to this commissioner (it is the WS *client*; we are
the server) at ``ws://<pod>:8080/round`` and drives a league via the coworld
commissioner protocol. We override the default platform commissioner for one
reason: cue-n-woo is served by a single GPU worker, so we must NOT let every
episode launch at once. This server:

  * caps simultaneous in-flight games to keep the worker near a target load
    (see scheduling.py), staggering episode starts;
  * guarantees every entrant policy plays at least one game;
  * otherwise applies straightforward average-score ranking.

Protocol (from coworld.commissioner.protocol + the platform driver):
  * one-shot requests answered on /round: schedule_rounds_request,
    rank_division_request, describe_division_request.
  * streaming round: platform sends round_start; we reply with schedule_episodes
    (a bounded first window); platform streams episode_result/episode_failed back;
    we release more episodes on each completion and finish with round_complete.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import websockets
from websockets.asyncio.server import serve

from coworld.commissioner.protocol import (
    DescribeDivisionRequest,
    DescribeDivisionResponse,
    DivisionDescription,
    DivisionRanking,
    EpisodeFailed,
    EpisodeRequest,
    EpisodeResult,
    RankDivisionRequest,
    RankDivisionResponse,
    RankingEntry,
    RoundComplete,
    RoundStart,
    ScheduleEpisodes,
    ScheduleRoundsRequest,
    ScheduleRoundsResponse,
    RoundSpec,
    RoundConfig,
    DivisionLeaderboardEntry,
)

from v2.coworld.commissioner.scheduling import (
    CommissionerSchedulingConfig,
    round_robin_pairings,
)

PORT = int(os.environ.get("COMMISSIONER_PORT", "8080"))


def _episode_for(pair: tuple[Any, Any], variant_id: str, seq: int) -> EpisodeRequest:
    a, b = pair
    return EpisodeRequest(
        request_id=f"ep-{seq}",
        variant_id=variant_id,
        policy_version_ids=[a, b],
        tags={"commissioner": "cue-n-woo-throttled"},
    )


def _plan_round(round_start: RoundStart) -> tuple[list[EpisodeRequest], CommissionerSchedulingConfig]:
    """Build the full episode list for a round (every entrant in >=1 game)."""
    cfg_dict = round_start.league.commissioner_config or {}
    # Pull the game timeout from the variant config when present so the load math
    # tracks the real per-game window.
    game_timeout = None
    if round_start.variants:
        game_timeout = round_start.variants[0].game_config.get("round_timeout_seconds")
    sched = CommissionerSchedulingConfig.from_config(cfg_dict, game_timeout_seconds=game_timeout)

    variant_id = round_start.variants[0].id if round_start.variants else "default"
    entrants = [m.policy_version_id for m in round_start.memberships]
    pairings = round_robin_pairings(entrants)
    episodes = [_episode_for(pair, variant_id, i) for i, pair in enumerate(pairings)]
    return episodes, sched


class RoundConductor:
    """Drives one streaming round, throttling in-flight episodes."""

    def __init__(self, websocket: Any, round_start: RoundStart) -> None:
        self._ws = websocket
        self._round_start = round_start
        self._episodes, self._sched = _plan_round(round_start)
        self._next_idx = 0
        self._in_flight = 0
        self._results: list[EpisodeResult] = []
        self._failures: list[EpisodeFailed] = []
        self._max_in_flight = self._sched.max_concurrent_games()
        self._stagger = self._sched.stagger_seconds()
        self._lock = asyncio.Lock()
        self._done = asyncio.Event()

    async def run(self) -> RoundComplete:
        # Prime the first window of episodes, staggered so their phases don't land
        # on the worker in lockstep.
        await self._fill_window(initial=True)
        if not self._episodes:
            return self._build_complete()
        await self._done.wait()
        return self._build_complete()

    async def _fill_window(self, *, initial: bool = False) -> None:
        async with self._lock:
            to_send: list[EpisodeRequest] = []
            while self._in_flight < self._max_in_flight and self._next_idx < len(self._episodes):
                to_send.append(self._episodes[self._next_idx])
                self._next_idx += 1
                self._in_flight += 1
        for offset, episode in enumerate(to_send):
            # Stagger starts. The first of an initial burst goes immediately; the
            # rest (and any refill) are spaced by the configured stagger.
            delay = 0.0 if (initial and offset == 0) else self._stagger * offset
            asyncio.create_task(self._send_after(episode, delay))

    async def _send_after(self, episode: EpisodeRequest, delay: float) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        await self._ws.send(json.dumps(ScheduleEpisodes(episodes=[episode]).to_json()))

    async def on_episode_finished(self) -> None:
        async with self._lock:
            self._in_flight -= 1
            finished = len(self._results) + len(self._failures)
            all_scheduled = self._next_idx >= len(self._episodes)
        if all_scheduled and finished >= len(self._episodes):
            self._done.set()
            return
        await self._fill_window()

    def record_result(self, result: EpisodeResult) -> None:
        self._results.append(result)

    def record_failure(self, failed: EpisodeFailed) -> None:
        self._failures.append(failed)

    def _build_complete(self) -> RoundComplete:
        # Average score per policy across the round; rank high-to-low.
        totals: dict[Any, float] = {}
        counts: dict[Any, int] = {}
        for result in self._results:
            for score in result.scores:
                totals[score.policy_version_id] = totals.get(score.policy_version_id, 0.0) + score.score
                counts[score.policy_version_id] = counts.get(score.policy_version_id, 0) + 1
        ranked = sorted(
            totals.keys(),
            key=lambda pid: totals[pid] / max(1, counts[pid]),
            reverse=True,
        )
        entries = [
            RankingEntry(
                policy_version_id=pid,
                rank=i + 1,
                score=totals[pid] / max(1, counts[pid]),
                result_metadata={"games": counts[pid]},
            )
            for i, pid in enumerate(ranked)
        ]
        division_id = self._round_start.divisions[0].id if self._round_start.divisions else None
        results = [DivisionRanking(division_id=division_id, rankings=entries)] if division_id else []
        return RoundComplete(results=results)


async def _handle_round(websocket: Any, round_start: RoundStart) -> None:
    conductor = RoundConductor(websocket, round_start)
    runner = asyncio.create_task(conductor.run())
    try:
        async for raw in websocket:
            message = _platform_message(raw)
            if isinstance(message, EpisodeResult):
                conductor.record_result(message)
                await conductor.on_episode_finished()
            elif isinstance(message, EpisodeFailed):
                conductor.record_failure(message)
                await conductor.on_episode_finished()
            if runner.done():
                break
    finally:
        complete = await runner
        await websocket.send(json.dumps(complete.to_json()))


def _platform_message(raw: str) -> Any:
    """Parse an inbound platform message (round flow uses platform->commissioner types)."""
    from coworld.commissioner.protocol import EpisodeAccepted, EpisodesRejected

    payload = json.loads(raw)
    t = payload.get("type")
    body = {k: v for k, v in payload.items() if k != "type"}
    if t == "episode_result":
        return EpisodeResult.model_validate(body)
    if t == "episode_failed":
        return EpisodeFailed.model_validate(body)
    if t == "episodes_accepted":
        return EpisodeAccepted.model_validate(body)
    if t == "episodes_rejected":
        return EpisodesRejected.model_validate(body)
    if t == "round_start":
        return RoundStart.model_validate(body)
    return payload


async def handler(websocket: Any) -> None:
    """One platform connection. First message selects the interaction type."""
    raw = await websocket.recv()
    payload = json.loads(raw)
    msg_type = payload.get("type")
    body = {k: v for k, v in payload.items() if k != "type"}

    if msg_type == "round_start":
        await _handle_round(websocket, RoundStart.model_validate(body))
    elif msg_type == "schedule_rounds_request":
        req = ScheduleRoundsRequest.model_validate(body)
        rounds = [
            RoundSpec(division_id=d.id, round_config=RoundConfig(), execution_backend="container")
            for d in req.divisions
        ]
        await websocket.send(json.dumps(ScheduleRoundsResponse(rounds=rounds).to_json()))
    elif msg_type == "rank_division_request":
        req = RankDivisionRequest.model_validate(body)
        await websocket.send(json.dumps(_rank_division(req).to_json()))
    elif msg_type == "describe_division_request":
        await websocket.send(json.dumps(_describe_division().to_json()))
    else:
        # Unknown opener: nothing to do.
        return


def _rank_division(req: RankDivisionRequest) -> RankDivisionResponse:
    # Average score across the division's round results, high-to-low.
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    names: dict[str, str | None] = {}
    versions: dict[str, set] = {}
    for r in req.round_results:
        totals[r.player_id] = totals.get(r.player_id, 0.0) + r.score
        counts[r.player_id] = counts.get(r.player_id, 0) + 1
        names[r.player_id] = r.player_name
        versions.setdefault(r.player_id, set()).add(r.policy_version_id)
    ranked = sorted(totals.keys(), key=lambda p: totals[p] / max(1, counts[p]), reverse=True)
    entries = [
        DivisionLeaderboardEntry(
            player_id=p,
            player_name=names.get(p),
            rank=i + 1,
            score=totals[p] / max(1, counts[p]),
            rounds_played=counts[p],
            policy_version_ids=versions.get(p, set()),
        )
        for i, p in enumerate(ranked)
    ]
    return RankDivisionResponse(rankings=entries)


def _describe_division() -> DescribeDivisionResponse:
    return DescribeDivisionResponse(
        description=DivisionDescription(
            round_structure="Round-robin: every entrant plays at least one two-player game.",
            scoring_mechanics="Average game score across the round; higher is better.",
            leaderboard_rules="Policies are ranked by mean score over all games played.",
        )
    )


async def main() -> None:
    async with serve(handler, "0.0.0.0", PORT):
        print(f"cue-n-woo commissioner listening on :{PORT}/round", flush=True)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
