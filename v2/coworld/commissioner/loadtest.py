#!/usr/bin/env python3
"""Measure realistic per-game worker load to calibrate the commissioner.

Simulates the worker traffic of a full cue-n-woo game and runs many concurrent
games against a live worker, reporting:
  * worker-busy seconds attributable to one game (to set
    ``worker_seconds_per_game`` in scheduling.py / commissioner_config), and
  * how aggregate throughput holds up as concurrency rises.

Usage:  python -m v2.coworld.commissioner.loadtest [worker_url] [concurrency]

A "game" issues the worker calls a real 2-player episode makes:
  * 6 generate calls (3 private questions x 2 players), <=128 tokens
  * 12 choice-logprobs calls (3 proposals x 2 players x 2 orderings) at finalize
The generate calls are spread out (players think between turns); the scoring
calls burst together at the end.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request

WORKER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7870"
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 8

CONCEPTS = [
    "warm supportive therapist, gentle reflective language",
    "noir detective narration, smoky atmosphere",
    "terse technical documentation, precise API wording",
    "sports commentator excitement, play-by-play energy",
    "victorian letter, formal ornate prose",
    "pirate captain, nautical slang, boisterous",
    "zen minimalism, haiku-like, seasonal imagery",
    "cyberpunk street tech, neon, jargon",
]


def _post(path: str, requests: list[dict]) -> tuple[list[dict], float]:
    body = json.dumps({"requests": requests}).encode()
    t = time.perf_counter()
    resp = json.load(
        urllib.request.urlopen(
            urllib.request.Request(WORKER + path, data=body, headers={"Content-Type": "application/json"}),
            timeout=600,
        )
    )
    return resp["results"], time.perf_counter() - t


async def post(path: str, requests: list[dict]) -> tuple[list[dict], float]:
    return await asyncio.to_thread(_post, path, requests)


async def one_game(game_idx: int) -> float:
    """Run one game's worth of worker calls; return total worker wall-seconds it used."""
    concept = {"type": "text", "text": CONCEPTS[game_idx % len(CONCEPTS)]}
    busy = 0.0
    # 6 private-question generations (one at a time per player turn).
    for q in range(6):
        _, dt = await post(
            "/generate",
            [{
                "id": f"g{game_idx}-{q}",
                "prompt": "Answer the question directly and helpfully.\n\nQuestion: Tell me about your ideal day.",
                "concept": concept,
                "flas": {"flowtime": 2, "steps": 3},
                "sampling": {"max_tokens": 128, "max_prompt_tokens": 1024, "temperature": 0.7},
            }],
        )
        busy += dt
    # Finalize scoring burst: 6 proposals x 2 orderings = 12 choice-logprobs.
    for s in range(12):
        _, dt = await post(
            "/choice-logprobs",
            [{
                "id": f"s{game_idx}-{s}",
                "prompt": "Choose the answer that best fits.\nQuestion: favorite season?",
                "choices": ["a quiet autumn evening", "bright summer noon"],
                "concept": concept,
                "flas": {"flowtime": 2, "steps": 3},
                "ordering": {"mode": "given_order"},
            }],
        )
        busy += dt
    return busy


async def main() -> None:
    print(f"worker: {WORKER}  concurrency: {CONCURRENCY}")
    print("warming up (1 game)...")
    warm = await one_game(0)
    print(f"  single-game worker-busy seconds (serial, no contention): {warm:.1f}s\n")

    print(f"running {CONCURRENCY} concurrent games...")
    t0 = time.perf_counter()
    busy_times = await asyncio.gather(*[one_game(i) for i in range(CONCURRENCY)])
    wall = time.perf_counter() - t0
    total_busy = sum(busy_times)
    print(f"  wall time for {CONCURRENCY} concurrent games: {wall:.1f}s")
    print(f"  mean per-game worker-busy (with contention):   {total_busy / CONCURRENCY:.1f}s")
    print(f"  aggregate worker-busy seconds:                 {total_busy:.1f}s")
    # If the worker were a single serial queue, wall ~= total_busy. Batching makes
    # wall < total_busy. The ratio tells us the effective parallel speedup.
    print(f"  effective speedup vs serial (total_busy/wall): {total_busy / wall:.2f}x")
    print()
    print("CALIBRATION: set worker_seconds_per_game to the single-game number above")
    print("(the serial-busy figure), which is the load one game adds to the worker.")


if __name__ == "__main__":
    asyncio.run(main())
