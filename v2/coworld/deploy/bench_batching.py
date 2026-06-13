#!/usr/bin/env python3
"""Benchmark + correctness check for cross-concept batching (generate + scoring).

Run against a worker started from this repo (default http://127.0.0.1:7870):

    .venv/bin/python v2/coworld/deploy/bench_batching.py

For BOTH generate and choice-logprobs it:
1. Correctness: sends N requests each with a DIFFERENT concept in ONE call and
   confirms the worker batched them (generate reports batch_size == N) with
   sane, non-degenerate outputs.
2. Throughput: sweeps batch sizes and reports work/sec, so we can see how
   throughput scales with batch on this GPU.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

WORKER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7870"

CONCEPTS = [
    "warm supportive therapist, gentle reflective language",
    "noir detective narration, smoky atmosphere, clipped cynicism",
    "terse technical documentation, precise API wording",
    "sports commentator excitement, play-by-play energy",
    "victorian letter, formal ornate prose, polite restraint",
    "pirate captain, nautical slang, boisterous",
    "zen minimalism, haiku-like, seasonal imagery",
    "cyberpunk street tech, neon, jargon",
]
GEN_PROMPT = "Answer the question directly and helpfully.\n\nQuestion: Describe your ideal afternoon."
MAX_TOKENS = 128


def post(path: str, requests: list[dict]) -> tuple[dict, float]:
    body = json.dumps({"requests": requests}).encode()
    t = time.perf_counter()
    resp = json.load(
        urllib.request.urlopen(
            urllib.request.Request(WORKER + path, data=body, headers={"Content-Type": "application/json"}),
            timeout=600,
        )
    )
    return resp, time.perf_counter() - t


def gen_call(batch_size: int) -> dict:
    reqs = [
        {
            "id": f"r{i}",
            "prompt": GEN_PROMPT,
            "concept": {"type": "text", "text": CONCEPTS[i % len(CONCEPTS)]},
            "flas": {"flowtime": 2, "steps": 3},
            "sampling": {"max_tokens": MAX_TOKENS, "max_prompt_tokens": 1024, "temperature": 0.7},
        }
        for i in range(batch_size)
    ]
    resp, wall = post("/generate", reqs)
    results = resp["results"]
    out = sum(r["output_tokens"] for r in results)
    return {
        "batch": batch_size,
        "reported_batch_size": results[0].get("batch_size"),
        "wall_s": round(wall, 2),
        "out_tok": out,
        "tok_per_s": round(out / wall, 1),
        "all_nonempty": all(r["output_tokens"] > 0 for r in results),
        "sample": results[0]["text"][:70],
    }


def score_call(batch_size: int) -> dict:
    reqs = [
        {
            "id": f"s{i}",
            "prompt": "Choose the answer that best fits.\nQuestion: favorite season?",
            "choices": ["a quiet autumn evening", "bright summer noon"],
            "concept": {"type": "text", "text": CONCEPTS[i % len(CONCEPTS)]},
            "flas": {"flowtime": 2, "steps": 3},
            "ordering": {"mode": "given_order"},
        }
        for i in range(batch_size)
    ]
    resp, wall = post("/choice-logprobs", reqs)
    results = resp["results"]
    return {
        "batch": batch_size,
        "wall_s": round(wall, 3),
        "req_per_s": round(batch_size / wall, 1),
        "all_valid": all(abs(sum(r["probabilities"]) - 1.0) < 1e-3 for r in results),
        "sample_probs": [round(p, 3) for p in results[0]["probabilities"]],
    }


def main() -> None:
    print(f"worker: {WORKER}\n")

    print("=== GENERATE correctness: 4 different concepts in one call ===")
    r = gen_call(4)
    print(json.dumps(r, indent=2))
    print("CROSS-CONCEPT GENERATE:", "OK" if r["reported_batch_size"] == 4 and r["all_nonempty"] else "FAILED", "\n")

    print("=== GENERATE throughput sweep ===")
    print(f"{'batch':>6} {'reported':>9} {'wall_s':>8} {'out_tok':>8} {'tok/s':>8}")
    for b in (1, 2, 4, 8, 16, 32):
        try:
            r = gen_call(b)
            print(f"{r['batch']:>6} {r['reported_batch_size']:>9} {r['wall_s']:>8} {r['out_tok']:>8} {r['tok_per_s']:>8}")
        except Exception as exc:  # noqa: BLE001
            print(f"{b:>6}  ERROR: {exc}")

    print("\n=== SCORING correctness: 4 different concepts in one call ===")
    r = score_call(4)
    print(json.dumps(r, indent=2))
    print("CROSS-CONCEPT SCORING:", "OK" if r["all_valid"] else "FAILED", "\n")

    print("=== SCORING throughput sweep ===")
    print(f"{'batch':>6} {'wall_s':>8} {'req/s':>8}")
    for b in (1, 2, 4, 8, 16, 32, 64):
        try:
            r = score_call(b)
            print(f"{r['batch']:>6} {r['wall_s']:>8} {r['req_per_s']:>8}")
        except Exception as exc:  # noqa: BLE001
            print(f"{b:>6}  ERROR: {exc}")


if __name__ == "__main__":
    main()
