#!/usr/bin/env python3
"""Deterministic stub player: no LLM, no AWS, runs fully offline.

Used by certification so the smoke test exercises the whole game flow without a
model or credentials. It asks precommitted questions, submits precommitted
proposals, and — unlike a real player — ALWAYS answers every opponent question,
ignoring the question text entirely. Answers are drawn from a backup list (and
fall back to generated gibberish), so a stub never declines and never trips the
minimum-length answer rule.
"""
from __future__ import annotations

import asyncio
import json
import os

import websockets
from websockets.exceptions import ConnectionClosed

from v2.coworld.harness import truncate_to_token_limit


ASKS = [
    "Describe your answer style in one sentence.",
    "Name three words that fit your current style.",
    "Give one short example sentence in your current style.",
]

PROPOSALS = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What color is a clear daytime sky?", "answer": "blue"},
    {"question": "How many days are in a leap year?", "answer": "366"},
]

# Backup answers, used in order regardless of the question. All satisfy the
# minimum non-space character rule so the stub never produces an invalid answer.
BACKUP_ANSWERS = ["banana", "purple", "mountain", "seventeen", "quietly", "lantern"]


def stub_answer(index: int) -> str:
    """An answer for the opponent's question at ``index``, ignoring its text.

    Cycles through BACKUP_ANSWERS; past the list it emits deterministic gibberish
    (always >= 3 non-space characters), so the stub answers every question.
    """
    if index < len(BACKUP_ANSWERS):
        return BACKUP_ANSWERS[index]
    return f"zzx{index}"


async def main() -> None:
    url = os.environ["COWORLD_PLAYER_WS_URL"]
    asked = False
    proposed = False
    answered = False
    # The server closes the socket the moment the final action triggers scoring,
    # so a send/recv can race that shutdown. That close is the end-of-game signal
    # for the last actor, not a failure: exit cleanly instead of crashing.
    try:
        async with websockets.connect(url) as ws:
            async for raw in ws:
                state = json.loads(raw)
                if state.get("type") == "error":
                    continue
                phase = state.get("phase")
                limit = int(state.get("limits", {}).get("max_answer_tokens", 12))
                if phase == "private_questions" and not asked:
                    for question in ASKS:
                        await ws.send(json.dumps({"type": "ask", "question": question}))
                        await ws.recv()
                    asked = True
                elif phase == "proposals" and not proposed:
                    proposals = [
                        {"question": p["question"], "answer": truncate_to_token_limit(p["answer"], limit)}
                        for p in PROPOSALS
                    ]
                    await ws.send(json.dumps({"type": "propose", "proposals": proposals}))
                    proposed = True
                elif phase == "answers" and not answered:
                    opponent_questions = state.get("opponent_questions", [])
                    answers = [truncate_to_token_limit(stub_answer(i), limit) for i in range(len(opponent_questions))]
                    await ws.send(json.dumps({"type": "answer", "answers": answers}))
                    answered = True
                elif phase == "reveal":
                    return
    except ConnectionClosed:
        return


if __name__ == "__main__":
    asyncio.run(main())
