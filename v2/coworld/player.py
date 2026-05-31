#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os

import websockets

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


async def main() -> None:
    url = os.environ["COWORLD_PLAYER_WS_URL"]
    async with websockets.connect(url) as ws:
        asked = False
        proposed = False
        answered = False
        async for raw in ws:
            state = json.loads(raw)
            if state.get("type") == "error":
                continue
            phase = state.get("phase")
            if phase == "private_questions" and not asked:
                for question in ASKS:
                    await ws.send(json.dumps({"type": "ask", "question": question}))
                    await ws.recv()
                asked = True
            elif phase == "proposals" and not proposed:
                limit = int(state.get("limits", {}).get("max_answer_tokens", 12))
                proposals = [
                    {"question": proposal["question"], "answer": truncate_to_token_limit(proposal["answer"], limit)}
                    for proposal in PROPOSALS
                ]
                await ws.send(json.dumps({"type": "propose", "proposals": proposals}))
                proposed = True
            elif phase == "blind_answers" and not answered:
                limit = int(state.get("limits", {}).get("max_answer_tokens", 12))
                guesses = [truncate_to_token_limit(guess(q.get("question", "")), limit) for q in state.get("opponent_questions", [])]
                while len(guesses) < 3:
                    guesses.append("unknown")
                await ws.send(json.dumps({"type": "answer", "answers": guesses[:3]}))
                answered = True
            elif phase == "reveal":
                return


def guess(question: str) -> str:
    lower = question.lower()
    if "capital of france" in lower:
        return "Paris"
    if "sky" in lower:
        return "blue"
    if "leap year" in lower:
        return "366"
    return "unknown"


if __name__ == "__main__":
    asyncio.run(main())
