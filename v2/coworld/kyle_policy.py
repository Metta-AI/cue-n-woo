#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os

import websockets

from v2.coworld.harness import truncate_to_token_limit


PHASE_1_IDEAS = [
    "Tell me about a philosophy",
    "If you were a DnD character, what class would you be?",
    "You're planning a vacation. Where do you go? What do you do?",
]

PHASE_2_IDEAS = [
    "If you were a DnD character, what class would you be?",
    "Where is your ideal vacation location?",
    "What is your favorite color?",
]


def infer_answer(question: str, transcript: list[dict[str, str]]) -> str:
    text = " ".join(turn.get("answer", "") for turn in transcript)
    lower_question = question.lower()
    lower_text = text.lower()
    if "dnd" in lower_question or "class" in lower_question:
        for cls in ["wizard", "bard", "rogue", "fighter", "cleric", "paladin", "ranger", "druid", "warlock", "sorcerer", "barbarian", "monk"]:
            if cls in lower_text:
                return cls
    if "vacation" in lower_question or "location" in lower_question or "where" in lower_question:
        for place in ["beach", "mountain", "forest", "city", "paris", "tokyo", "rome", "island", "museum", "castle"]:
            if place in lower_text:
                return place
    if "color" in lower_question:
        for color in ["blue", "red", "green", "black", "white", "gold", "silver", "purple", "pink", "gray", "orange", "yellow"]:
            if color in lower_text:
                return color
    return "unknown"


def suggested_proposals(transcript: list[dict[str, str]], max_answer_tokens: int) -> list[dict[str, str]]:
    proposals = []
    for question in PHASE_2_IDEAS:
        answer = truncate_to_token_limit(infer_answer(question, transcript), max_answer_tokens)
        proposals.append({"question": question, "answer": answer or "unknown"})
    return proposals


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
            limit = int(state.get("limits", {}).get("max_answer_tokens", 12))
            if phase == "private_questions" and not asked:
                for question in PHASE_1_IDEAS:
                    await ws.send(json.dumps({"type": "ask", "question": question}))
                    await ws.recv()
                asked = True
            elif phase == "proposals" and not proposed:
                await ws.send(json.dumps({"type": "propose", "proposals": suggested_proposals(state["me"]["charlie"], limit)}))
                proposed = True
            elif phase == "blind_answers" and not answered:
                guesses = [
                    truncate_to_token_limit(infer_answer(question.get("question", ""), state["me"]["charlie"]), limit)
                    for question in state.get("opponent_questions", [])
                ]
                while len(guesses) < 3:
                    guesses.append("unknown")
                await ws.send(json.dumps({"type": "answer", "answers": guesses[:3]}))
                answered = True
            elif phase == "reveal":
                return


if __name__ == "__main__":
    asyncio.run(main())
