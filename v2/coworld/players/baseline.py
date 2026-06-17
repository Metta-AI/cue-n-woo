#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
import websockets
from websockets.exceptions import ConnectionClosed

from v2.coworld.harness import game_rules_for_policy, truncate_to_token_limit


DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "us-east-1"
MAX_ATTEMPTS = 3
BEDROCK_ATTEMPTS = 5
DEFAULT_BEDROCK_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS = 20.0
DEFAULT_DECISION_TIMEOUT_SECONDS = 30.0

FALLBACK_QUESTIONS = [
    "What kind of place feels most comfortable to you?",
    "Describe a small object you would keep nearby.",
    "What color best fits your mood today?",
    "How would you spend a quiet afternoon?",
    "Name a food that sounds appealing right now.",
    "What sort of music would suit this moment?",
]

FALLBACK_PROPOSALS = [
    {
        "question": "What kind of place feels most comfortable to you?",
        "answer": "a quiet library",
    },
    {"question": "What color best fits your mood today?", "answer": "soft blue"},
    {"question": "Name a food that sounds appealing right now.", "answer": "warm soup"},
    {"question": "How would you spend a quiet afternoon?", "answer": "reading slowly"},
    {
        "question": "What sort of music would suit this moment?",
        "answer": "gentle piano",
    },
    {
        "question": "Describe a small object you would keep nearby.",
        "answer": "a brass key",
    },
]

FALLBACK_ANSWERS = [
    "quiet library",
    "soft blue",
    "warm soup",
    "reading slowly",
    "gentle piano",
    "brass key",
]


SUBMIT_TOOL = {
    "toolSpec": {
        "name": "submit_action",
        "description": "Submit the next game action.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "object",
                        "additionalProperties": True,
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["ask", "propose", "answer"],
                            },
                            "question": {"type": "string"},
                            "proposals": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["question", "answer"],
                                    "properties": {
                                        "question": {"type": "string"},
                                        "answer": {"type": "string"},
                                    },
                                },
                            },
                            "answers": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                },
            }
        },
    }
}


def positive_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


class ClaudePolicy:
    """LLM player harness. The model makes every real decision via submit_action.

    ``advice`` maps a phase name to optional, non-binding guidance the player
    wants to suggest to the model (e.g. starter questions). It is injected into
    the prompt as explicitly optional; the model is free to ignore it. The
    baseline player passes no advice.
    """

    def __init__(self, advice: dict[str, str] | None = None) -> None:
        self.model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID") or os.environ.get(
            "BEDROCK_MODEL", DEFAULT_MODEL_ID
        )
        self.region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_REGION
        )
        self.bedrock_connect_timeout_seconds = positive_float_env(
            "BEDROCK_CONNECT_TIMEOUT_SECONDS", DEFAULT_BEDROCK_CONNECT_TIMEOUT_SECONDS
        )
        self.bedrock_read_timeout_seconds = positive_float_env(
            "BEDROCK_READ_TIMEOUT_SECONDS", DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS
        )
        self.decision_timeout_seconds = positive_float_env(
            "CUE_N_WOO_DECISION_TIMEOUT_SECONDS", DEFAULT_DECISION_TIMEOUT_SECONDS
        )
        self.client = None
        self.advice = advice or {}
        self.history: list[dict[str, Any]] = []
        fallback_seed = os.environ.get("COWORLD_PLAYER_SEED") or os.environ.get(
            "COWORLD_PLAYER_WS_URL", "cue-n-woo-fallback"
        )
        self.random = random.Random(fallback_seed)
        self.question_offset = self.random.randrange(len(FALLBACK_QUESTIONS))

    def phase_advice_text(self, phase: str | None) -> str:
        suggestion = self.advice.get(phase or "")
        if not suggestion:
            return ""
        return (
            "\n\nOptional suggestion (not a requirement — you are making the real decision, "
            f"so use, adapt, or ignore this as you see fit):\n{suggestion}"
        )

    def decide(
        self, state: dict[str, Any], validation_error: str | None = None
    ) -> dict[str, Any]:
        prompt_state = state_for_decision_prompt(state)
        transcript_notes = private_transcript_notes(prompt_state)
        judge_max_tokens = int(state.get("limits", {}).get("judge_max_tokens", 128))
        phase_instructions = action_instructions_for_phase(prompt_state.get("phase"))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"{game_rules_for_policy()}\n\n"
                            f"Judge response limit: the judge's generated answer to each private question is limited to {judge_max_tokens} output tokens. "
                            "If you bundle many subquestions, the judge may run out of tokens before answering all of them. "
                            "Treat missing or cut-off text as unavailable information, not as a deliberate answer.\n\n"
                            f"Private transcript so far:\n{transcript_notes}\n\n"
                            f"Current observation JSON:\n{json.dumps(compact_state(prompt_state), ensure_ascii=True)}\n\n"
                            f"Previous validation error: {validation_error or 'none'}\n\n"
                            "Call submit_action with exactly one legal next action. "
                            f"{phase_instructions} "
                            "Do not output prose outside the tool call."
                            f"{self.phase_advice_text(state.get('phase'))}"
                        )
                    }
                ],
            }
        ]
        try:
            response = self._converse_with_retry(messages)
            for block in response["output"]["message"]["content"]:
                tool_use = block.get("toolUse")
                if tool_use and tool_use["name"] == "submit_action":
                    action = tool_use["input"]["action"]
                    self.history.append(
                        {
                            "state_phase": state.get("phase"),
                            "action": action,
                            "source": "bedrock",
                        }
                    )
                    return action
            raise RuntimeError("Claude did not call submit_action.")
        except Exception as exc:
            action = self.fallback_action(state)
            self.history.append(
                {
                    "state_phase": state.get("phase"),
                    "action": action,
                    "source": "fallback",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return action

    def _converse_with_retry(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.client is None:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    connect_timeout=self.bedrock_connect_timeout_seconds,
                    read_timeout=self.bedrock_read_timeout_seconds,
                    retries={"max_attempts": 1},
                ),
            )
        for attempt in range(BEDROCK_ATTEMPTS):
            try:
                return self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    toolConfig={
                        "tools": [SUBMIT_TOOL],
                        "toolChoice": {"tool": {"name": "submit_action"}},
                    },
                    inferenceConfig={"maxTokens": 1024},
                )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if (
                    code
                    not in {
                        "ServiceUnavailableException",
                        "ThrottlingException",
                        "TooManyRequestsException",
                    }
                    or attempt == BEDROCK_ATTEMPTS - 1
                ):
                    raise
                time.sleep(2**attempt)
            except BotoCoreError:
                raise
        raise RuntimeError("Bedrock retry loop exited unexpectedly.")

    def fallback_action(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = state.get("phase")
        limits = state.get("limits", {})
        answer_limit = int(limits.get("max_answer_tokens", 12))

        if phase == "private_questions":
            asked_count = len(state.get("me", {}).get("judge", []))
            question = FALLBACK_QUESTIONS[
                (self.question_offset + asked_count) % len(FALLBACK_QUESTIONS)
            ]
            return {"type": "ask", "question": question}

        if phase == "proposals":
            proposals = self.random.sample(
                FALLBACK_PROPOSALS, k=min(3, len(FALLBACK_PROPOSALS))
            )
            while len(proposals) < 3:
                proposals.append(self.random.choice(FALLBACK_PROPOSALS))
            return {
                "type": "propose",
                "proposals": [
                    {
                        "question": proposal["question"],
                        "answer": truncate_to_token_limit(
                            proposal["answer"], answer_limit
                        ),
                    }
                    for proposal in proposals[:3]
                ],
            }

        if phase == "answers":
            opponent_questions = state.get("opponent_questions") or [{}, {}, {}]
            fallback_answers = self.random.sample(
                FALLBACK_ANSWERS, k=len(FALLBACK_ANSWERS)
            )
            answers = [
                truncate_to_token_limit(
                    fallback_answers[idx % len(fallback_answers)], answer_limit
                )
                for idx, _question in enumerate(opponent_questions[:3])
            ]
            while len(answers) < 3:
                answer = fallback_answers[len(answers) % len(fallback_answers)]
                answers.append(truncate_to_token_limit(answer, answer_limit))
            return {"type": "answer", "answers": answers}

        return {
            "type": "answer",
            "answers": ["quiet library", "soft blue", "warm soup"],
        }


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": state.get("phase"),
        "remaining_seconds": state.get("remaining_seconds"),
        "limits": state.get("limits"),
        "slot": state.get("slot"),
        "me": state.get("me"),
        "opponent_questions": state.get("opponent_questions"),
        "public_questions": state.get("public_questions"),
        "counts": state.get("counts"),
    }


def state_for_decision_prompt(state: dict[str, Any]) -> dict[str, Any]:
    prompt_state = json.loads(json.dumps(state))
    if prompt_state.get("phase") == "proposals":
        prompt_state.setdefault("me", {})["judge"] = []
    return prompt_state


def action_instructions_for_phase(phase: str | None) -> str:
    if phase == "private_questions":
        return (
            "For private_questions, submit one ask action. "
            "When asking private questions, first use the private transcript above: do not repeat the same topic or ask another near-duplicate personality/preference survey unless you are deliberately disambiguating a previous answer. "
            "Bundled questions are allowed, but each bundle should cover genuinely new dimensions or focused follow-ups on specific surprising details from the judge's previous answers."
        )
    if phase == "proposals":
        return (
            "For proposals, submit exactly three proposals. "
            "Treat this as a fresh challenge-writing turn: do not rely on, mention, summarize, or infer from previous private questions, previous judge answers, or any earlier conversation turns."
        )
    if phase == "answers":
        return "For answers, submit exactly three answers."
    return "Submit the next legal action for the current phase."


def private_transcript_notes(state: dict[str, Any]) -> str:
    turns = state.get("me", {}).get("judge", [])
    if not turns:
        return "No private questions have been answered yet."
    notes = []
    for idx, turn in enumerate(turns, start=1):
        notes.append(
            f"Q{idx}: {turn.get('question', '')}\nA{idx}: {turn.get('answer', '')}"
        )
    return "\n\n".join(notes)


async def decide_with_timeout(
    policy: ClaudePolicy,
    state: dict[str, Any],
    validation_error: str | None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else policy.decision_timeout_seconds
    )
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(policy.decide, state, validation_error),
            timeout=timeout,
        )
    except TimeoutError:
        action = policy.fallback_action(state)
        policy.history.append(
            {
                "state_phase": state.get("phase"),
                "action": action,
                "source": "fallback",
                "error": f"TimeoutError: decision exceeded {timeout:.1f}s",
            }
        )
        return action


async def run(advice: dict[str, str] | None = None) -> None:
    """Drive an LLM player over the game WebSocket.

    ``advice`` is optional per-phase guidance to suggest to the model; the
    baseline player passes none. Other players (e.g. kyle) reuse this with their
    own non-binding suggestions.
    """
    url = os.environ["COWORLD_PLAYER_WS_URL"]
    policy = ClaudePolicy(advice=advice)
    pending_error: str | None = None
    # The server closes the socket as soon as the final action triggers scoring,
    # so a send/recv can race that shutdown. That close IS the end-of-game signal
    # for the last actor, not a failure: exit cleanly instead of crashing.
    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            async for raw in ws:
                state = json.loads(raw)
                if state.get("type") == "error":
                    pending_error = state.get("error", "unknown validation error")
                    continue
                if state.get("phase") == "reveal":
                    return
                for _ in range(MAX_ATTEMPTS):
                    action = await decide_with_timeout(policy, state, pending_error)
                    pending_error = None
                    await ws.send(json.dumps(action))
                    reply = json.loads(await ws.recv())
                    if reply.get("type") != "error":
                        state = reply
                        break
                    pending_error = reply.get("error", "unknown validation error")
    except ConnectionClosed:
        return


async def main() -> None:
    await run()


if __name__ == "__main__":
    asyncio.run(main())
