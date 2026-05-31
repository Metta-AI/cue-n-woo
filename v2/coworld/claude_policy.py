#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
import websockets

from v2.coworld.harness import game_rules_for_policy


DEFAULT_MODEL_ID = "us.anthropic.claude-opus-4-8"
DEFAULT_REGION = "us-east-1"
MAX_ATTEMPTS = 3
BEDROCK_ATTEMPTS = 5


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
                            "type": {"type": "string", "enum": ["ask", "propose", "answer"]},
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


class ClaudePolicy:
    def __init__(self) -> None:
        self.model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", DEFAULT_MODEL_ID)
        self.region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        self.history: list[dict[str, Any]] = []

    def decide(self, state: dict[str, Any], validation_error: str | None = None) -> dict[str, Any]:
        transcript_notes = private_transcript_notes(state)
        charlie_max_tokens = int(state.get("limits", {}).get("charlie_max_tokens", 128))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"{game_rules_for_policy()}\n\n"
                            f"Charlie response limit: Charlie's generated answer to each private question is limited to {charlie_max_tokens} output tokens. "
                            "If you bundle many subquestions, Charlie may run out of tokens before answering all of them. "
                            "Treat missing or cut-off text as unavailable information, not as a deliberate answer.\n\n"
                            f"Private transcript so far:\n{transcript_notes}\n\n"
                            f"Current observation JSON:\n{json.dumps(compact_state(state), ensure_ascii=True)}\n\n"
                            f"Previous validation error: {validation_error or 'none'}\n\n"
                            "Call submit_action with exactly one legal next action. "
                            "For private_questions, submit one ask action. "
                            "When asking private questions, first use the private transcript above: do not repeat the same topic or ask another near-duplicate personality/preference survey unless you are deliberately disambiguating a previous answer. "
                            "Bundled questions are allowed, but each bundle should cover genuinely new dimensions or focused follow-ups on specific surprising details from Charlie's previous answers. "
                            "For proposals, submit exactly three proposals. "
                            "Each proposal's hidden answer should be a specific answer Charlie already gave or a narrow inference from the transcript, not a generic factual answer. "
                            "For blind_answers, submit exactly three answers. "
                            "Do not output prose outside the tool call."
                        )
                    }
                ],
            }
        ]
        response = self._converse_with_retry(messages)
        for block in response["output"]["message"]["content"]:
            tool_use = block.get("toolUse")
            if tool_use and tool_use["name"] == "submit_action":
                action = tool_use["input"]["action"]
                self.history.append({"state_phase": state.get("phase"), "action": action})
                return action
        raise RuntimeError("Claude did not call submit_action.")

    def _converse_with_retry(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        for attempt in range(BEDROCK_ATTEMPTS):
            try:
                return self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    toolConfig={"tools": [SUBMIT_TOOL], "toolChoice": {"tool": {"name": "submit_action"}}},
                    inferenceConfig={"maxTokens": 1024},
                )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"ServiceUnavailableException", "ThrottlingException", "TooManyRequestsException"} or attempt == BEDROCK_ATTEMPTS - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Bedrock retry loop exited unexpectedly.")


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": state.get("phase"),
        "remaining_seconds": state.get("remaining_seconds"),
        "limits": state.get("limits"),
        "role": state.get("role"),
        "me": state.get("me"),
        "opponent_questions": state.get("opponent_questions"),
        "public_questions": state.get("public_questions"),
        "counts": state.get("counts"),
    }


def private_transcript_notes(state: dict[str, Any]) -> str:
    turns = state.get("me", {}).get("charlie", [])
    if not turns:
        return "No private questions have been answered yet."
    notes = []
    for idx, turn in enumerate(turns, start=1):
        notes.append(f"Q{idx}: {turn.get('question', '')}\nA{idx}: {turn.get('answer', '')}")
    return "\n\n".join(notes)


async def main() -> None:
    url = os.environ["COWORLD_PLAYER_WS_URL"]
    policy = ClaudePolicy()
    pending_error: str | None = None
    async with websockets.connect(url, ping_interval=None) as ws:
        async for raw in ws:
            state = json.loads(raw)
            if state.get("type") == "error":
                pending_error = state.get("error", "unknown validation error")
                continue
            if state.get("phase") == "reveal":
                return
            for _ in range(MAX_ATTEMPTS):
                action = await asyncio.to_thread(policy.decide, state, pending_error)
                pending_error = None
                await ws.send(json.dumps(action))
                reply = json.loads(await ws.recv())
                if reply.get("type") != "error":
                    state = reply
                    break
                pending_error = reply.get("error", "unknown validation error")


if __name__ == "__main__":
    asyncio.run(main())
