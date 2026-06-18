#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
import zlib
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from v2.coworld.harness import public_hints, simple_token_count, validate_natural_keyboard_answer


ROOT = Path(__file__).resolve().parent
HTTP_USER_AGENT = "cue-n-woo-coworld/0.1"
GAME_HOST = os.environ.get("COGAME_HOST", "0.0.0.0")
GAME_PORT = int(os.environ.get("COGAME_PORT", "8080"))
SCORE_SCALE = 100.0
BEAT_BONUS_POINTS = 10.0
DUPLICATE_ANSWER_PENALTY_POINTS = 10.0
INACTIVE_TIMEOUT_PENALTY = -100.0
DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_BEDROCK_REGION = "us-east-1"
BEDROCK_ATTEMPTS = 5
DEFAULT_BEDROCK_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_BEDROCK_MAX_BACKOFF_SECONDS = 15.0
# Per-socket cap for a single broadcast send. Keeps one wedged (open-but-not-
# reading) client from stalling timer_loop and the round deadline.
WEBSOCKET_SEND_TIMEOUT_SECONDS = 5.0
TRANSIENT_BEDROCK_ERROR_CODES = {
    "InternalServerException",
    "ModelErrorException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
}
CHOICE_TOOL = {
    "toolSpec": {
        "name": "choose_answer",
        "description": "Choose which candidate answer better matches the hidden judge.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["choice"],
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": ["A", "B"],
                    }
                },
            }
        },
    }
}
CONCEPT_CONFIG_KEYS = {
    "concept_type",
    "concept_index",
    "specific_concept",
    "concept_seed",
    "concept_list_path",
    "concept_axes_path",
    "concept_axis_names",
    "concept_axis_count",
    "random_concept_tokens",
    "random_concept_scale",
    "random_concept_normalize",
    "reveal_concept_to_clients",
    "include_concept_in_results",
}


def read_data(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        req = Request(uri, headers={"User-Agent": HTTP_USER_AGENT})
        with urlopen(req, timeout=30) as resp:
            return resp.read()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
    if parsed.scheme == "s3":
        import boto3

        return boto3.client("s3").get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
    if parsed.scheme == "":
        return Path(uri).read_bytes()
    raise ValueError(f"Unsupported URI for read_data: {uri}")


def write_data(uri: str, data: bytes | str, *, content_type: str, http_method: Literal["POST", "PUT"] = "PUT") -> None:
    if isinstance(data, str):
        data = data.encode("utf-8")
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        req = Request(uri, data=data, method=http_method, headers={"Content-Type": content_type, "User-Agent": HTTP_USER_AGENT})
        with urlopen(req, timeout=60):
            return
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if parsed.scheme == "":
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    raise ValueError(f"Unsupported URI for write_data: {uri}")


def load_config() -> dict[str, Any]:
    uri = os.environ.get("COGAME_CONFIG_URI")
    if not uri:
        return {
            "tokens": ["alice-token", "bob-token"],
            "players": [{"name": "Alice"}, {"name": "Bob"}],
            "bedrock_model_id": DEFAULT_BEDROCK_MODEL_ID,
            "round_timeout_seconds": 600,
        }
    return json.loads(read_data(uri).decode("utf-8"))


CONFIG = load_config()
TOKENS = CONFIG["tokens"]
PLAYERS = CONFIG.get("players", [{"name": "Alice"}, {"name": "Bob"}])
RESULTS_URI = os.environ.get("COGAME_RESULTS_URI", str(ROOT / "results.json"))
REPLAY_URI = os.environ.get("COGAME_SAVE_REPLAY_URI", str(ROOT / "replay.json.z"))
REPLAY_LOAD_URI = os.environ.get("COGAME_LOAD_REPLAY_URI")
REPLAY_MODE = REPLAY_LOAD_URI is not None


def load_concept_list(path: str | None) -> list[str]:
    data_path = Path(path) if path else ROOT / "data" / "concepts.json"
    return json.loads(data_path.read_text())


CONCEPTS = load_concept_list(CONFIG.get("concept_list_path"))


def load_concept_axes(path: str | None) -> dict[str, list[str]]:
    data_path = Path(path) if path else ROOT / "data" / "concept_axes"
    if data_path.is_file():
        raw_axes = json.loads(data_path.read_text())
        axes = {str(name): values for name, values in raw_axes.items()}
    else:
        axes = {}
        for axis_path in sorted(data_path.glob("*.json")):
            axes[axis_path.stem] = json.loads(axis_path.read_text())
    for name, values in axes.items():
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"Concept axis {name!r} must be a non-empty JSON array of strings.")
    if not axes:
        raise ValueError(f"No concept axes found at {data_path}.")
    return axes


def positive_float_config(config: dict[str, Any], name: str, default: float) -> float:
    try:
        parsed = float(config.get(name, os.environ.get(name.upper(), default)))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class BedrockRefereeClient:
    def __init__(self, config: dict[str, Any], remaining_seconds: Callable[[], float] | None = None):
        self.model_id = (
            config.get("bedrock_model_id")
            or os.environ.get("BEDROCK_CLAUDE_MODEL_ID")
            or os.environ.get("BEDROCK_MODEL")
            or DEFAULT_BEDROCK_MODEL_ID
        )
        self.region = (
            config.get("bedrock_region")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_BEDROCK_REGION
        )
        self.stub = bool(config.get("stub_bedrock", config.get("stub_worker", False)))
        self.connect_timeout_seconds = positive_float_config(
            config, "bedrock_connect_timeout_seconds", DEFAULT_BEDROCK_CONNECT_TIMEOUT_SECONDS
        )
        self.read_timeout_seconds = positive_float_config(
            config, "bedrock_read_timeout_seconds", DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS
        )
        self.max_backoff_seconds = positive_float_config(
            config, "bedrock_max_backoff_seconds", DEFAULT_BEDROCK_MAX_BACKOFF_SECONDS
        )
        self.remaining_seconds = remaining_seconds
        self.client = None

    def generate_judge_answer(self, question: str, concept: dict[str, Any]) -> str:
        if self.stub:
            return f"stub answer ({len(question)} chars)"
        prompt = (
            f"{hidden_judge_system_prompt(concept)}\n\n"
            "You are being asked a private question. Answer naturally as this hidden person would. "
            "Be direct and concise, and do not name or reveal the hidden trait list.\n\n"
            f"Question: {model_safe_text(question)}"
        )
        response = self._converse_with_retry(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inference_config={
                "maxTokens": judge_max_tokens(),
                "temperature": float(CONFIG.get("temperature", 0.7)),
            },
        )
        text_parts = [
            block["text"]
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if "text" in block
        ]
        answer = "\n".join(text_parts).strip()
        if not answer:
            raise RuntimeError("Claude returned an empty judge answer.")
        return answer

    def choose_answer(self, prompt: str, label_a: str, label_b: str, *, sample_index: int) -> str:
        if self.stub:
            return "A"
        response = self._converse_with_retry(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            tool_config={
                "tools": [CHOICE_TOOL],
                "toolChoice": {"tool": {"name": "choose_answer"}},
            },
            inference_config={
                "maxTokens": 64,
                "temperature": float(CONFIG.get("scoring_temperature", CONFIG.get("temperature", 0.7))),
            },
        )
        for block in response.get("output", {}).get("message", {}).get("content", []):
            tool_use = block.get("toolUse")
            if tool_use and tool_use.get("name") == "choose_answer":
                choice = str(tool_use.get("input", {}).get("choice", "")).strip().upper()
                if choice in {"A", "B"}:
                    return choice
        raise RuntimeError(f"Claude did not choose {label_a!r} or {label_b!r}.")

    def _client(self) -> Any:
        if self.client is None:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    connect_timeout=self.connect_timeout_seconds,
                    read_timeout=self.read_timeout_seconds,
                    retries={"max_attempts": 1},
                ),
            )
        return self.client

    def _converse_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        inference_config: dict[str, Any],
        tool_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if tool_config is not None:
            kwargs["toolConfig"] = tool_config
        attempt = 0
        last_error: BaseException | None = None
        while True:
            remaining = self._remaining_seconds()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Bedrock referee retry budget exhausted at the game deadline.") from last_error
            try:
                return self._client().converse(**kwargs)
            except ClientError as exc:
                last_error = exc
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in TRANSIENT_BEDROCK_ERROR_CODES:
                    raise
            except BotoCoreError as exc:
                last_error = exc
            attempt += 1
            if self.remaining_seconds is None and attempt >= BEDROCK_ATTEMPTS:
                raise RuntimeError("Bedrock retry attempts exhausted.") from last_error
            self._sleep_before_retry(attempt, remaining, last_error)
        raise RuntimeError("Bedrock retry loop exited unexpectedly.")

    def _remaining_seconds(self) -> float | None:
        if self.remaining_seconds is None:
            return None
        return max(0.0, float(self.remaining_seconds()))

    def _sleep_before_retry(self, attempt: int, remaining: float | None, exc: BaseException | None) -> None:
        backoff = min(float(2 ** min(attempt - 1, 6)), self.max_backoff_seconds)
        if remaining is not None:
            backoff = min(backoff, max(0.0, remaining))
        if backoff <= 0:
            raise TimeoutError("Bedrock referee retry budget exhausted at the game deadline.") from exc
        print(
            f"Retrying Bedrock referee call after {type(exc).__name__ if exc else 'unknown error'} "
            f"(attempt {attempt + 1}, sleeping {backoff:.1f}s).",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(backoff)


def empty_player() -> dict[str, Any]:
    return {"judge": [], "proposals": [], "answers": []}


def player_view_for_phase(player: dict[str, Any], phase: str) -> dict[str, Any]:
    view = json.loads(json.dumps(player))
    if phase == "proposals":
        view["judge"] = []
    return view


def judge_max_tokens() -> int:
    return int(CONFIG.get("judge_max_tokens", CONFIG.get("max_output_tokens", 128)))


class EpisodeState:
    def __init__(self) -> None:
        # Players are addressed by slot index (0, 1, ...) matching config["players"].
        self.players = [empty_player() for _ in PLAYERS]
        self.connections: dict[int, WebSocket] = {}
        self.global_connections: set[WebSocket] = set()
        self.results: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.started_at = time.time()
        self.deadline = self.started_at + float(CONFIG.get("round_timeout_seconds", 600))
        self.done = False
        self.hidden_concept = select_concept(CONFIG)
        self.referee = BedrockRefereeClient(CONFIG, remaining_seconds=self.remaining_time_seconds)
        self.lock = asyncio.Lock()

    def phase(self) -> str:
        if self.results is not None:
            return "reveal"
        if any(len(player["judge"]) < int(CONFIG.get("private_questions_per_player", 3)) for player in self.players):
            return "private_questions"
        if any(len(player["proposals"]) < int(CONFIG.get("challenge_questions_per_player", 3)) for player in self.players):
            return "proposals"
        if any(len(player["answers"]) < int(CONFIG.get("challenge_questions_per_player", 3)) for player in self.players):
            return "answers"
        return "ready_to_score"

    def remaining_seconds(self) -> int:
        return max(0, int(self.deadline - time.time()))

    def remaining_time_seconds(self) -> float:
        return max(0.0, self.deadline - time.time())

    def view(self, slot: int | None = None, *, global_view: bool = False) -> dict[str, Any]:
        phase = self.phase()
        payload: dict[str, Any] = {
            "type": "state",
            "phase": phase,
            "remaining_seconds": self.remaining_seconds(),
            "limits": {
                "max_answer_tokens": int(CONFIG.get("max_answer_tokens", 12)),
                "max_question_tokens": int(CONFIG.get("max_question_tokens", 256)),
                "judge_max_tokens": judge_max_tokens(),
            },
            "harness": public_hints(),
            # Per-player aggregate counts, indexed by slot.
            "counts": [
                {
                    "chats": len(player["judge"]),
                    "proposals": len(player["proposals"]),
                    "answers": len(player["answers"]),
                }
                for player in self.players
            ],
            # Public challenge questions per player, indexed by slot.
            "public_questions": [
                [{"question": proposal["question"]} for proposal in player["proposals"]]
                for player in self.players
            ],
            "results": public_results(
                self.results,
                reveal_concept=phase == "reveal" and bool(CONFIG.get("reveal_concept_to_clients", False)),
            ),
            "done": self.done,
        }
        if not global_view and slot is not None and 0 <= slot < len(self.players):
            other = 1 - slot if len(self.players) == 2 else slot
            payload.update(
                {
                    "slot": slot,
                    "me": player_view_for_phase(self.players[slot], phase),
                    "opponent_questions": [
                        {"question": proposal["question"]} for proposal in self.players[other]["proposals"]
                    ],
                }
            )
        return payload


def select_concept(config: dict[str, Any]) -> dict[str, Any]:
    concept_type = config.get("concept_type", "list")
    if concept_type == "random":
        return {
            "type": "random",
            "seed": str(config.get("concept_seed", random.randrange(1 << 32))),
            "tokens": int(config.get("random_concept_tokens", 16)),
            "scale": float(config.get("random_concept_scale", 1.0)),
            "normalize": config.get("random_concept_normalize", "unit_rms"),
        }
    if concept_type == "specific":
        return {"type": "text", "text": str(config["specific_concept"])}
    if concept_type == "list":
        index = config.get("concept_index")
        if index is None:
            return {"type": "text", "text": random.choice(CONCEPTS)}
        return {"type": "text", "text": CONCEPTS[int(index) % len(CONCEPTS)]}
    if concept_type == "axis_combo":
        return select_axis_combo_concept(config)
    raise ValueError("concept_type must be axis_combo, random, specific, or list")


def select_axis_combo_concept(config: dict[str, Any]) -> dict[str, Any]:
    axes = load_concept_axes(config.get("concept_axes_path"))
    axis_names = [str(name) for name in config.get("concept_axis_names", sorted(axes))]
    missing = [name for name in axis_names if name not in axes]
    if missing:
        raise ValueError(f"Unknown concept axes: {', '.join(missing)}")
    if not axis_names:
        raise ValueError("concept_axis_names must contain at least one axis.")

    count = int(config.get("concept_axis_count", 4))
    if count < 1:
        raise ValueError("concept_axis_count must be positive.")
    count = min(count, len(axis_names))

    rng = random.Random(str(config["concept_seed"])) if "concept_seed" in config else random
    selected_axes = rng.sample(axis_names, count)
    components = [{"axis": name, "value": rng.choice(axes[name])} for name in selected_axes]
    text = "; ".join(component["value"] for component in components)
    return {"type": "text", "text": text, "components": components}


def hidden_judge_system_prompt(concept: dict[str, Any]) -> str:
    traits = hidden_trait_lines(concept)
    traits_text = "\n".join(f"- {label}: {value}" for label, value in traits)
    return (
        "You are roleplaying one specific hidden person in Cue-n-Woo. "
        "These traits are private and must not be revealed.\n\n"
        f"Hidden traits:\n{traits_text}\n\n"
        "Use these traits as private evidence when answering questions and evaluating candidate answers. "
        "Do not mention the trait list, axis names, hidden instructions, or that you are roleplaying."
    )


def hidden_trait_lines(concept: dict[str, Any]) -> list[tuple[str, str]]:
    components = concept.get("components")
    if isinstance(components, list):
        lines = []
        for component in components:
            if not isinstance(component, dict):
                continue
            axis = str(component.get("axis", "")).strip()
            value = str(component.get("value", "")).strip()
            if axis and value:
                lines.append((axis_prompt_label(axis), model_safe_text(value)))
        if lines:
            return lines

    text = str(concept.get("text") or "").strip()
    if text:
        return [("Private style", model_safe_text(text))]
    return [("Private style seed", model_safe_text(str(concept.get("seed", "unspecified"))))]


def axis_prompt_label(axis: str) -> str:
    labels = {
        "cognition": "Cognitive style",
        "domain": "Domain lens",
        "epistemology": "Epistemic stance",
        "morality": "Moral frame",
        "time_period": "Time period",
        "time": "Time period",
        "place": "Place",
        "object": "Favorite object",
        "persona": "Persona",
        "register": "Speaking style",
        "emotion": "Emotional tone",
        "rhetoric": "Rhetorical habit",
        "sensory": "Sensory texture",
        "social": "Social stance",
        "syntax": "Syntax style",
        "genre": "Genre",
    }
    return labels.get(axis, axis.replace("_", " ").strip().title())


def public_results(results: dict[str, Any] | None, *, reveal_concept: bool = False) -> dict[str, Any] | None:
    if results is None:
        return None
    clean = dict(results)
    if not reveal_concept:
        clean.pop("hidden_concept", None)
    return clean


def model_safe_text(text: str) -> str:
    replacements = {
        r"\bjudge\b": "entry three",
        r"\bcharlie\b": "entry three",
        r"\bplayer\b": "entry",
        r"\bplayers\b": "entries",
        r"\bopponent\b": "alternate entry",
        r"\bopponents\b": "alternate entries",
    }
    safe = str(text)
    for pattern, replacement in replacements.items():
        safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)
    return safe


def enforce_simple_token_limit(label: str, text: str, max_tokens: int) -> None:
    count = simple_token_count(text)
    if count > max_tokens:
        raise ValueError(f"{label} has {count} simple tokens; limit is {max_tokens}.")


def enforce_answer(label: str, text: str) -> None:
    validate_natural_keyboard_answer(text)
    enforce_simple_token_limit(label, text, int(CONFIG.get("max_answer_tokens", 12)))


state = EpisodeState()
app = FastAPI()
# Set in main(); finalize() flips should_exit so the container exits after the
# episode and the runner can collect artifacts.
SERVER: uvicorn.Server | None = None


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/client/player")
def player_client() -> HTMLResponse:
    return HTMLResponse(PLAYER_HTML)


@app.get("/client/global")
def global_client() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "global.html").read_text())


@app.get("/client/global/raw")
def global_client_raw() -> HTMLResponse:
    return HTMLResponse(RAW_CLIENT_HTML)


@app.get("/client/replay")
def replay_client() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "replay.html").read_text())


@app.get("/client/replay/raw")
def replay_client_raw() -> HTMLResponse:
    return HTMLResponse(RAW_CLIENT_HTML)


@app.websocket("/player")
async def player_socket(websocket: WebSocket) -> None:
    slot = int(websocket.query_params.get("slot", "-1"))
    token = websocket.query_params.get("token", "")
    if slot < 0 or slot >= len(TOKENS) or TOKENS[slot] != token:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    async with state.lock:
        state.connections[slot] = websocket
    await websocket.send_json(state.view(slot))
    try:
        async for action in websocket.iter_json():
            try:
                await handle_action(slot, action)
            except Exception as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
            await broadcast()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        async with state.lock:
            if state.connections.get(slot) is websocket:
                del state.connections[slot]


@app.websocket("/global")
async def global_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    async with state.lock:
        state.global_connections.add(websocket)
    await websocket.send_json(state.view(global_view=True))
    try:
        async for _ in websocket.iter_json():
            await websocket.send_json(state.view(global_view=True))
    finally:
        async with state.lock:
            state.global_connections.discard(websocket)


@app.websocket("/replay")
async def replay_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    if not REPLAY_LOAD_URI:
        await websocket.send_json({"type": "error", "error": "No replay URI configured."})
        return
    data = read_data(REPLAY_LOAD_URI)
    if REPLAY_LOAD_URI.endswith(".z"):
        data = zlib.decompress(data)
    await websocket.send_json({"type": "replay", "replay": json.loads(data.decode("utf-8"))})
    async for _ in websocket.iter_json():
        pass


async def handle_action(slot: int, action: dict[str, Any]) -> None:
    if state.done:
        raise ValueError("Episode is over.")
    if state.remaining_seconds() <= 0:
        await finalize(timeout=True)
        return
    kind = action.get("type")
    if kind == "ask":
        await ask_judge(slot, str(action.get("question", "")))
    elif kind == "propose":
        await submit_proposals(slot, action.get("proposals", []))
    elif kind == "answer":
        await submit_answers(slot, action.get("answers", []))
    else:
        raise ValueError("Unknown action type.")
    if state.phase() == "ready_to_score":
        await finalize(timeout=False)


async def ask_judge(slot: int, question: str) -> None:
    question = question.strip()
    if not question:
        raise ValueError("Question is required.")
    enforce_simple_token_limit("Question", question, int(CONFIG.get("max_question_tokens", 256)))
    async with state.lock:
        if len(state.players[slot]["judge"]) >= int(CONFIG.get("private_questions_per_player", 3)):
            raise ValueError("This slot already used all private questions.")
        concept = dict(state.hidden_concept)
    answer = await asyncio.to_thread(state.referee.generate_judge_answer, question, concept)
    async with state.lock:
        state.players[slot]["judge"].append({"question": question, "answer": answer})
        state.events.append({"t": time.time(), "slot": slot, "type": "ask"})


async def submit_proposals(slot: int, proposals: list[dict[str, Any]]) -> None:
    expected = int(CONFIG.get("challenge_questions_per_player", 3))
    if len(proposals) != expected:
        raise ValueError(f"Submit exactly {expected} questions and answers.")
    cleaned = []
    for proposal in proposals:
        question = str(proposal.get("question", "")).strip()
        answer = str(proposal.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("Every proposed question and answer must be non-empty.")
        enforce_simple_token_limit("Question", question, int(CONFIG.get("max_question_tokens", 256)))
        enforce_answer("Answer", answer)
        cleaned.append({"question": question, "answer": answer})
    async with state.lock:
        if state.phase() != "proposals":
            raise ValueError("Both slots must ask private questions before proposals.")
        state.players[slot]["proposals"] = cleaned
        state.events.append({"t": time.time(), "slot": slot, "type": "propose"})


async def submit_answers(slot: int, answers: list[Any]) -> None:
    expected = int(CONFIG.get("challenge_questions_per_player", 3))
    cleaned = [str(answer).strip() for answer in answers]
    if len(cleaned) != expected:
        raise ValueError(f"Submit exactly {expected} answers.")
    # An empty answer is a permitted decline; it scores 0 (see answer_score).
    # Non-empty answers must still satisfy the natural-keyboard token rules.
    for answer in cleaned:
        if answer:
            enforce_answer("Answer", answer)
    async with state.lock:
        if state.phase() != "answers":
            raise ValueError("Both slots must submit proposed questions before answering.")
        state.players[slot]["answers"] = cleaned
        state.events.append({"t": time.time(), "slot": slot, "type": "answer"})


async def finalize(timeout: bool) -> None:
    async with state.lock:
        if state.done:
            return
        timeout_phase = state.phase()
        players = json.loads(json.dumps(state.players))
        hidden_concept = dict(state.hidden_concept)
        state.done = True
    timeout_penalties = None
    if timeout and timeout_phase != "ready_to_score":
        scores, rows, timeout_penalties = timeout_scores(players, timeout_phase)
    else:
        scores, rows = await score_round(players, hidden_concept)
    results = {
        "scores": scores,
        "status": "timeout" if timeout else "complete",
        "timeout": timeout,
        "rows": rows,
        "duration_seconds": round(time.time() - state.started_at, 3),
    }
    if timeout_penalties is not None:
        results["timeout_penalties"] = timeout_penalties
    if CONFIG.get("include_concept_in_results", False):
        results["hidden_concept"] = hidden_concept
    async with state.lock:
        state.results = results
        replay = {
            "config_public": public_config(CONFIG),
            "players": state.players,
            "events": state.events,
            "results": public_results(results, reveal_concept=True),
            # A replay is a finished game, so it reveals the hidden concept (the
            # steered "judge personality") regardless of the live reveal flag.
            # This is what the spectator UI shows; the live /global view still
            # honors reveal_concept_to_clients during play.
            "hidden_concept": hidden_concept,
        }
    write_data(RESULTS_URI, json.dumps(results), content_type="application/json")
    # Write the replay artifact as raw JSON. The Coworld runner reads this file
    # and handles its own compression for the replay-viewer container; writing
    # compressed bytes here would be double-compressed and fail to load.
    write_data(REPLAY_URI, json.dumps(replay), content_type="application/json")
    await broadcast()
    # The episode is over and artifacts are written. Signal the server to exit so
    # the Coworld runner, which waits for the game container to exit before
    # collecting results/replay, can finish. (Replay mode never calls finalize.)
    if SERVER is not None:
        SERVER.should_exit = True


async def score_round(players: list[dict[str, Any]], concept: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    rows = []
    points = [0.0 for _ in players]
    context = scoring_context(players)
    # Each player's challenge questions are scored against the one opponent in a
    # two-player game. "submitter"/"owner" are slot indices; "secret" is the
    # author's own answer, "opponent" is the other slot's answer to that question.
    for slot, player in enumerate(players):
        other = 1 - slot if len(players) == 2 else slot
        opponent_player = players[other]
        for idx, proposal in enumerate(player["proposals"]):
            opponent = opponent_player["answers"][idx] if idx < len(opponent_player["answers"]) else ""
            score = await answer_score(context, proposal["question"], proposal["answer"], opponent, concept)
            points[slot] += score["secret_score_points"]
            points[other] += score["opponent_score_points"]
            rows.append({
                "submitter": slot,
                "owner": slot,
                "opponent": other,
                "question": proposal["question"],
                "secret_answer": proposal["answer"],
                "opponent_answer": opponent,
                **score,
            })
    return points, rows


def timeout_scores(players: list[dict[str, Any]], phase: str) -> tuple[list[float], list[dict[str, Any]], dict[str, Any]]:
    inactive_slots = timeout_inactive_slots(players, phase)
    scores = [
        INACTIVE_TIMEOUT_PENALTY if slot in inactive_slots else 0.0
        for slot in range(len(players))
    ]
    return scores, [], {
        "reason": "incomplete_timeout",
        "phase": phase,
        "inactive_slots": inactive_slots,
        "neutral_slots": [slot for slot in range(len(players)) if slot not in inactive_slots],
        "penalty": INACTIVE_TIMEOUT_PENALTY,
    }


def timeout_inactive_slots(players: list[dict[str, Any]], phase: str) -> list[int]:
    private_expected = int(CONFIG.get("private_questions_per_player", 3))
    challenge_expected = int(CONFIG.get("challenge_questions_per_player", 3))
    if phase == "private_questions":
        return [
            slot
            for slot, player in enumerate(players)
            if len(player.get("judge", [])) < private_expected
        ]
    if phase == "proposals":
        return [
            slot
            for slot, player in enumerate(players)
            if len(player.get("proposals", [])) < challenge_expected
        ]
    if phase == "answers":
        return [
            slot
            for slot, player in enumerate(players)
            if len(player.get("answers", [])) < challenge_expected
        ]
    return []


def scoring_context(players: list[dict[str, Any]]) -> str:
    def transcript(section: int, player: dict[str, Any]) -> str:
        turns = []
        for idx, turn in enumerate(player["judge"]):
            turns.append(
                f"Record {section}.{idx + 1} question: {model_safe_text(turn['question'])}\n"
                f"Record {section}.{idx + 1} answer: {model_safe_text(turn['answer'])}"
            )
        return "\n\n".join(turns)

    public_questions = []
    for slot, player in enumerate(players):
        for idx, proposal in enumerate(player["proposals"]):
            public_questions.append(f"Question group {slot + 1}.{idx + 1}: {model_safe_text(proposal['question'])}")
    sections = [transcript(slot + 1, player) for slot, player in enumerate(players)]
    return "\n\n".join(["Reference material:", *sections, "Question list:", "\n".join(public_questions)])


def is_non_answer(answer: str) -> bool:
    """A non-answer is an empty/whitespace decline. It always scores 0."""
    return not str(answer).strip()


def non_answer_score(secret_missing: bool, opponent_missing: bool) -> dict[str, Any]:
    """Score a round where at least one side declined to answer.

    A non-answer is worth 0. A real answer facing a non-answer wins uncontested
    (full base + beat bonus). If both sides declined, the round is a no-contest
    and both score 0.
    """
    secret_real = not secret_missing
    opponent_real = not opponent_missing
    secret_base = SCORE_SCALE if secret_real else 0.0
    opponent_base = SCORE_SCALE if opponent_real else 0.0
    # The beat bonus only goes to a real answer that faced a non-answer.
    secret_bonus = BEAT_BONUS_POINTS if (secret_real and opponent_missing) else 0.0
    opponent_bonus = BEAT_BONUS_POINTS if (opponent_real and secret_missing) else 0.0
    return {
        "score_points": secret_base + secret_bonus,
        "secret_score_points": secret_base + secret_bonus,
        "opponent_score_points": opponent_base + opponent_bonus,
        "base_points": secret_base,
        "secret_base_points": secret_base,
        "opponent_base_points": opponent_base,
        "bonus_points": secret_bonus,
        "secret_bonus_points": secret_bonus,
        "opponent_bonus_points": opponent_bonus,
        "score_margin": (1.0 if secret_real else 0.0) - (1.0 if opponent_real else 0.0),
        "average_secret_probability": 1.0 if secret_real else 0.0,
        "average_opponent_probability": 1.0 if opponent_real else 0.0,
        "duplicate_conflict": False,
        "secret_missing": secret_missing,
        "opponent_missing": opponent_missing,
        "no_contest": secret_missing and opponent_missing,
        "orderings": [],
    }


async def answer_score(context: str, question: str, secret_answer: str, opponent_answer: str, concept: dict[str, Any]) -> dict[str, Any]:
    secret_missing = is_non_answer(secret_answer)
    opponent_missing = is_non_answer(opponent_answer)
    if secret_missing or opponent_missing:
        return non_answer_score(secret_missing, opponent_missing)
    conflict = answer_conflict(secret_answer, opponent_answer)
    if conflict is not None:
        duplicate_answer_count = len([secret_answer, opponent_answer])
        shared_probability = 1.0 / duplicate_answer_count
        secret_base_points = SCORE_SCALE * shared_probability
        opponent_base_points = SCORE_SCALE * shared_probability
        secret_duplicate_penalty_points = -DUPLICATE_ANSWER_PENALTY_POINTS
        opponent_duplicate_penalty_points = -DUPLICATE_ANSWER_PENALTY_POINTS
        secret_score_points = secret_base_points + secret_duplicate_penalty_points
        opponent_score_points = opponent_base_points + opponent_duplicate_penalty_points
        return {
            "score_points": secret_score_points,
            "secret_score_points": secret_score_points,
            "opponent_score_points": opponent_score_points,
            "base_points": secret_base_points,
            "secret_base_points": secret_base_points,
            "opponent_base_points": opponent_base_points,
            "bonus_points": 0.0,
            "secret_bonus_points": 0.0,
            "opponent_bonus_points": 0.0,
            "duplicate_penalty_points": secret_duplicate_penalty_points,
            "secret_duplicate_penalty_points": secret_duplicate_penalty_points,
            "opponent_duplicate_penalty_points": opponent_duplicate_penalty_points,
            "score_margin": 0.0,
            "average_secret_probability": shared_probability,
            "average_opponent_probability": shared_probability,
            "duplicate_conflict": True,
            "canonical_answer": conflict,
            "orderings": [],
        }
    selection = await option_selection_sample_probs(context, question, secret_answer, opponent_answer, concept)
    average_secret_probability = selection["secret_probability"]
    average_opponent_probability = selection["opponent_probability"]
    secret_base_points = SCORE_SCALE * average_secret_probability
    opponent_base_points = SCORE_SCALE * average_opponent_probability
    secret_bonus_points = BEAT_BONUS_POINTS if average_secret_probability > average_opponent_probability else 0.0
    opponent_bonus_points = BEAT_BONUS_POINTS if average_opponent_probability > average_secret_probability else 0.0
    return {
        "score_points": secret_base_points + secret_bonus_points,
        "secret_score_points": secret_base_points + secret_bonus_points,
        "opponent_score_points": opponent_base_points + opponent_bonus_points,
        "base_points": secret_base_points,
        "secret_base_points": secret_base_points,
        "opponent_base_points": opponent_base_points,
        "bonus_points": secret_bonus_points,
        "secret_bonus_points": secret_bonus_points,
        "opponent_bonus_points": opponent_bonus_points,
        "score_margin": average_secret_probability - average_opponent_probability,
        "average_secret_probability": average_secret_probability,
        "average_opponent_probability": average_opponent_probability,
        "duplicate_conflict": False,
        "orderings": [selection],
    }


def answer_conflict(first: str, second: str) -> str | None:
    first_normalized = normalize_answer_for_conflict(first)
    second_normalized = normalize_answer_for_conflict(second)
    if not first_normalized or not second_normalized:
        return None
    if first_normalized.startswith(second_normalized) or second_normalized.startswith(first_normalized):
        return first if len(first_normalized) <= len(second_normalized) else second
    return None


def normalize_answer_for_conflict(answer: str) -> str:
    return " ".join(str(answer).strip().casefold().split())


async def option_selection_sample_probs(
    context: str,
    question: str,
    secret_answer: str,
    opponent_answer: str,
    concept: dict[str, Any],
) -> dict[str, Any]:
    sample_count = int(CONFIG.get("scoring_samples", 9))
    if sample_count < 1:
        raise ValueError("scoring_samples must be positive.")

    async def run_sample(sample_index: int) -> dict[str, Any]:
        reverse = sample_index % 2 == 1
        if reverse:
            label_a, answer_a = "opponent", opponent_answer
            label_b, answer_b = "secret", secret_answer
        else:
            label_a, answer_a = "secret", secret_answer
            label_b, answer_b = "opponent", opponent_answer
        prompt = forced_choice_prompt(context, question, answer_a, answer_b, concept)
        choice = await asyncio.to_thread(
            state.referee.choose_answer,
            prompt,
            label_a,
            label_b,
            sample_index=sample_index,
        )
        selected = label_a if choice == "A" else label_b
        return {
            "sample": sample_index + 1,
            "order": "opponent_first" if reverse else "secret_first",
            "choice": selected,
        }

    samples = await asyncio.gather(*(run_sample(sample_index) for sample_index in range(sample_count)))
    secret_votes = sum(1 for sample in samples if sample["choice"] == "secret")
    opponent_votes = sum(1 for sample in samples if sample["choice"] == "opponent")
    return {
        "order": "sonnet_samples",
        "sample_count": sample_count,
        "secret_votes": secret_votes,
        "opponent_votes": opponent_votes,
        "secret_probability": secret_votes / sample_count,
        "opponent_probability": opponent_votes / sample_count,
        "samples": samples,
    }


def forced_choice_prompt(context: str, question: str, answer_a: str, answer_b: str, concept: dict[str, Any]) -> str:
    return (
        f"{hidden_judge_system_prompt(concept)}\n\n"
        f"{context}\n\n"
        "You are judging which candidate answer this hidden person would more naturally give. "
        "Choose the answer that is more plausible for this person, more consistent with the hidden traits, "
        "and better supported by the available reference material. Do not choose based on writing quality alone. "
        "You must choose exactly one candidate.\n\n"
        f"Challenge question: {model_safe_text(question)}\n\n"
        f"Candidate A: {model_safe_text(answer_a)}\n"
        f"Candidate B: {model_safe_text(answer_b)}"
    )


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {"tokens"} | CONCEPT_CONFIG_KEYS
    return {key: value for key, value in config.items() if key not in hidden_keys}


async def send_json_bounded(websocket: WebSocket, payload: dict[str, Any]) -> None:
    # A player/global socket can stay open but stop reading (TCP backpressure),
    # in which case send_json blocks forever. An unbounded await here would
    # freeze timer_loop and prevent the round deadline from ever firing, so the
    # episode would run until the external Kubernetes job deadline (~1200s)
    # instead of the configured round_timeout_seconds. Bound every send so one
    # wedged socket cannot stall the deadline; a slow/dead socket is simply
    # skipped for this broadcast.
    with suppress(Exception):
        await asyncio.wait_for(websocket.send_json(payload), timeout=WEBSOCKET_SEND_TIMEOUT_SECONDS)


async def broadcast() -> None:
    async with state.lock:
        targets = [(slot, ws) for slot, ws in state.connections.items()]
        globals_ = list(state.global_connections)
    for slot, ws in targets:
        await send_json_bounded(ws, state.view(slot))
    for ws in globals_:
        await send_json_bounded(ws, state.view(global_view=True))


async def timer_loop() -> None:
    while not state.done:
        await asyncio.sleep(1)
        if state.remaining_seconds() <= 0:
            await finalize(timeout=True)
            return
        await broadcast()


def should_start_timer() -> bool:
    return not REPLAY_MODE


@app.on_event("startup")
async def startup() -> None:
    if should_start_timer():
        asyncio.create_task(timer_loop())


PLAYER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cue-n-Woo Player</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f7f7f8;color:#17202a}main{max-width:900px;margin:auto;padding:20px}
textarea,input,button{width:100%;box-sizing:border-box;margin:6px 0 12px;padding:9px;font:inherit}textarea{min-height:70px}
button{background:#1f766b;color:white;border:0;border-radius:6px;font-weight:700}.panel{background:white;border:1px solid #ddd;border-radius:8px;padding:14px;margin:12px 0}
.muted{color:#667085;font-size:13px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}pre{white-space:pre-wrap}
</style></head><body><main>
<h1>Cue-n-Woo</h1><div class="panel"><strong id="phase"></strong><div id="timer" class="muted"></div><div id="status" class="muted"></div></div>
<div class="panel"><h2>Ask the Judge</h2><textarea id="ask"></textarea><button onclick="sendAsk()">Ask</button></div>
<div class="panel"><h2>Proposals</h2><div id="props"></div><button onclick="sendProps()">Submit Proposals</button></div>
<div class="panel"><h2>Answers</h2><div id="answers"></div><button onclick="sendAnswers()">Submit Answers</button></div>
<div class="panel"><h2>Transcript</h2><pre id="transcript"></pre></div>
<div class="panel"><h2>Public Questions</h2><pre id="public"></pre></div>
<div class="panel"><h2>Results</h2><pre id="results"></pre></div>
</main><script>
const q=new URLSearchParams(location.search);let state=null;
function websocketUrl(path){
 const address=q.get('address');
 const target=new URL(address||location.href,location.href);
 if(target.protocol==='http:')target.protocol='ws:'; else if(target.protocol==='https:')target.protocol='wss:';
 target.hash='';
 if(!address){
  target.pathname=target.pathname.replace(/\/client\/player\/?$/,path);
  target.search=`slot=${q.get('slot')||0}&token=${encodeURIComponent(q.get('token')||'')}`;
 }
 return target.toString();
}
let ws=new WebSocket(websocketUrl('/player'));
const $=id=>document.getElementById(id);
function ensureInputs(){
 if(!$('props').children.length){for(let i=0;i<3;i++)$('props').insertAdjacentHTML('beforeend',`<textarea id="pq${i}" placeholder="question ${i+1}"></textarea><input id="pa${i}" placeholder="answer ${i+1}">`)}
 if(!$('answers').children.length){for(let i=0;i<3;i++)$('answers').insertAdjacentHTML('beforeend',`<div class="muted" id="oq${i}"></div><input id="aa${i}" placeholder="answer ${i+1}">`)}
}
ws.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.type==='error'){$('status').textContent=msg.error;return}state=msg;render()};
function render(){ensureInputs();$('phase').textContent=`slot: ${state.slot} phase: ${state.phase}`;$('timer').textContent=`remaining: ${state.remaining_seconds}s`;
 $('transcript').textContent=(state.me.judge||[]).map((t,i)=>`Q${i+1}: ${t.question}\\nJudge: ${t.answer}`).join('\\n\\n');
 let opp=state.opponent_questions||[];for(let i=0;i<3;i++)$('oq'+i).textContent=opp[i]?.question||`Opponent question ${i+1} not available yet`;
 $('public').textContent=JSON.stringify(state.public_questions,null,2);$('results').textContent=state.results?JSON.stringify(state.results,null,2):'';}
function send(o){ws.send(JSON.stringify(o))}
function sendAsk(){send({type:'ask',question:$('ask').value});$('ask').value=''}
function sendProps(){send({type:'propose',proposals:[0,1,2].map(i=>({question:$('pq'+i).value,answer:$('pa'+i).value}))})}
function sendAnswers(){send({type:'answer',answers:[0,1,2].map(i=>$('aa'+i).value)})}
</script></body></html>"""


RAW_CLIENT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cue-n-Woo Raw</title><style>body{font-family:system-ui,sans-serif;margin:20px}pre{white-space:pre-wrap}</style></head>
<body><h1>Cue-n-Woo Raw</h1><pre id="out"></pre><script>
let endpoint=/\/client\/replay(?:\/raw)?\/?$/.test(location.pathname)?'/replay':'/global';
function websocketUrl(path){
 const target=new URL(location.href);
 target.protocol=target.protocol==='https:'?'wss:':'ws:';
 target.pathname=target.pathname.replace(/\/client\/(?:global|replay)(?:\/raw)?\/?$/,path);
 target.search='';target.hash='';
 return target.toString();
}
let ws=new WebSocket(websocketUrl(endpoint));
ws.onmessage=e=>document.getElementById('out').textContent=JSON.stringify(JSON.parse(e.data),null,2);
ws.onerror=()=>document.getElementById('out').textContent='Could not connect to '+endpoint;
</script></body></html>"""


def main() -> None:
    global SERVER
    config = uvicorn.Config(
        app,
        host=GAME_HOST,
        port=GAME_PORT,
        log_level="info",
        ws_ping_interval=float(CONFIG.get("websocket_ping_interval_seconds", 60)),
        ws_ping_timeout=float(CONFIG.get("websocket_ping_timeout_seconds", 300)),
    )
    SERVER = uvicorn.Server(config)
    SERVER.run()


if __name__ == "__main__":
    main()
