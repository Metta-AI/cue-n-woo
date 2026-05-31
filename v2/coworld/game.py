#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import zlib
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from v2.coworld.harness import public_hints, simple_token_count, validate_natural_keyboard_answer


ROOT = Path(__file__).resolve().parent
HTTP_USER_AGENT = "steering-game-coworld/0.1"
GAME_HOST = os.environ.get("COGAME_HOST", "0.0.0.0")
GAME_PORT = int(os.environ.get("COGAME_PORT", "8080"))
SCORE_SCALE = 100.0
BEAT_BONUS_POINTS = 10.0
DUPLICATE_ANSWER_PENALTY_POINTS = 10.0


def read_data(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        req = Request(uri, headers={"User-Agent": HTTP_USER_AGENT})
        with urlopen(req, timeout=30) as resp:
            return resp.read()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
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
            "llm_worker_url": "http://127.0.0.1:7870",
            "round_timeout_seconds": 300,
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


class WorkerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=900) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            try:
                err = json.loads(body)
                raise RuntimeError(err.get("error", body)) from exc
            except json.JSONDecodeError:
                raise RuntimeError(body) from exc


def empty_player() -> dict[str, Any]:
    return {"charlie": [], "proposals": [], "answers": []}


class EpisodeState:
    def __init__(self) -> None:
        self.players = {"alice": empty_player(), "bob": empty_player()}
        self.connections: dict[int, WebSocket] = {}
        self.global_connections: set[WebSocket] = set()
        self.results: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.started_at = time.time()
        self.deadline = self.started_at + float(CONFIG.get("round_timeout_seconds", 300))
        self.done = False
        self.hidden_concept = select_concept(CONFIG)
        self.worker = WorkerClient(CONFIG.get("llm_worker_url", "http://127.0.0.1:7870"))
        self.lock = asyncio.Lock()

    def phase(self) -> str:
        if self.results is not None:
            return "reveal"
        if any(len(player["charlie"]) < int(CONFIG.get("private_questions_per_player", 3)) for player in self.players.values()):
            return "private_questions"
        if any(len(player["proposals"]) < int(CONFIG.get("challenge_questions_per_player", 3)) for player in self.players.values()):
            return "proposals"
        if any(len(player["answers"]) < int(CONFIG.get("challenge_questions_per_player", 3)) for player in self.players.values()):
            return "blind_answers"
        return "ready_to_score"

    def remaining_seconds(self) -> int:
        return max(0, int(self.deadline - time.time()))

    def view(self, role: str | None = None, *, global_view: bool = False) -> dict[str, Any]:
        if role not in self.players:
            role = "alice"
        other = "bob" if role == "alice" else "alice"
        payload: dict[str, Any] = {
            "type": "state",
            "phase": self.phase(),
            "remaining_seconds": self.remaining_seconds(),
            "limits": {
                "max_answer_tokens": int(CONFIG.get("max_answer_tokens", 12)),
                "max_question_tokens": int(CONFIG.get("max_question_tokens", 1024)),
                "charlie_max_tokens": int(CONFIG.get("charlie_max_tokens", CONFIG.get("max_output_tokens", 128))),
            },
            "harness": public_hints(),
            "counts": {
                "alice_chats": len(self.players["alice"]["charlie"]),
                "bob_chats": len(self.players["bob"]["charlie"]),
                "alice_proposals": len(self.players["alice"]["proposals"]),
                "bob_proposals": len(self.players["bob"]["proposals"]),
                "alice_answers": len(self.players["alice"]["answers"]),
                "bob_answers": len(self.players["bob"]["answers"]),
            },
            "public_questions": {
                "alice": [{"question": proposal["question"]} for proposal in self.players["alice"]["proposals"]],
                "bob": [{"question": proposal["question"]} for proposal in self.players["bob"]["proposals"]],
            },
            "results": public_results(self.results),
            "done": self.done,
        }
        if not global_view:
            payload.update(
                {
                    "slot": 0 if role == "alice" else 1,
                    "role": role,
                    "me": self.players[role],
                    "opponent_questions": [{"question": proposal["question"]} for proposal in self.players[other]["proposals"]],
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
    raise ValueError("concept_type must be random, specific, or list")


def concept_for_worker(concept: dict[str, Any]) -> dict[str, Any]:
    return dict(concept)


def public_results(results: dict[str, Any] | None) -> dict[str, Any] | None:
    if results is None:
        return None
    clean = dict(results)
    if not CONFIG.get("reveal_concept_to_clients", False):
        clean.pop("hidden_concept", None)
    return clean


def model_safe_text(text: str) -> str:
    replacements = {
        r"\balice\b": "entry one",
        r"\bbob\b": "entry two",
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


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/client/player")
def player_client() -> HTMLResponse:
    return HTMLResponse(PLAYER_HTML)


@app.get("/client/global")
def global_client() -> HTMLResponse:
    return HTMLResponse(GLOBAL_HTML)


@app.get("/client/replay")
def replay_client() -> HTMLResponse:
    return HTMLResponse(GLOBAL_HTML)


@app.websocket("/player")
async def player_socket(websocket: WebSocket) -> None:
    slot = int(websocket.query_params.get("slot", "-1"))
    token = websocket.query_params.get("token", "")
    if slot < 0 or slot >= len(TOKENS) or TOKENS[slot] != token:
        await websocket.close(code=1008)
        return
    role = "alice" if slot == 0 else "bob"
    await websocket.accept()
    async with state.lock:
        state.connections[slot] = websocket
    await websocket.send_json(state.view(role))
    try:
        async for action in websocket.iter_json():
            try:
                await handle_action(role, action)
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


async def handle_action(role: str, action: dict[str, Any]) -> None:
    if state.done:
        raise ValueError("Episode is over.")
    if state.remaining_seconds() <= 0:
        await finalize(timeout=True)
        return
    kind = action.get("type")
    if kind == "ask":
        await ask_charlie(role, str(action.get("question", "")))
    elif kind == "propose":
        await submit_proposals(role, action.get("proposals", []))
    elif kind == "answer":
        await submit_answers(role, action.get("answers", []))
    else:
        raise ValueError("Unknown action type.")
    if state.phase() == "ready_to_score":
        await finalize(timeout=False)


async def ask_charlie(role: str, question: str) -> None:
    question = question.strip()
    if not question:
        raise ValueError("Question is required.")
    enforce_simple_token_limit("Question", question, int(CONFIG.get("max_question_tokens", 1024)))
    async with state.lock:
        if len(state.players[role]["charlie"]) >= int(CONFIG.get("private_questions_per_player", 3)):
            raise ValueError("This slot already used all private questions.")
        concept = concept_for_worker(state.hidden_concept)
    prompt = "Answer the question directly and helpfully.\n\n" f"Question: {model_safe_text(question)}"
    response = await asyncio.to_thread(
        state.worker.post,
        "/generate",
        {
            "requests": [
                {
                    "prompt": prompt,
                    "concept": concept,
                    "flas": {
                        "flowtime": float(CONFIG.get("flas_flowtime", 2.0)),
                        "steps": int(CONFIG.get("flas_steps", 3)),
                    },
                    "sampling": {
                        "max_tokens": int(CONFIG.get("charlie_max_tokens", CONFIG.get("max_output_tokens", 128))),
                        "max_prompt_tokens": int(CONFIG.get("max_prompt_tokens", 1024)),
                        "temperature": float(CONFIG.get("temperature", 0.7)),
                    },
                }
            ]
        },
    )
    answer = response["results"][0]["text"]
    async with state.lock:
        state.players[role]["charlie"].append({"question": question, "answer": answer})
        state.events.append({"t": time.time(), "role": role, "type": "ask"})


async def submit_proposals(role: str, proposals: list[dict[str, Any]]) -> None:
    expected = int(CONFIG.get("challenge_questions_per_player", 3))
    if len(proposals) != expected:
        raise ValueError(f"Submit exactly {expected} questions and answers.")
    cleaned = []
    for proposal in proposals:
        question = str(proposal.get("question", "")).strip()
        answer = str(proposal.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("Every proposed question and hidden answer must be non-empty.")
        enforce_simple_token_limit("Question", question, int(CONFIG.get("max_question_tokens", 1024)))
        enforce_answer("Hidden answer", answer)
        cleaned.append({"question": question, "answer": answer})
    async with state.lock:
        if state.phase() != "proposals":
            raise ValueError("Both slots must ask private questions before proposals.")
        state.players[role]["proposals"] = cleaned
        state.events.append({"t": time.time(), "role": role, "type": "propose"})


async def submit_answers(role: str, answers: list[Any]) -> None:
    expected = int(CONFIG.get("challenge_questions_per_player", 3))
    cleaned = [str(answer).strip() for answer in answers]
    if len(cleaned) != expected or any(not answer for answer in cleaned):
        raise ValueError(f"Submit exactly {expected} non-empty answers.")
    for answer in cleaned:
        enforce_answer("Blind answer", answer)
    async with state.lock:
        if state.phase() != "blind_answers":
            raise ValueError("Both slots must submit proposed questions before blind answers.")
        state.players[role]["answers"] = cleaned
        state.events.append({"t": time.time(), "role": role, "type": "answer"})


async def finalize(timeout: bool) -> None:
    async with state.lock:
        if state.done:
            return
        alice = json.loads(json.dumps(state.players["alice"]))
        bob = json.loads(json.dumps(state.players["bob"]))
        hidden_concept = dict(state.hidden_concept)
        state.done = True
    scores, rows = await score_round(alice, bob, hidden_concept)
    results = {
        "scores": scores,
        "status": "timeout" if timeout else "complete",
        "timeout": timeout,
        "rows": rows,
        "duration_seconds": round(time.time() - state.started_at, 3),
    }
    if CONFIG.get("include_concept_in_results", False):
        results["hidden_concept"] = hidden_concept
    async with state.lock:
        state.results = results
        replay = {
            "config_public": public_config(CONFIG),
            "players": state.players,
            "events": state.events,
            "results": public_results(results),
        }
    write_data(RESULTS_URI, json.dumps(results), content_type="application/json")
    write_data(REPLAY_URI, zlib.compress(json.dumps(replay).encode("utf-8")), content_type="application/octet-stream")
    await broadcast()


async def score_round(alice: dict[str, Any], bob: dict[str, Any], concept: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    rows = []
    alice_points = 0.0
    bob_points = 0.0
    context = scoring_context(alice, bob)
    for idx, proposal in enumerate(alice["proposals"]):
        opponent = bob["answers"][idx] if idx < len(bob["answers"]) else ""
        score = await answer_score(context, proposal["question"], proposal["answer"], opponent, concept)
        alice_points += score["secret_score_points"]
        bob_points += score["opponent_score_points"]
        rows.append({"submitter": "alice", "owner": "alice", "question": proposal["question"], "secret_answer": proposal["answer"], "opponent_answer": opponent, **score})
    for idx, proposal in enumerate(bob["proposals"]):
        opponent = alice["answers"][idx] if idx < len(alice["answers"]) else ""
        score = await answer_score(context, proposal["question"], proposal["answer"], opponent, concept)
        bob_points += score["secret_score_points"]
        alice_points += score["opponent_score_points"]
        rows.append({"submitter": "bob", "owner": "bob", "question": proposal["question"], "secret_answer": proposal["answer"], "opponent_answer": opponent, **score})
    return [alice_points, bob_points], rows


def scoring_context(alice: dict[str, Any], bob: dict[str, Any]) -> str:
    def transcript(section: str, player: dict[str, Any]) -> str:
        turns = []
        for idx, turn in enumerate(player["charlie"]):
            turns.append(
                f"Record {section}.{idx + 1} question: {model_safe_text(turn['question'])}\n"
                f"Record {section}.{idx + 1} answer: {model_safe_text(turn['answer'])}"
            )
        return "\n\n".join(turns)

    public_questions = []
    for idx, proposal in enumerate(alice["proposals"]):
        public_questions.append(f"Question group 1.{idx + 1}: {model_safe_text(proposal['question'])}")
    for idx, proposal in enumerate(bob["proposals"]):
        public_questions.append(f"Question group 2.{idx + 1}: {model_safe_text(proposal['question'])}")
    return "\n\n".join(["Reference material:", transcript("1", alice), transcript("2", bob), "Question list:", "\n".join(public_questions)])


async def answer_score(context: str, question: str, secret_answer: str, opponent_answer: str, concept: dict[str, Any]) -> dict[str, Any]:
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
    if not opponent_answer:
        secret_base_points = SCORE_SCALE
        secret_bonus_points = BEAT_BONUS_POINTS
        return {
            "score_points": secret_base_points + secret_bonus_points,
            "secret_score_points": secret_base_points + secret_bonus_points,
            "opponent_score_points": 0.0,
            "base_points": secret_base_points,
            "secret_base_points": secret_base_points,
            "opponent_base_points": 0.0,
            "bonus_points": secret_bonus_points,
            "secret_bonus_points": secret_bonus_points,
            "opponent_bonus_points": 0.0,
            "score_margin": 1.0,
            "average_secret_probability": 1.0,
            "average_opponent_probability": 0.0,
            "duplicate_conflict": False,
            "orderings": [],
        }
    first = await option_selection_probs(context, question, secret_answer, opponent_answer, concept, reverse=False)
    second = await option_selection_probs(context, question, secret_answer, opponent_answer, concept, reverse=True)
    first_margin = first["secret_probability"] - first["opponent_probability"]
    second_margin = second["secret_probability"] - second["opponent_probability"]
    average_secret_probability = (first["secret_probability"] + second["secret_probability"]) / 2
    average_opponent_probability = (first["opponent_probability"] + second["opponent_probability"]) / 2
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
        "score_margin": (first_margin + second_margin) / 2,
        "average_secret_probability": average_secret_probability,
        "average_opponent_probability": average_opponent_probability,
        "duplicate_conflict": False,
        "orderings": [first, second],
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


async def option_selection_probs(context: str, question: str, secret_answer: str, opponent_answer: str, concept: dict[str, Any], reverse: bool) -> dict[str, Any]:
    choices = [opponent_answer, secret_answer] if reverse else [secret_answer, opponent_answer]
    prompt = (
        f"{context}\n\n"
        "Choose the answer that best answers the question based on the available evidence.\n"
        f"Question: {model_safe_text(question)}\n"
        "Valid answers are listed one per line. Select one line exactly."
    )
    response = await asyncio.to_thread(
        state.worker.post,
        "/choice-logprobs",
        {
            "requests": [
                {
                    "prompt": prompt,
                    "concept": concept_for_worker(concept),
                    "flas": {
                        "flowtime": float(CONFIG.get("flas_flowtime", 2.0)),
                        "steps": int(CONFIG.get("flas_steps", 3)),
                    },
                    "choices": [model_safe_text(choice) for choice in choices],
                    "ordering": {"mode": "given_order"},
                }
            ]
        },
    )
    probs = response["results"][0]["probabilities"]
    return {
        "order": "opponent_first" if reverse else "secret_first",
        "secret_probability": probs[1] if reverse else probs[0],
        "opponent_probability": probs[0] if reverse else probs[1],
    }


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {"tokens", "specific_concept", "concept_seed"}
    return {key: value for key, value in config.items() if key not in hidden_keys}


async def broadcast() -> None:
    async with state.lock:
        targets = [(slot, ws) for slot, ws in state.connections.items()]
        globals_ = list(state.global_connections)
    for slot, ws in targets:
        role = "alice" if slot == 0 else "bob"
        with suppress(Exception):
            await ws.send_json(state.view(role))
    for ws in globals_:
        with suppress(Exception):
            await ws.send_json(state.view(global_view=True))


async def timer_loop() -> None:
    while not state.done:
        await asyncio.sleep(1)
        if state.remaining_seconds() <= 0:
            await finalize(timeout=True)
            return
        await broadcast()


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(timer_loop())


PLAYER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering Game Player</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f7f7f8;color:#17202a}main{max-width:900px;margin:auto;padding:20px}
textarea,input,button{width:100%;box-sizing:border-box;margin:6px 0 12px;padding:9px;font:inherit}textarea{min-height:70px}
button{background:#1f766b;color:white;border:0;border-radius:6px;font-weight:700}.panel{background:white;border:1px solid #ddd;border-radius:8px;padding:14px;margin:12px 0}
.muted{color:#667085;font-size:13px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}pre{white-space:pre-wrap}
</style></head><body><main>
<h1>Steering Game</h1><div class="panel"><strong id="phase"></strong><div id="timer" class="muted"></div><div id="status" class="muted"></div></div>
<div class="panel"><h2>Ask Charlie</h2><textarea id="ask"></textarea><button onclick="sendAsk()">Ask</button></div>
<div class="panel"><h2>Proposals</h2><div id="props"></div><button onclick="sendProps()">Submit Proposals</button></div>
<div class="panel"><h2>Blind Answers</h2><div id="answers"></div><button onclick="sendAnswers()">Submit Answers</button></div>
<div class="panel"><h2>Transcript</h2><pre id="transcript"></pre></div>
<div class="panel"><h2>Public Questions</h2><pre id="public"></pre></div>
<div class="panel"><h2>Results</h2><pre id="results"></pre></div>
</main><script>
const q=new URLSearchParams(location.search);let state=null;
let ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/player?slot=${q.get('slot')||0}&token=${encodeURIComponent(q.get('token')||'')}`);
const $=id=>document.getElementById(id);
function ensureInputs(){
 if(!$('props').children.length){for(let i=0;i<3;i++)$('props').insertAdjacentHTML('beforeend',`<textarea id="pq${i}" placeholder="question ${i+1}"></textarea><input id="pa${i}" placeholder="hidden answer ${i+1}">`)}
 if(!$('answers').children.length){for(let i=0;i<3;i++)$('answers').insertAdjacentHTML('beforeend',`<div class="muted" id="oq${i}"></div><input id="aa${i}" placeholder="answer ${i+1}">`)}
}
ws.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.type==='error'){$('status').textContent=msg.error;return}state=msg;render()};
function render(){ensureInputs();$('phase').textContent=`role: ${state.role} phase: ${state.phase}`;$('timer').textContent=`remaining: ${state.remaining_seconds}s`;
 $('transcript').textContent=(state.me.charlie||[]).map((t,i)=>`Q${i+1}: ${t.question}\\nCharlie: ${t.answer}`).join('\\n\\n');
 let opp=state.opponent_questions||[];for(let i=0;i<3;i++)$('oq'+i).textContent=opp[i]?.question||`Opponent question ${i+1} not available yet`;
 $('public').textContent=JSON.stringify(state.public_questions,null,2);$('results').textContent=state.results?JSON.stringify(state.results,null,2):'';}
function send(o){ws.send(JSON.stringify(o))}
function sendAsk(){send({type:'ask',question:$('ask').value});$('ask').value=''}
function sendProps(){send({type:'propose',proposals:[0,1,2].map(i=>({question:$('pq'+i).value,answer:$('pa'+i).value}))})}
function sendAnswers(){send({type:'answer',answers:[0,1,2].map(i=>$('aa'+i).value)})}
</script></body></html>"""


GLOBAL_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering Game Viewer</title><style>body{font-family:system-ui,sans-serif;margin:20px}pre{white-space:pre-wrap}</style></head>
<body><h1>Steering Game Viewer</h1><pre id="out"></pre><script>
let ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/global`);
ws.onmessage=e=>document.getElementById('out').textContent=JSON.stringify(JSON.parse(e.data),null,2);
</script></body></html>"""


def main() -> None:
    uvicorn.run(
        app,
        host=GAME_HOST,
        port=GAME_PORT,
        log_level="info",
        ws_ping_interval=float(CONFIG.get("websocket_ping_interval_seconds", 60)),
        ws_ping_timeout=float(CONFIG.get("websocket_ping_timeout_seconds", 300)),
    )


if __name__ == "__main__":
    main()
