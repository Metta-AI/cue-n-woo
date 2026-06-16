#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import math
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

from v2 import signing
from v2.coworld.harness import public_hints, simple_token_count, validate_natural_keyboard_answer


ROOT = Path(__file__).resolve().parent
HTTP_USER_AGENT = "cue-n-woo-coworld/0.1"
GAME_HOST = os.environ.get("COGAME_HOST", "0.0.0.0")
GAME_PORT = int(os.environ.get("COGAME_PORT", "8080"))
SCORE_SCALE = 100.0
BEAT_BONUS_POINTS = 10.0
DUPLICATE_ANSWER_PENALTY_POINTS = 10.0
INACTIVE_TIMEOUT_PENALTY = -100.0


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
            "llm_worker_url": "http://127.0.0.1:7870",
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


def load_signing_key(require: bool = False) -> Any | None:
    """Private key used to claim tournament priority on the worker, or None.

    Hosted episodes receive a short-lived URL resolved by the Coworld backend
    from the manifest's symbolic secret URI; local runs may set
    WORKER_SIGNING_KEY directly. When no key is available the game still works:
    its worker requests go unsigned and are served at normal priority. Unsigned
    is the expected mode for any local user, since they cannot read the private
    key.

    When ``require`` is true (config ``require_signing``), the inability to sign
    is a hard error instead of a silent downgrade. Tournaments set this so a
    broken key fetch fails loudly rather than quietly forfeiting priority and
    competing with public traffic.

    Runtime coupling (verified against metta-ai/metta): hosted episode dispatch
    resolves secret://coworld/cue_n_woo/tournament_signing_key into a presigned
    HTTPS URL before starting the game container. Hosted play, replay, downloaded
    images, and local runs keep the symbolic URI and degrade to unsigned unless
    the key is overridden locally.
    """
    inline = os.environ.get("WORKER_SIGNING_KEY")
    if inline:
        return signing.load_private_key(inline)
    key_uri = os.environ.get("WORKER_SIGNING_KEY_URI")
    if key_uri:
        # The published manifest sets a symbolic Coworld secret URI for every
        # run. Hosted episodes receive a presigned URL; local users keep the
        # symbolic URI and degrade to unsigned unless signing is required.
        try:
            seed_b64 = read_data(key_uri).decode("utf-8").strip()
        except Exception as exc:
            if require:
                raise RuntimeError(f"require_signing is set but WORKER_SIGNING_KEY_URI is unreadable: {exc}") from exc
            print(f"WORKER_SIGNING_KEY_URI unreadable ({exc}); running unsigned.", flush=True)
            return None
        return signing.load_private_key(seed_b64)
    if require:
        raise RuntimeError("require_signing is set but no WORKER_SIGNING_KEY or WORKER_SIGNING_KEY_URI is configured.")
    return None


class WorkerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.stub = bool(CONFIG.get("stub_worker", False))
        # Don't fetch/require a signing key in stub mode: certification runs
        # offline with no worker and no AWS credentials.
        self.signing_key = None if self.stub else load_signing_key(require=bool(CONFIG.get("require_signing", False)))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stub:
            return self._stub_response(path, payload)
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.signing_key is not None:
            timestamp = int(time.time())
            headers[signing.TIMESTAMP_HEADER] = str(timestamp)
            headers[signing.SIGNATURE_HEADER] = signing.sign_request(self.signing_key, timestamp, data)
        req = Request(self.base_url + path, data=data, headers=headers, method="POST")
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

    def _stub_response(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Deterministic offline responses so certification needs no live worker.

        Mirrors the real worker's response shapes without any model. The judge
        "answers" are derived from the request so the smoke test exercises the
        full game flow (ask -> propose -> answer -> score) end to end.
        """
        requests = payload.get("requests", [])
        if path == "/generate":
            results = []
            for req in requests:
                prompt = str(req.get("prompt", ""))
                results.append({"id": req.get("id"), "text": f"stub answer ({len(prompt)} chars)",
                                "finish_reason": "eos", "input_tokens": 0, "output_tokens": 4, "latency_ms": 0.0})
            return {"results": results}
        if path == "/choice-logprobs":
            results = []
            for req in requests:
                choices = req.get("choices", [])
                n = max(1, len(choices))
                results.append({"id": req.get("id"), "probabilities": [1.0 / n] * n, "orderings": []})
            return {"results": results}
        raise RuntimeError(f"stub worker does not handle {path}")


def empty_player() -> dict[str, Any]:
    return {"judge": [], "proposals": [], "answers": []}


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
        self.worker = WorkerClient(CONFIG.get("llm_worker_url", "http://127.0.0.1:7870"))
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

    def view(self, slot: int | None = None, *, global_view: bool = False) -> dict[str, Any]:
        phase = self.phase()
        payload: dict[str, Any] = {
            "type": "state",
            "phase": phase,
            "remaining_seconds": self.remaining_seconds(),
            "limits": {
                "max_answer_tokens": int(CONFIG.get("max_answer_tokens", 12)),
                "max_question_tokens": int(CONFIG.get("max_question_tokens", 1024)),
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
            "results": public_results(self.results),
            "done": self.done,
        }
        if not global_view and slot is not None and 0 <= slot < len(self.players):
            other = 1 - slot if len(self.players) == 2 else slot
            payload.update(
                {
                    "slot": slot,
                    "me": self.players[slot],
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
    enforce_simple_token_limit("Question", question, int(CONFIG.get("max_question_tokens", 1024)))
    async with state.lock:
        if len(state.players[slot]["judge"]) >= int(CONFIG.get("private_questions_per_player", 3)):
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
                        "max_tokens": judge_max_tokens(),
                        "max_prompt_tokens": int(CONFIG.get("max_prompt_tokens", 1024)),
                        "temperature": float(CONFIG.get("temperature", 0.7)),
                    },
                }
            ]
        },
    )
    answer = response["results"][0]["text"]
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
        enforce_simple_token_limit("Question", question, int(CONFIG.get("max_question_tokens", 1024)))
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
            "results": public_results(results),
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
    context = scoring_context()
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


def scoring_context() -> str:
    return "You will be presented with a question/challenge and two possible answers. Please select one of the two answers."


def is_non_answer(answer: str) -> bool:
    """A non-answer is an empty/whitespace decline. It always scores 0."""
    return not str(answer).strip()


def non_answer_score(secret_missing: bool, opponent_missing: bool) -> dict[str, Any]:
    """Score an answer matchup where at least one side declined to answer.

    A non-answer is worth 0. A real answer facing a non-answer wins uncontested
    (full base + beat bonus). If both sides declined, the matchup is a no-contest
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
    first = await delta_option_selection_probs(context, question, secret_answer, opponent_answer, concept, reverse=False)
    second = await delta_option_selection_probs(context, question, secret_answer, opponent_answer, concept, reverse=True)
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
    return await delta_option_selection_probs(context, question, secret_answer, opponent_answer, concept, reverse)


async def delta_option_selection_probs(context: str, question: str, secret_answer: str, opponent_answer: str, concept: dict[str, Any], reverse: bool) -> dict[str, Any]:
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
                    "id": "steered",
                    "prompt": prompt,
                    "concept": concept_for_worker(concept),
                    "flas": {
                        "flowtime": float(CONFIG.get("flas_flowtime", 2.0)),
                        "steps": int(CONFIG.get("flas_steps", 3)),
                    },
                    "choices": [model_safe_text(choice) for choice in choices],
                    "ordering": {"mode": "given_order"},
                },
                {
                    "id": "unsteered",
                    "prompt": prompt,
                    "concept": concept_for_worker(concept),
                    "flas": {
                        "flowtime": float(CONFIG.get("unsteered_flas_flowtime", 0.0)),
                        "steps": int(CONFIG.get("flas_steps", 3)),
                    },
                    "choices": [model_safe_text(choice) for choice in choices],
                    "ordering": {"mode": "given_order"},
                },
            ]
        },
    )
    by_id = {result.get("id"): result["probabilities"] for result in response["results"]}
    steered = by_id["steered"]
    unsteered = by_id["unsteered"]
    steered_secret = steered[1] if reverse else steered[0]
    steered_opponent = steered[0] if reverse else steered[1]
    unsteered_secret = unsteered[1] if reverse else unsteered[0]
    unsteered_opponent = unsteered[0] if reverse else unsteered[1]
    delta_log_odds = log_odds(steered_secret, steered_opponent) - log_odds(unsteered_secret, unsteered_opponent)
    secret_probability = sigmoid(delta_log_odds)
    opponent_probability = 1.0 - secret_probability
    return {
        "order": "opponent_first" if reverse else "secret_first",
        "secret_probability": secret_probability,
        "opponent_probability": opponent_probability,
        "delta_log_odds": delta_log_odds,
        "steered_secret_probability": steered_secret,
        "steered_opponent_probability": steered_opponent,
        "unsteered_secret_probability": unsteered_secret,
        "unsteered_opponent_probability": unsteered_opponent,
    }


def log_odds(first: float, second: float) -> float:
    eps = 1e-12
    return math.log(max(eps, first)) - math.log(max(eps, second))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {"tokens", "specific_concept", "concept_seed"}
    return {key: value for key, value in config.items() if key not in hidden_keys}


async def broadcast() -> None:
    async with state.lock:
        targets = [(slot, ws) for slot, ws in state.connections.items()]
        globals_ = list(state.global_connections)
    for slot, ws in targets:
        with suppress(Exception):
            await ws.send_json(state.view(slot))
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
let ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/player?slot=${q.get('slot')||0}&token=${encodeURIComponent(q.get('token')||'')}`);
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
let endpoint=location.pathname.includes('/replay/')?'/replay':'/global';
let ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}${endpoint}`);
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
