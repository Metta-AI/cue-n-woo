"""Offline Coworld smoke test for the bundled game and stub players.

Run with:

    PYTHONPATH=. .devvenv/bin/python -m pytest v2/tests/test_coworld_offline_smoke.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"game server exited before healthz\nstdout:\n{stdout}\nstderr:\n{stderr}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"game server did not become healthy at {url}")


def _run_stub_player(port: int, slot: int, token: str, repo_root: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["COWORLD_PLAYER_WS_URL"] = f"ws://127.0.0.1:{port}/player?slot={slot}&token={token}"
    return subprocess.Popen(
        [sys.executable, "-m", "v2.coworld.players.stub_players"],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_coworld_game_and_stub_players_finish_offline(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    port = _free_port()
    config_path = tmp_path / "config.json"
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.json"
    tokens = ["alice-token", "bob-token"]
    config_path.write_text(
        json.dumps(
            {
                "tokens": tokens,
                "players": [{"name": "Alice"}, {"name": "Bob"}],
                "llm_worker_url": "https://cue-n-woo-worker.softmax-research.net",
                "stub_worker": True,
                "round_timeout_seconds": 60,
                "private_questions_per_player": 3,
                "challenge_questions_per_player": 3,
                "max_question_tokens": 1024,
                "max_answer_tokens": 12,
                "max_prompt_tokens": 1024,
                "judge_max_tokens": 128,
                "temperature": 0,
                "concept_type": "list",
                "concept_index": 0,
                "flas_flowtime": 2,
                "flas_steps": 3,
                "reveal_concept_to_clients": False,
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo_root),
            "COGAME_HOST": "127.0.0.1",
            "COGAME_PORT": str(port),
            "COGAME_CONFIG_URI": config_path.as_uri(),
            "COGAME_RESULTS_URI": results_path.as_uri(),
            "COGAME_SAVE_REPLAY_URI": replay_path.as_uri(),
        }
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "v2.coworld.game"],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_health(port, server)
        players = [_run_stub_player(port, slot, token, repo_root) for slot, token in enumerate(tokens)]
        for player in players:
            stdout, stderr = player.communicate(timeout=15)
            assert player.returncode == 0, f"stub player failed\nstdout:\n{stdout}\nstderr:\n{stderr}"

        stdout, stderr = server.communicate(timeout=15)
        assert server.returncode == 0, f"game server failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert results["status"] == "complete"
    assert results["timeout"] is False
    assert len(results["scores"]) == 2
    assert len(results["rows"]) == 6
    assert [len(player["judge"]) for player in replay["players"]] == [3, 3]
    assert len(replay["results"]["rows"]) == 6
