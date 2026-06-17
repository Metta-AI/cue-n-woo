"""Tests for game-side Bedrock referee behavior.

Run with the dev venv (cryptography + fastapi, no torch/flas needed):

    PYTHONPATH=. .devvenv/bin/python -m pytest v2/tests/test_game_signing.py
"""

from __future__ import annotations

import asyncio
import json

from botocore.config import Config

import v2.coworld.game as game
from v2.coworld.harness import simple_token_count, truncate_to_token_limit


def test_bedrock_referee_uses_sonnet_default(monkeypatch) -> None:
    monkeypatch.delenv("BEDROCK_CLAUDE_MODEL_ID", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)

    client = game.BedrockRefereeClient({})

    assert "sonnet" in client.model_id
    assert client.region == "us-east-1"


def test_bedrock_referee_env_model_override(monkeypatch) -> None:
    monkeypatch.setenv("BEDROCK_CLAUDE_MODEL_ID", "custom-sonnet")

    client = game.BedrockRefereeClient({})

    assert client.model_id == "custom-sonnet"


def test_bedrock_client_uses_bounded_timeouts(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        pass

    def fake_client(service_name, *, region_name, config):
        captured["service_name"] = service_name
        captured["region_name"] = region_name
        captured["config"] = config
        return FakeClient()

    monkeypatch.setattr(game.boto3, "client", fake_client)

    client = game.BedrockRefereeClient(
        {
            "bedrock_region": "us-west-2",
            "bedrock_connect_timeout_seconds": 3,
            "bedrock_read_timeout_seconds": 7,
        }
    )

    assert client._client() is not None
    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "us-west-2"
    assert isinstance(captured["config"], Config)
    assert captured["config"].connect_timeout == 3.0
    assert captured["config"].read_timeout == 7.0
    assert captured["config"].retries == {"max_attempts": 1}


def test_token_budget_uses_four_character_estimate() -> None:
    assert simple_token_count("") == 0
    assert simple_token_count("abcd") == 1
    assert simple_token_count("abcde") == 2
    assert truncate_to_token_limit("abcdefghij", 2) == "abcdefgh"


def test_option_selection_uses_nine_sonnet_samples(monkeypatch) -> None:
    choices = ["A", "B", "A", "A", "B", "A", "B", "A", "A"]
    calls = []

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    class FakeReferee:
        def choose_answer(self, prompt, label_a, label_b, *, sample_index):
            calls.append((prompt, label_a, label_b, sample_index))
            return choices[sample_index]

    monkeypatch.setattr(game.state, "referee", FakeReferee())
    monkeypatch.setattr(game.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setitem(game.CONFIG, "scoring_samples", 9)

    result = asyncio.run(
        game.option_selection_sample_probs(
            "context",
            "question",
            "secret",
            "opponent",
            {"type": "text", "text": "dry wit"},
        )
    )

    assert len(calls) == 9
    assert [call[3] for call in calls] == list(range(9))
    assert result["sample_count"] == 9
    assert result["secret_votes"] == 4
    assert result["opponent_votes"] == 5
    assert result["secret_probability"] == 4 / 9
    assert result["opponent_probability"] == 5 / 9


def test_live_mode_starts_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(game, "REPLAY_MODE", False)
    assert game.should_start_timer() is True


def test_replay_mode_does_not_start_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(game, "REPLAY_MODE", True)
    assert game.should_start_timer() is False


def test_global_client_serves_pretty_spectator_page() -> None:
    body = game.global_client().body.decode("utf-8")

    assert "Global spectator view" in body
    assert "raw json" in body
    assert "<pre id=\"out\">" not in body
    assert 'href="global/raw"' in body
    assert "address=new URLSearchParams(location.search).get('address')" in body
    assert "replace(/\\/client\\/global\\/?$/,path)" in body


def test_replay_client_uses_proxy_relative_websocket_and_raw_link() -> None:
    body = game.replay_client().body.decode("utf-8")

    assert "Cue 'n' Woo" in body
    assert 'href="replay/raw"' in body
    assert "replace(/\\/client\\/replay(?:\\/.*)?$/,path)" in body
    assert "new WebSocket(websocketUrl('/replay'))" in body


def test_raw_clients_serve_json_debug_page() -> None:
    global_body = game.global_client_raw().body.decode("utf-8")
    replay_body = game.replay_client_raw().body.decode("utf-8")

    assert "<pre id=\"out\">" in global_body
    assert "<pre id=\"out\">" in replay_body
    assert "let endpoint=/\\/client\\/replay(?:\\/raw)?\\/?$/.test(location.pathname)?'/replay':'/global';" in global_body
    assert "replace(/\\/client\\/(?:global|replay)(?:\\/raw)?\\/?$/,path)" in replay_body


def test_active_player_and_global_views_do_not_expose_hidden_concept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(game.CONFIG, "reveal_concept_to_clients", True)
    episode = game.EpisodeState()
    episode.hidden_concept = {
        "type": "text",
        "text": "pirate persona",
        "components": [{"axis": "persona", "value": "pirate"}],
    }

    for snapshot in [episode.view(0), episode.view(global_view=True)]:
        assert "hidden_concept" not in _keys(snapshot)
        assert not any("concept" in key for key in _keys(snapshot))
        assert "pirate persona" not in json.dumps(snapshot)


def test_public_results_reveal_concept_only_when_explicitly_revealed() -> None:
    results = {
        "scores": [0.0, 0.0],
        "hidden_concept": {"type": "text", "text": "noir detective persona"},
    }

    assert "hidden_concept" not in game.public_results(results, reveal_concept=False)
    assert game.public_results(results, reveal_concept=True)["hidden_concept"] == results["hidden_concept"]


def test_public_config_strips_concept_selection_internals() -> None:
    config = {
        "tokens": ["secret"],
        "players": [{"name": "Alice"}, {"name": "Bob"}],
        "concept_type": "axis_combo",
        "concept_index": 3,
        "specific_concept": "specific persona",
        "concept_seed": "round-seed",
        "concept_list_path": "/private/concepts.json",
        "concept_axes_path": "/private/axes",
        "concept_axis_names": ["persona"],
        "concept_axis_count": 1,
        "random_concept_tokens": 16,
        "random_concept_scale": 1.0,
        "random_concept_normalize": "unit_rms",
        "reveal_concept_to_clients": True,
        "include_concept_in_results": True,
        "round_timeout_seconds": 300,
    }

    public = game.public_config(config)

    assert public == {
        "players": [{"name": "Alice"}, {"name": "Bob"}],
        "round_timeout_seconds": 300,
    }


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(_keys(item))
        return keys
    return set()
