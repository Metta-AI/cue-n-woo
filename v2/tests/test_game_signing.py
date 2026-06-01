"""Tests for game-side signing-key resolution and the require_signing flag.

Run with the dev venv (cryptography + fastapi, no torch/flas needed):

    PYTHONPATH=. .devvenv/bin/python -m pytest v2/tests/test_game_signing.py
"""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import v2.coworld.game as game


def seed_b64() -> str:
    priv = Ed25519PrivateKey.generate()
    return base64.b64encode(
        priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode("ascii")


def test_inline_key_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_SIGNING_KEY", seed_b64())
    monkeypatch.delenv("WORKER_SIGNING_KEY_URI", raising=False)
    assert game.load_signing_key() is not None


def test_no_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_SIGNING_KEY", raising=False)
    monkeypatch.delenv("WORKER_SIGNING_KEY_URI", raising=False)
    assert game.load_signing_key() is None


def test_no_key_with_require_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_SIGNING_KEY", raising=False)
    monkeypatch.delenv("WORKER_SIGNING_KEY_URI", raising=False)
    with pytest.raises(RuntimeError, match="require_signing"):
        game.load_signing_key(require=True)


def test_unreadable_uri_degrades_without_require(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_SIGNING_KEY", raising=False)
    monkeypatch.setenv("WORKER_SIGNING_KEY_URI", "s3://nonexistent-bucket-xyz/missing-key")
    monkeypatch.setattr(game, "read_data", lambda uri: (_ for _ in ()).throw(RuntimeError("denied")))
    assert game.load_signing_key(require=False) is None


def test_unreadable_uri_raises_with_require(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_SIGNING_KEY", raising=False)
    monkeypatch.setenv("WORKER_SIGNING_KEY_URI", "s3://nonexistent-bucket-xyz/missing-key")
    monkeypatch.setattr(game, "read_data", lambda uri: (_ for _ in ()).throw(RuntimeError("denied")))
    with pytest.raises(RuntimeError, match="require_signing"):
        game.load_signing_key(require=True)


def test_uri_key_loads_when_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKER_SIGNING_KEY", raising=False)
    monkeypatch.setenv("WORKER_SIGNING_KEY_URI", "s3://bucket/key")
    seed = seed_b64()
    monkeypatch.setattr(game, "read_data", lambda uri: seed.encode("utf-8"))
    assert game.load_signing_key() is not None
