"""Tests for tournament request signing and worker priority classification.

Run with the dev venv (cryptography only, no torch/flas needed):

    PYTHONPATH=. .devvenv/bin/python -m pytest v2/tests/test_signing.py
"""

from __future__ import annotations

import base64
import json
import threading
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import v2.llm_worker as worker
from v2 import signing


def make_keypair() -> tuple[Ed25519PrivateKey, str]:
    priv = Ed25519PrivateKey.generate()
    pub_b64 = base64.b64encode(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    return priv, pub_b64


def test_sign_verify_round_trip() -> None:
    priv, pub_b64 = make_keypair()
    pub = signing.load_public_key(pub_b64)
    body = b'{"requests":[{"prompt":"hi"}]}'
    ts = 1_000_000
    sig = signing.sign_request(priv, ts, body)
    assert signing.verify_request(pub, ts, sig, body, now=ts) is True


def test_verify_rejects_stale_timestamp() -> None:
    priv, pub_b64 = make_keypair()
    pub = signing.load_public_key(pub_b64)
    body = b"payload"
    ts = 1_000_000
    sig = signing.sign_request(priv, ts, body)
    assert signing.verify_request(pub, ts, sig, body, now=ts + 1000) is False


def test_verify_rejects_tampered_body() -> None:
    priv, pub_b64 = make_keypair()
    pub = signing.load_public_key(pub_b64)
    ts = 1_000_000
    sig = signing.sign_request(priv, ts, b"original")
    assert signing.verify_request(pub, ts, sig, b"tampered", now=ts) is False


def test_verify_rejects_wrong_key() -> None:
    priv, _ = make_keypair()
    _, other_pub_b64 = make_keypair()
    other_pub = signing.load_public_key(other_pub_b64)
    body = b"payload"
    ts = 1_000_000
    sig = signing.sign_request(priv, ts, body)
    assert signing.verify_request(other_pub, ts, sig, body, now=ts) is False


def _server_with_key(pub_b64: str) -> worker.WorkerServer:
    server = worker.WorkerServer.__new__(worker.WorkerServer)
    server.public_key = signing.load_public_key(pub_b64)
    return server


def test_request_priority_valid_signature_is_tournament() -> None:
    priv, pub_b64 = make_keypair()
    server = _server_with_key(pub_b64)
    body = b'{"requests":[{"prompt":"hi"}]}'
    ts = int(time.time())
    sig = signing.sign_request(priv, ts, body)
    headers = {signing.TIMESTAMP_HEADER: str(ts), signing.SIGNATURE_HEADER: sig}
    assert server.request_priority(headers, body) == worker.PRIORITY_TOURNAMENT


def test_request_priority_unsigned_is_normal_not_rejected() -> None:
    _, pub_b64 = make_keypair()
    server = _server_with_key(pub_b64)
    assert server.request_priority({}, b"body") == worker.PRIORITY_NORMAL


def test_request_priority_bad_signature_is_normal() -> None:
    priv, pub_b64 = make_keypair()
    server = _server_with_key(pub_b64)
    body = b"body"
    ts = int(time.time())
    sig = signing.sign_request(priv, ts, body)
    # Tampered body forfeits priority but is still served at normal priority.
    assert server.request_priority({signing.TIMESTAMP_HEADER: str(ts), signing.SIGNATURE_HEADER: sig}, b"other") == worker.PRIORITY_NORMAL
    # Non-integer timestamp.
    assert server.request_priority({signing.TIMESTAMP_HEADER: "nope", signing.SIGNATURE_HEADER: sig}, body) == worker.PRIORITY_NORMAL


def test_request_priority_no_public_key_disables_signing() -> None:
    server = worker.WorkerServer.__new__(worker.WorkerServer)
    server.public_key = None
    assert server.request_priority({signing.TIMESTAMP_HEADER: "1", signing.SIGNATURE_HEADER: "x"}, b"b") == worker.PRIORITY_NORMAL


def test_scheduler_serves_tournament_before_queued_normal() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.order: list[str] = []
            self.gate = threading.Event()

        def run_batch(self, op: str, payloads: list[dict]) -> list[dict]:
            self.gate.wait()
            self.order.extend(p["id"] for p in payloads)
            return [{"id": p["id"]} for p in payloads]

    model = FakeModel()
    # max_batch_size=1 isolates ordering from batching.
    server = worker.WorkerServer(model, max_batch_size=1)

    threads = []
    for name, prio in [
        ("n1", worker.PRIORITY_NORMAL),
        ("t1", worker.PRIORITY_TOURNAMENT),
        ("n2", worker.PRIORITY_NORMAL),
    ]:
        fut = server.scheduler.submit("generate", {"id": name}, prio)
        threads.append(threading.Thread(target=fut.wait))
        threads[-1].start()
        time.sleep(0.05)  # deterministic enqueue order

    time.sleep(0.1)
    model.gate.set()
    for t in threads:
        t.join()

    # n1 ran first (queue was empty); among the pair queued behind it, the
    # tournament request preempts the later normal one.
    assert model.order.index("t1") < model.order.index("n2")
