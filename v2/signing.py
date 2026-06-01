"""Ed25519 request signing shared by the coworld game and the LLM worker.

The worker is exposed publicly and intentionally serves anyone: unsigned
requests are served at normal priority with no limit. A request that carries a
valid, fresh signature is served at *tournament* priority instead. The signer
(the hosted game) holds the private key; the worker holds only the public key,
so the publicly downloadable worker image and config contain no secret. Forging
the signature is the only way to jump the queue, and that requires the private
key, which is never shipped.

Canonical signed message: ``f"{timestamp}\\n{sha256_hex(body)}"``. The timestamp
guards against replay; the body digest binds the signature to the exact payload.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

TIMESTAMP_HEADER = "X-Tournament-Timestamp"
SIGNATURE_HEADER = "X-Tournament-Signature"

# Default tournament public key (base64 of the 32 raw Ed25519 public bytes). A
# public key is not a secret, so it is safe to commit and bake into the image.
# Override with WORKER_SIGNING_PUBLIC_KEY to rotate without a code change.
DEFAULT_PUBLIC_KEY_B64 = "QYvcuOqN2ttj8+5ZnbbIHvWviL4I5Et2M+U8vv0D0d4="

# How far the request timestamp may drift from the worker clock, in seconds.
MAX_TIMESTAMP_SKEW_SECONDS = 60


def canonical_message(timestamp: int, body: bytes) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return f"{timestamp}\n{digest}".encode("utf-8")


def load_private_key(seed_b64: str) -> Ed25519PrivateKey:
    seed = base64.b64decode(seed_b64)
    return Ed25519PrivateKey.from_private_bytes(seed)


def load_public_key(public_b64: str) -> Ed25519PublicKey:
    raw = base64.b64decode(public_b64)
    return Ed25519PublicKey.from_public_bytes(raw)


def sign_request(private_key: Ed25519PrivateKey, timestamp: int, body: bytes) -> str:
    signature = private_key.sign(canonical_message(timestamp, body))
    return base64.b64encode(signature).decode("ascii")


def verify_request(
    public_key: Ed25519PublicKey,
    timestamp: int,
    signature_b64: str,
    body: bytes,
    *,
    now: int,
    max_skew_seconds: int = MAX_TIMESTAMP_SKEW_SECONDS,
) -> bool:
    """Return True only for a fresh signature that matches this exact body."""
    if abs(now - timestamp) > max_skew_seconds:
        return False
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(signature, canonical_message(timestamp, body))
    except InvalidSignature:
        return False
    return True


def resolve_public_key() -> Ed25519PublicKey | None:
    """Public key the worker verifies against, or None if none is configured."""
    public_b64 = os.environ.get("WORKER_SIGNING_PUBLIC_KEY", DEFAULT_PUBLIC_KEY_B64)
    if not public_b64:
        return None
    return load_public_key(public_b64)
