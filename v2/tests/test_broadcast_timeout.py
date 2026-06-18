"""A wedged player socket must not stall broadcasts or the round timer.

If a player's WebSocket stays open but stops reading, ``send_json`` blocks
forever. An unbounded await inside ``broadcast`` would freeze ``timer_loop`` so
the configured ``round_timeout_seconds`` never fires and the episode runs until
the external Kubernetes job deadline (~1200s) instead. ``broadcast`` must bound
each send and skip a stuck socket.
"""
from __future__ import annotations

import asyncio

from v2.coworld import game


class HangingSocket:
    """Open-but-not-reading socket: send_json never completes."""

    def __init__(self) -> None:
        self.sends = 0

    async def send_json(self, payload: dict) -> None:
        self.sends += 1
        await asyncio.Event().wait()  # never set -> blocks forever


class FastSocket:
    def __init__(self) -> None:
        self.received: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.received.append(payload)


def test_broadcast_skips_wedged_socket_and_still_delivers(monkeypatch) -> None:
    monkeypatch.setattr(game, "WEBSOCKET_SEND_TIMEOUT_SECONDS", 0.05)
    hung = HangingSocket()
    healthy = FastSocket()
    monkeypatch.setattr(game.state, "connections", {0: hung, 1: healthy})
    monkeypatch.setattr(game.state, "global_connections", set())

    async def run() -> None:
        # Whole broadcast must finish promptly despite the hung socket.
        await asyncio.wait_for(game.broadcast(), timeout=2.0)

    asyncio.run(run())

    assert hung.sends == 1  # attempted once, timed out, did not block
    assert len(healthy.received) == 1  # healthy peer still received its state


def test_send_json_bounded_returns_after_timeout(monkeypatch) -> None:
    monkeypatch.setattr(game, "WEBSOCKET_SEND_TIMEOUT_SECONDS", 0.05)
    hung = HangingSocket()

    async def run() -> None:
        # Must return on its own (swallowing the timeout), not raise/hang.
        await asyncio.wait_for(game.send_json_bounded(hung, {"x": 1}), timeout=2.0)

    asyncio.run(run())
    assert hung.sends == 1
