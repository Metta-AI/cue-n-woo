# Cue-n-Woo Player Protocol

Players connect to the Coworld game WebSocket at `/player?slot=<slot>&token=<token>`.

Players are addressed by slot index (`0`, `1`, ...) matching `config.players`. Player-indexed fields
(`public_questions`, `counts`) are lists ordered by slot.

The server sends JSON observations:

```json
{
  "type": "state",
  "slot": 0,
  "phase": "private_questions",
  "remaining_seconds": 300,
  "counts": [
    {"chats": 0, "proposals": 0, "answers": 0},
    {"chats": 0, "proposals": 0, "answers": 0}
  ],
  "me": {
    "judge": [],
    "proposals": [],
    "answers": []
  },
  "opponent_questions": [],
  "public_questions": [[], []],
  "results": null
}
```

`me.judge` is the player's private transcript with the steered judge (question/answer turns).

The hidden steering concept is not sent to players while the game is active.

Players send one of these actions:

```json
{"type": "ask", "question": "..."}
```

```json
{
  "type": "propose",
  "proposals": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}
```

```json
{"type": "answer", "answers": ["...", "...", "..."]}
```

An empty answer string is a permitted decline and scores 0.

The server replies with a fresh `state` message after each accepted action, or:

```json
{"type": "error", "error": "..."}
```
