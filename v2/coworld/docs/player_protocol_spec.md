# Steering Game Player Protocol

Players connect to the Coworld game WebSocket at `/player?slot=<slot>&token=<token>`.

The server sends JSON observations:

```json
{
  "type": "state",
  "slot": 0,
  "role": "alice",
  "phase": "private_questions",
  "remaining_seconds": 300,
  "counts": {
    "alice_chats": 0,
    "bob_chats": 0,
    "alice_proposals": 0,
    "bob_proposals": 0,
    "alice_answers": 0,
    "bob_answers": 0
  },
  "me": {
    "charlie": [],
    "proposals": [],
    "answers": []
  },
  "opponent_questions": [],
  "public_questions": {
    "alice": [],
    "bob": []
  },
  "results": null
}
```

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

The server replies with a fresh `state` message after each accepted action, or:

```json
{"type": "error", "error": "..."}
```

