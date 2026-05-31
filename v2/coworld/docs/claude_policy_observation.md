# Claude Policy Observation

`v2/coworld/claude_policy.py` is a Coworld player policy. It connects to the game server over the player WebSocket and calls Amazon Bedrock Converse with Claude Opus 4.8.

## Bedrock Request

For each decision, Claude receives exactly one user message and one forced tool:

- `modelId`: `us.anthropic.claude-opus-4-8` by default, overridable with `BEDROCK_CLAUDE_MODEL_ID`.
- `inferenceConfig`: `{"maxTokens": 1024}`.
- `toolChoice`: forced to `submit_action`.
- The model must return a `submit_action` tool call.

The policy does not pass temperature or other sampling settings to Claude. Game-side generation settings such as Charlie token caps, local worker temperature, FLAS strength, and concept selection are controlled by the Coworld game config and are shown to the policy only through the observation when appropriate.

## Prompt Text

The user message is assembled in this order:

1. `game_rules_for_policy()` from `harness.py`.
2. A Charlie response-limit notice:

   ```text
   Charlie response limit: Charlie's generated answer to each private question is limited to <charlie_max_tokens> output tokens. If you bundle many subquestions, Charlie may run out of tokens before answering all of them. Treat missing or cut-off text as unavailable information, not as a deliberate answer.
   ```

3. `Private transcript so far`, built from `state["me"]["charlie"]`.
4. `Current observation JSON`, built by `compact_state(state)`.
5. The previous game validation error, if any.
6. Action instructions for the current phase.

## Private Transcript

The private transcript section is rendered separately from the JSON to make prior Charlie responses hard to miss:

```text
Private transcript so far:
Q1: <first private question>
A1: <first Charlie response>

Q2: <second private question>
A2: <second Charlie response>
```

If Charlie's previous response was truncated by `charlie_max_tokens`, Claude sees only the truncated text stored by the game server. The policy now tells Claude that truncated or missing text is unavailable information.

## Observation JSON

`compact_state(state)` includes:

- `phase`
- `remaining_seconds`
- `limits`
- `role`
- `me`
- `opponent_questions`
- `public_questions`
- `counts`

For a player, `me` includes that player's private Charlie transcript, submitted proposals, and submitted blind answers:

```json
{
  "phase": "private_questions",
  "remaining_seconds": 240,
  "limits": {
    "max_answer_tokens": 12,
    "max_question_tokens": 1024,
    "charlie_max_tokens": 128
  },
  "role": "alice",
  "me": {
    "charlie": [
      {
        "question": "previous private question",
        "answer": "previous Charlie response"
      }
    ],
    "proposals": [],
    "answers": []
  },
  "opponent_questions": [],
  "public_questions": {
    "alice": [],
    "bob": []
  },
  "counts": {
    "alice_chats": 1,
    "bob_chats": 0,
    "alice_proposals": 0,
    "bob_proposals": 0,
    "alice_answers": 0,
    "bob_answers": 0
  }
}
```

## Sequencing

During `private_questions`, the policy sends one `ask` action, then waits for the game server's next WebSocket state before deciding again. The game server awaits the local LLM worker response before appending to `state["me"]["charlie"]` and broadcasting the next state. Therefore Claude's second private-question decision should include Charlie's first response, and Claude's third decision should include the first two responses.

If Claude repeats a near-duplicate private question, the expected bug surface is the policy prompt or model behavior, not missing sequencing.
