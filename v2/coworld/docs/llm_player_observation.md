# LLM Player Observation

`v2/coworld/players/baseline.py` is the Coworld LLM player harness. It connects to the game server over the player WebSocket and calls Amazon Bedrock Converse with Claude Sonnet. Other players (e.g. `v2/coworld/players/kyle.py`) reuse this harness, optionally passing non-binding per-phase advice the model may ignore.

## Bedrock Request

For each decision, Claude receives exactly one user message and one forced tool:

- `modelId`: `us.anthropic.claude-sonnet-4-6` by default, overridable with `BEDROCK_CLAUDE_MODEL_ID` or `BEDROCK_MODEL`.
- `inferenceConfig`: `{"maxTokens": 1024}`.
- `toolChoice`: forced to `submit_action`.
- The model must return a `submit_action` tool call.

The policy does not pass temperature or other sampling settings to Claude. Game-side generation settings such as the judge token cap, Bedrock model/region, scoring sample count, and concept selection are controlled by the Coworld game config and are shown to the policy only through the observation when appropriate.

## Prompt Text

The user message is assembled in this order:

1. `game_rules_for_policy()` from `harness.py`.
2. A judge response-limit notice:

   ```text
   Judge response limit: the judge's generated answer to each private question is limited to <judge_max_tokens> output tokens. If you bundle many subquestions, the judge may run out of tokens before answering all of them. Treat missing or cut-off text as unavailable information, not as a deliberate answer.
   ```

3. `Private transcript so far`, built from `state["me"]["judge"]` except during `proposals`, where it is deliberately empty.
4. `Current observation JSON`, built by `compact_state(state)`.
5. The previous game validation error, if any.
6. Action instructions for the current phase.
7. Optional per-phase advice, only if the player supplied any (see below).

## Optional Advice

`ClaudePolicy(advice=...)` takes a `{phase: text}` map of non-binding suggestions. When the current phase has advice, it is appended to the prompt under an explicit "Optional suggestion (not a requirement — you are making the real decision …)" preamble, so the model may use, adapt, or ignore it. The baseline player passes no advice; `players/kyle.py` supplies starter questions for `private_questions` and `proposals`. No advice ever constrains the action — the LLM makes every real decision.

## Private Transcript

The private transcript section is rendered separately from the JSON to make prior judge responses hard to miss during phases that may use it:

```text
Private transcript so far:
Q1: <first private question>
A1: <first judge response>

Q2: <second private question>
A2: <second judge response>
```

If the judge's previous response was truncated by `judge_max_tokens`, Claude sees only the truncated text stored by the game server. The policy now tells Claude that truncated or missing text is unavailable information.

During `proposals`, the transcript is redacted before prompt construction and `compact_state(state)["me"]["judge"]` is also empty. Challenge-writing should be a fresh turn: private questions, private prompts, and prior judge answers must not influence the proposed challenges.

## Observation JSON

`compact_state(state)` includes:

- `phase`
- `remaining_seconds`
- `limits`
- `slot`
- `me`
- `opponent_questions`
- `public_questions`
- `counts`

For a player, `me` includes that player's private judge transcript, submitted proposals, and submitted answers, except that `me.judge` is empty during `proposals`. Player-indexed fields (`public_questions`, `counts`) are lists ordered by slot:

```json
{
  "phase": "private_questions",
  "remaining_seconds": 240,
  "limits": {
    "max_answer_tokens": 12,
    "max_question_tokens": 256,
    "judge_max_tokens": 128
  },
  "slot": 0,
  "me": {
    "judge": [
      {
        "question": "previous private question",
        "answer": "previous judge response"
      }
    ],
    "proposals": [],
    "answers": []
  },
  "opponent_questions": [],
  "public_questions": [[], []],
  "counts": [
    {"chats": 1, "proposals": 0, "answers": 0},
    {"chats": 0, "proposals": 0, "answers": 0}
  ]
}
```

## Sequencing

During `private_questions`, the policy sends one `ask` action, then waits for the game server's next WebSocket state before deciding again. The game server awaits the Bedrock judge response before appending to `state["me"]["judge"]` and broadcasting the next state. Therefore Claude's second private-question decision should include the judge's first response, and Claude's third decision should include the first two responses.

If Claude repeats a near-duplicate private question, the expected bug surface is the policy prompt or model behavior, not missing sequencing.
