# Optional LLM Player Observation

`v2/coworld/players/baseline.py` is an optional Coworld player harness. It is not part of the Cue-n-Woo game server or judge. The game server talks to the FLAS/Gemma worker for judge answers and scoring; this harness is only a bundled automated player that connects over the same player WebSocket as any other player.

The baseline harness calls Amazon Bedrock Converse by default. Other players (e.g. `v2/coworld/players/kyle.py`) reuse this harness, optionally passing non-binding per-phase advice the policy model may ignore.

## Bedrock Request

For each automated-player decision, the policy model receives exactly one user message and one forced tool:

- `modelId`: `us.anthropic.claude-opus-4-6-v1` by default, overridable with `BEDROCK_CLAUDE_MODEL_ID`.
- `inferenceConfig`: `{"maxTokens": 1024}`.
- `toolChoice`: forced to `submit_action`.
- The model must return a `submit_action` tool call.

The policy request does not pass temperature or other sampling settings. Game-side generation settings such as the judge token cap, local worker temperature, FLAS strength, and concept selection are controlled by the Coworld game config and are shown to the policy only through the observation when appropriate.

## Prompt Text

The user message is assembled in this order:

1. `game_rules_for_policy()` from `harness.py`.
2. A judge response-limit notice:

   ```text
   Judge response limit: the judge's generated answer to each private question is limited to <judge_max_tokens> output tokens. If you bundle many subquestions, the judge may run out of tokens before answering all of them. Treat missing or cut-off text as unavailable information, not as a deliberate answer.
   ```

3. `Private transcript so far`, built from `state["me"]["judge"]`.
4. `Current observation JSON`, built by `compact_state(state)`.
5. The previous game validation error, if any.
6. Action instructions for the current phase.
7. Optional per-phase advice, only if the player supplied any (see below).

## Optional Advice

The baseline policy class takes a `{phase: text}` map of non-binding suggestions. When the current phase has advice, it is appended to the prompt under an explicit "Optional suggestion (not a requirement — you are making the real decision …)" preamble, so the model may use, adapt, or ignore it. The baseline player passes no advice; `players/kyle.py` supplies starter questions for `private_questions` and `proposals`. No advice ever constrains the action — the LLM makes every real decision.

## Policy Visibility

The bundled automated player sees the same player-visible game state it would get over the WebSocket:

- During `private_questions`, `me.judge` contains that player's previous private questions and judge answers.
- During `proposals`, `me.judge` is still available so the player can write challenge questions and answers from its own transcript.
- During `answers`, `opponent_questions` contains the opponent's challenge questions without the opponent's hidden answers.

Replay output includes the full internal state after the episode ends.

Cue-n-Woo worker calls are separate from this optional player harness. The judge generation worker receives only the current private question. The scoring worker receives only the current challenge question and candidate answers, with no accumulated transcript/reference context.

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

Player-indexed fields (`public_questions`, `counts`) are lists ordered by slot:

```json
{
  "phase": "private_questions",
  "remaining_seconds": 240,
  "limits": {
    "max_answer_tokens": 12,
    "max_question_tokens": 1024,
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

During `private_questions`, the policy sends one `ask` action, then waits for the game server's next WebSocket state before deciding again. The game server awaits the local LLM worker response before appending to its internal state and broadcasting the next player-visible state. The bundled policy model's second and third private-question decisions include that player's earlier private questions and judge answers.
