# Cue-n-Woo Coworld

This folder contains a Coworld-oriented version of the cue-n-woo steering game. It is separate from the local browser
server in `v2/game_server.py`.

The Coworld game runnable:

- reads runtime config from `COGAME_CONFIG_URI`;
- serves `GET /healthz`;
- serves `GET /client/player?slot=...&token=...`;
- accepts player WebSockets at `/player?slot=...&token=...`;
- serves `GET /client/global` and `/global`;
- writes results to `COGAME_RESULTS_URI`;
- writes replay JSON to `COGAME_SAVE_REPLAY_URI`;
- calls Bedrock Claude Sonnet directly for hidden-judge answers and scoring.

The game config controls token limits, temperature, Bedrock model/region, concept source, scoring sample count, and a hard round timer. The
default question limit is 256 simple tokens. When the game cannot use a provider tokenizer, per-item limits are enforced
with a 4-character token estimate. The default round timer is 600 seconds. The default concept source is `axis_combo`: it samples several JSON axes from
`data/concept_axes/`, chooses one option from each sampled axis, and joins those phrases into the hidden steering
concept. Existing `list`, `specific`, and `random` concept sources are still supported.

Policy debugging notes:

- `docs/llm_player_observation.md` documents the exact Bedrock Converse request and the state the LLM player sees.
- Judge responses are capped by `judge_max_tokens`; the Claude policy prompt explicitly tells the model that prior
  judge answers may be truncated at that output-token limit.

Scoring notes:

- If two submitted answers are exact duplicates, or if one answer is a full string prefix of the other after whitespace
  and case normalization, the game treats them as the same conflicting answer. The scorer keeps the shortest conflicting
  answer as `canonical_answer` and skips the Sonnet choice samples. In the current two-player game, both matching answers
  receive 40 points: 50 shared-probability base points minus a 10 point duplicate-answer penalty.
- Otherwise, each distinct answer on a challenge is scored from `scoring_samples` forced-choice Bedrock Claude Sonnet
  calls, defaulting to 9 samples: `100 * sampled_probability_of_that_answer`,
  plus a 10 point bonus for each distinct competing answer it beats. In the current two-player game that is either 0 or
  10 bonus points per answer. The player score is the sum of the points earned by that player's answers, regardless of
  who submitted the question.
