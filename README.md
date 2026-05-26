# steering-game

Local web UI for a small theory-of-mind question game backed by FLAS activation steering with Gemma 2 9B IT.

The app loads:

- FLAS checkpoint: `flas-ai/flas-gemma-2-9b-it`
- Public Gemma mirror: `unsloth/gemma-2-9b-it`

It serves a browser UI where:

- Alice and Bob use separate endpoints: `/alice` and `/bob`.
- Each endpoint can ask Charlie exactly three private questions. Charlie answers each with up to 100 tokens.
- Alice and Bob then each propose three challenge questions plus hidden answers.
- Once both sets of questions are submitted, each endpoint sees only the opponent's questions and submits blind answers.
- When both answer sheets are submitted, the game reveals everything and scores the round.

Scoring uses the simplified rule from the prototype: present the two answers in both option orderings, then compare the softmax probability of the first token where the answer strings differ. The final probability is averaged across the two orderings.

## Setup

```bash
./scripts/setup_flas.sh
```

This creates `.venv`, installs FLAS from GitHub, and downloads the FLAS 9B checkpoint into `checkpoints/`.

## Run

```bash
./scripts/run_server.sh
```

Open:

```text
http://127.0.0.1:7860
```

Use these player endpoints:

```text
http://127.0.0.1:7860/alice
http://127.0.0.1:7860/bob
```

From a local machine connected to a remote GPU host:

```bash
ssh -N -L 7860:127.0.0.1:7860 <host>
```

## Notes

- The checkpoint is not committed. It is downloaded by `scripts/setup_flas.sh`.
- Gemma 2 9B IT + FLAS uses about 18 GB of VRAM for interactive inference.
- `flowtime` is the main steering strength for Charlie's reasoning concept. Start around `1.0` to `2.0`.
- `steps` controls numerical integration quality more than strength. The released checkpoint defaults to `3`.
- The game state is in memory. Restarting the server resets the round.
- Eve is not in this corrected role-separated flow yet; the current implementation is Alice/Bob/Charlie only.
