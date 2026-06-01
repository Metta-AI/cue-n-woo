# cue-n-woo

Local web UI for a small theory-of-mind question game (formerly "steering-game") backed by FLAS activation steering
with Gemma 2 9B IT.

The app loads:

- FLAS checkpoint: `flas-ai/flas-gemma-2-9b-it`
- Public Gemma mirror: `unsloth/gemma-2-9b-it`

It serves a browser UI where:

- Alice and Bob use separate endpoints: `/alice` and `/bob`.
- Each endpoint can ask Charlie exactly three private questions. Charlie answers each with up to 100 tokens.
- Alice and Bob then each propose three challenge questions plus hidden answers.
- Once both sets of questions are submitted, each endpoint sees only the opponent's questions and submits blind answers.
- Charlie is steered by a randomly selected hidden style/concept. Alice and Bob do not see it until reveal.
- When both answer sheets are submitted, the game reveals everything, including Charlie's hidden concept, and scores the round.

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
- Charlie and scoring use fixed FLAS settings: `flowtime=2`, `steps=3`, `temperature=0.7`.
- The hidden concept is sampled from a broad style pool, including technical documentation, therapist, scientific reviewer, noir detective, pirate, legal analysis, storybook, philosophy, marketing, field report, fantasy chronicle, cyberpunk, comedy, Zen minimalism, finance, Victorian letter, sports commentary, conspiracy speculation, cooking show, bureaucratic formality, and additional styles.
- Alice/Bob/Charlie are UI/game labels only. Prompts sent to the steered model use neutral record/question wording and sanitize those role names from user-provided text.
- The game state is in memory. Restarting the server resets the round.
- Eve is not in this corrected role-separated flow yet; the current implementation is Alice/Bob/Charlie only.
