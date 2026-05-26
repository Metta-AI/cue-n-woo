# steering-game

Small local web UI for FLAS activation steering with Gemma 2 9B IT.

The app loads:

- FLAS checkpoint: `flas-ai/flas-gemma-2-9b-it`
- Public Gemma mirror: `unsloth/gemma-2-9b-it`

It serves a browser UI where you enter:

- a user prompt
- a natural-language steering concept
- FLAS `flowtime` strength
- Euler integration step count

Each response is paired with the top 10 first-token probabilities computed from the logits directly, independent of sampling temperature.

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

From a local machine connected to a remote GPU host:

```bash
ssh -N -L 7860:127.0.0.1:7860 <host>
```

## Notes

- The checkpoint is not committed. It is downloaded by `scripts/setup_flas.sh`.
- Gemma 2 9B IT + FLAS uses about 18 GB of VRAM for interactive inference.
- `flowtime` is the main steering strength. Start around `1.0` to `2.0`.
- `steps` controls numerical integration quality more than strength. The released checkpoint defaults to `3`.
