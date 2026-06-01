# v2 LLM worker

This directory contains a game-agnostic local worker for FLAS/Gemma inference.
The existing `app.py` is left unchanged.

## Why the worker is separate

The v1 app keeps game state, HTTP routing, and the loaded FLAS/Gemma model in one
process. The v2 worker owns only model loading, request scheduling, batching, and
primitive inference operations. A game server can call it over local HTTP now and
over an authenticated network boundary later.

## Steering-vector notes

The standard activation-steering literature does not treat random vectors as a
best-practice semantic steering method. Common approaches derive vectors from
model activations:

- Activation Addition computes a steering vector from contrasting prompt pairs
  and adds it during inference:
  https://openreview.net/forum?id=2XBPdPIcFK
- Mean-centering improves steering by subtracting the mean activation of a
  broader distribution from target activations:
  https://arxiv.org/abs/2312.03813
- Reliability work finds steering works best when activation differences align
  with a coherent direction:
  https://arxiv.org/abs/2505.22637

The `{"type":"random"}` concept is therefore implemented as a reproducible
experimental/control input in FLAS concept-hidden space, not as a claim that
random directions are semantically meaningful. The default samples standard
normal vectors and normalizes each token vector to unit RMS before applying
`scale`, which matches the scale convention used by RMS-normalized transformer
hidden states better than raw unbounded Gaussian magnitudes.

## Authentication and priority

The worker serves everyone: unsigned requests run at normal priority with no
limit. Requests carrying a valid, fresh Ed25519 signature (from the coworld
game, which holds the private key) run at tournament priority and preempt
public traffic. The worker holds only the public key. See
`v2/coworld/docs/worker_auth.md` for the scheme, key delivery, and TLS exposure.

## Run

From the repo root:

```bash
.venv/bin/python v2/llm_worker.py --host 127.0.0.1 --port 7870
```

Load the model:

```bash
curl -sS -X POST -H 'Content-Type: application/json' \
  -d '{}' http://127.0.0.1:7870/load
```

Health:

```bash
curl -sS http://127.0.0.1:7870/health
```

Generate:

```bash
curl -sS -X POST -H 'Content-Type: application/json' \
  -d '{
    "requests": [{
      "prompt": "Answer directly: what color is a clear daytime sky?",
      "concept": {"type": "random", "seed": "style-17", "tokens": 64, "scale": 1.0},
      "flas": {"flowtime": 2.0, "steps": 3},
      "sampling": {"max_tokens": 40, "temperature": 0.7}
    }]
  }' http://127.0.0.1:7870/generate
```

Choice logprobs:

```bash
curl -sS -X POST -H 'Content-Type: application/json' \
  -d '{
    "requests": [{
      "prompt": "Choose the answer that best fits the evidence.",
      "concept": {"type": "text", "text": "terse technical documentation"},
      "flas": {"flowtime": 2.0, "steps": 3},
      "choices": ["blue", "green", "gray"],
      "ordering": {"mode": "all_permutations"}
    }]
  }' http://127.0.0.1:7870/choice-logprobs
```

Microbatch benchmark:

```bash
curl -sS -X POST -H 'Content-Type: application/json' \
  -d '{
    "batch_sizes": [1, 2, 4],
    "prompt_tokens": 128,
    "max_tokens": 32,
    "concept": {"type": "random", "seed": "bench", "tokens": 64},
    "flas": {"flowtime": 2.0, "steps": 3}
  }' http://127.0.0.1:7870/bench/microbatch
```

The benchmark performs repeated internal trials but does not expose that knob in
the HTTP API.

