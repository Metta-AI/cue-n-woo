# Worker Authentication and Priority

The LLM worker (`v2/llm_worker.py`) runs on a GPU host and is reachable over a
public HTTPS endpoint. It intentionally serves **anyone**: an unsigned request
is processed at normal priority with no per-user limit. The only privileged
operation is **queue priority** — tournament episodes must never be starved by
public traffic.

## Threat model

The coworld game image and its config are publicly downloadable, so they must
contain no secret. We therefore split keys asymmetrically:

- The **game** (referee) holds an Ed25519 **private** key and signs each worker
  request.
- The **worker** holds only the corresponding **public** key
  (`v2/signing.py:DEFAULT_PUBLIC_KEY_B64`, overridable via
  `WORKER_SIGNING_PUBLIC_KEY`). A public key is not a secret; it ships in the
  image.

Forging a signature requires the private key, which is never shipped. A local
user running the downloaded image has no private key, so their requests are
unsigned and run at normal priority — exactly the intended behavior.

## Signature scheme

Canonical signed message: `"{timestamp}\n{sha256_hex(body)}"`.

Request headers:

- `X-Tournament-Timestamp`: unix seconds.
- `X-Tournament-Signature`: base64 Ed25519 signature over the canonical message.

The worker grants tournament priority only when the signature verifies against
the public key **and** the timestamp is within
`signing.MAX_TIMESTAMP_SKEW_SECONDS` (60s) of its clock (replay guard). A
missing, malformed, stale, or invalid signature silently falls back to normal
priority — it is never rejected.

## Key delivery to the game

`game.py` resolves the private key, in order:

1. `WORKER_SIGNING_KEY` — base64 raw seed, inline (local/dev).
2. `WORKER_SIGNING_KEY_URI` — any `read_data` URI, including
   `s3://observatory-private/...`, read with the game pod's ambient AWS
   identity (hosted).
3. None — the game runs unsigned (normal priority).

Set the `require_signing` config flag (true on the tournament variant) to turn an
unavailable key into a startup failure instead of a silent unsigned downgrade, so
a tournament never quietly forfeits its priority and competes with public
traffic. Local/certification runs leave it false and degrade to unsigned.

### Hosted runtime coupling

The hosted game container runs under the `episode-runner` Kubernetes service
account, whose IAM role `orchestrator-eval-worker` has `s3:GetObject` on
`observatory-private`. The private key lives at
`s3://observatory-private/cue-n-woo/tournament_signing_key`; set
`WORKER_SIGNING_KEY_URI` to that value in the hosted game config. If infra
changes the game pod's service account or the bucket policy, the fetch fails and
the game falls back to running unsigned.

## Exposing the worker without tailscale

The worker still binds `127.0.0.1`. Put a TLS reverse proxy (Caddy/nginx) in
front of it on the GPU host and point the manifest's `llm_worker_url` at the
public HTTPS endpoint. The bearer-free signature plus TLS is the trust boundary.

## Key rotation

1. Generate a new keypair.
2. Update the public key (commit `DEFAULT_PUBLIC_KEY_B64` or set
   `WORKER_SIGNING_PUBLIC_KEY`) and restart the worker.
3. Replace the private object the game reads.

There is no overlap window: rotate the worker public key and the game private
key together. During a brief mismatch, tournament requests degrade to normal
priority rather than failing.
