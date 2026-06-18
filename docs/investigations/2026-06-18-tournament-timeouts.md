# Cue-n-Woo tournament timeouts — investigation & fixes (2026-06-18)

## Summary

Players in the Cue N Woo league were being disqualified at high rates because
episodes were not finishing inside the configured `round_timeout_seconds` (600s).
Two distinct failure modes are happening, both ultimately driven by **Amazon
Bedrock (Claude Sonnet) throttling and latency** in the episode-running AWS
account:

- **Mode A — legitimate slow rounds, clean DQ at ~600s (dominant).** The round
  timer fires correctly but a player (or the judge) could not complete a phase in
  time, so the stalled slot gets the `-100` inactivity penalty
  (`INACTIVE_TIMEOUT_PENALTY`, `v2/coworld/game.py`).
- **Mode B — zombie episodes running ~1210s (the "wonky" games).** The in-game
  600s timer does **not** fire; the episode runs until the external Kubernetes
  Job `activeDeadlineSeconds` (1200s) hard-kills the pod, surfacing as
  `failed`/`cancelled`/`pod_not_found` episodes and `failed` rounds.

## Evidence

All figures from the Observatory Postgres (read-only) + S3 episode artifacts,
over ~2 days ending 2026-06-18. League `Cue N Woo`
(`league_e28faac2-d187-4526-b73b-432c43943aed`), commissioner key `container`.

### Symptom quantification

- **~13% of completed cue episodes** ran ≥595s (the ~600s wall). The rate is
  bursty and spiking: by day it went 5.3% → 2.0% → **19.1%** (06-16→17→18), and
  individual hours on 06-18 hit **30–77%** timeouts.
- **~312 cue episodes ended as `failed`** clustering at ~1000–1250s:
  `cancelled` (215, median 1025s), `unknown` (87, median 1224s),
  `pod_not_found` (8, 1247s), `container_failed` (2). These are Mode B.
- Episode runtime percentiles: p50 85.9s, **p95 610.8s, p99 723.9s, max 1363s**.
  p95/p99 sit on top of / past the 600s timer.
- Many DQs occur in `private_questions` — a phase that is only 3 `ask` calls and
  makes **zero scoring calls**. The only thing that can make that phase take 10
  minutes is the **judge** (Bedrock) not responding. This is the cleanest proof
  that the bottleneck is Bedrock, not player compute.

### Root-cause evidence

- **Throttling is present and severe.** Episode `game.stdout.log` artifacts show
  explicit `Retrying Bedrock referee call after ThrottlingException …`. In a
  random sample of 30 near-timeout episodes, **15 showed ThrottlingException**;
  individual games logged **up to 218** throttle/retry lines.
- **Bedrock latency is high.** CloudWatch `AWS/Bedrock InvocationLatency`:
  p50 ~3–4.5s, **p99 ~20–37s** per call (measured in account 751442549699; the
  episode account shares the same regional Sonnet capacity).
- **Per-episode Bedrock call volume is large.** Each episode issues ~6 judge
  answers + `challenge_questions_per_player(3) * 2 players * scoring_samples(9)`
  = **54 forced-choice scoring calls** ≈ **60 Bedrock calls/episode**, with
  dozens of episodes running concurrently (peak ~62 simultaneous). Under
  throttling this volume cannot complete in 600s.

### Mode B mechanism (timer fails to fire)

- The round deadline is enforced by `timer_loop` (`game.py`), a cooperative
  `asyncio` task that each second calls `await broadcast()`.
- `broadcast()` did an **unbounded** `await ws.send_json(...)` per connection.
  A player socket that stays open but stops reading (TCP backpressure) makes that
  await hang forever; `suppress(Exception)` does not catch a hang.
- A hung broadcast freezes `timer_loop`, so `remaining_seconds() <= 0` is never
  re-checked → the 600s deadline never fires → the episode runs until the metta
  dispatcher's `active_deadline_seconds = JOB_TIMEOUT_SECONDS = 20*60 = 1200s`
  (metta `app_backend/.../job_runner/dispatcher.py`) kills the pod.
- A timed-out episode's `results.json` recorded `duration_seconds: 1209.96` with
  `status: timeout` — i.e. `finalize` only ran at pod shutdown (~1210s), not at
  600s. The game stdout showed both players connect, then silence until SIGTERM.

## Fixes applied in this repo (branch `boggsj/cue-n-woo-timeout-hardening`)

1. **Bound every broadcast send** (`game.py`): new `send_json_bounded()` wraps
   each `send_json` in `asyncio.wait_for(..., WEBSOCKET_SEND_TIMEOUT_SECONDS=5)`,
   so one wedged socket can no longer stall `timer_loop` or the per-action
   broadcast. This converts Mode B zombie episodes into clean ~600s DQs,
   eliminating the ~1200s pod-kill failures and halving the worst-case resource
   hold. Regression test: `v2/tests/test_broadcast_timeout.py`.
2. **Reduce `scoring_samples` 9 → 5** (manifest + template defaults, default
   variant, certification, `game.py` fallback, docs). Cuts per-episode scoring
   calls 54 → 30 (~44%), reducing aggregate regional Bedrock demand and therefore
   throttling for all episodes. 5 is odd (no ties). **Tradeoff:** a noisier
   forced-choice probability estimate per challenge — this is a deliberate
   product/fairness call and is isolated in its own commit so it can be reverted
   independently. Manifest `version` bumped 0.2.26 → 0.2.27.

## Flagged for owners — NOT changed here (cross-repo / infra)

- **600s game timer vs 1200s k8s Job deadline mismatch** (metta
  `dispatcher.py:JOB_TIMEOUT_SECONDS`). With the broadcast fix the game should now
  self-terminate at 600s, but the 2x gap is worth reconciling so a genuinely
  stuck game is killed nearer its own deadline.
- **Bedrock Sonnet quota in the tournament account (583928386201).** The decisive
  number — regional TPM/RPM vs. concurrent episode demand. I could not read it:
  my SSO user has **no permission-set assignment** in 583928386201 (both
  `tournament` and `tournament-admin` profiles return `GetRoleCredentials: No
  access`). Needs an admin to (a) grant access or (b) report
  `InvocationThrottles` + `service-quotas` for Claude Sonnet 4. A quota increase
  or Provisioned Throughput may be the real cure if demand exceeds quota.
- **Dispatch concurrency / round scheduling** (commissioner repo
  `commissioners-cue-n-woo`). Throttling scales with simultaneous episodes;
  throttling episode dispatch would reduce peak Bedrock pressure.
- **Referee retry budget** (`game.py:_converse_with_retry`). It retries transient
  Bedrock errors until `remaining_seconds <= 0` (the game deadline). Under a
  throttle storm this can spend most of a round in backoff. Lower-confidence as a
  primary cause (Mode B episodes were stuck on *players*, not the referee), but a
  per-call retry cap that fails fast and reserves scoring-time budget is worth
  considering.

## How to reproduce the analysis

- DB access: `kubectl --context softmax-main -n observatory exec deploy/observatory-backend -- psql "$STATS_DB_READ_ONLY_URI" …`
  (requires `AWS_PROFILE=softmax`; the read-only URI is in the pod env).
- Episode artifacts: `job_requests.result->'artifact_urls'` → `s3://observatory-private/jobs/<id>/{results.json,debug.zip,worker_timings.json,spec.json}`.
- Link cue episodes: `episode_requests` → `policy_pools` → `rounds` → `divisions` → `leagues` where `league_id = league_e28faac2-…`.
