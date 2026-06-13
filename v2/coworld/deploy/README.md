# cue-n-woo serving: autoscaling GPU worker fleet

Everything needed to serve the cue-n-woo FLAS/Gemma worker at scale: a single
box for benchmarking, an autoscaling SkyServe fleet for production, the
pre-baked fast-boot image, and the SkyPilot patch our AWS account requires.

The game itself is unchanged — it talks to one `llm_worker_url`. That URL just
points at the fleet's load balancer instead of a single box.

## Files

| File | Purpose |
|------|---------|
| `worker.sky.yaml` | Launch ONE worker box (benchmarking / baking the AMI). |
| `worker_service.sky.yaml` | Autoscaling SkyServe fleet, slow cold-boot (installs deps + downloads weights each boot). Reference / fallback. |
| `worker_service_fast.sky.yaml` | Autoscaling SkyServe fleet on the pre-baked AMI — replicas boot in a couple minutes. **Use this for the event.** |
| `bench_batching.py` | Correctness + throughput check for cross-concept batching (`python v2/coworld/deploy/bench_batching.py` against a running worker). |
| `bake_ami.md` | How to bake the ready-to-serve AMI (venv + checkpoint + cached weights). |
| `skypilot_patch/` | Patch + apply script that lets SkyPilot launch with no instance profile in our account. |

## One-time / per-environment setup

### 1. Patch SkyPilot (required in this AWS account)

No available identity in account `751442549699` has `iam:PassRole`, so SkyPilot
cannot attach its default `skypilot-v1` instance profile and every launch fails.
Our workers need no AWS permissions at runtime (they serve a model from local
disk), so we launch with **no instance profile**, gated by an env var.

```bash
# Patch the active SkyPilot install (idempotent). Re-run after any `uv sync`/
# venv rebuild — it wipes the patch.
python v2/coworld/deploy/skypilot_patch/apply_patch.py
python v2/coworld/deploy/skypilot_patch/apply_patch.py --check   # 0 = patched
```

Then **always** launch with the env var set (and the AWS profile that can
DescribeImages):

```bash
export SKYPILOT_SKIP_INSTANCE_PROFILE=1
export AWS_PROFILE=softmax
```

### 2. Bake the fast-boot AMI

See `bake_ami.md`. Produces an AMI with the venv, FLAS checkpoint, and base
Gemma-2-9b weights already on disk, so a replica boots → loads from local disk →
serves in ~1-2 min instead of ~1 hour. Put the AMI id into
`worker_service_fast.sky.yaml`'s `image_id`.

## Run the fleet

```bash
export SKYPILOT_SKIP_INSTANCE_PROFILE=1 AWS_PROFILE=softmax
# from the repo root (workdir-free fast spec is self-contained on the AMI):
uv run --project /home/kyleherndon/metta sky serve up \
  -n cue-n-woo-workers v2/coworld/deploy/worker_service_fast.sky.yaml

uv run --project /home/kyleherndon/metta sky serve status cue-n-woo-workers
uv run --project /home/kyleherndon/metta sky serve down cue-n-woo-workers
```

`sky serve status` prints the load-balancer endpoint. Point the game's
`llm_worker_url` (manifest config) at it, or repoint the stable public DNS
ingress (`v2/coworld/deploy/worker-ingress.yaml`) at it so the URL is durable.

## How autoscaling behaves (from SkyServe internals)

A controller box (cheap CPU instance) runs a load balancer + autoscaler:
- LB routes each request to a ready replica (health-gated via `/health`).
- Autoscaler keeps a 60s QPS window; `desired = ceil(qps / target_qps_per_replica)`,
  clipped to `[min_replicas, max_replicas]`, plus `num_overprovision` warm spares
  ahead of demand. Scale-up/down only fire after `upscale_delay`/`downscale_delay`
  of sustained signal (hysteresis), so brief spikes/lulls don't thrash.

Calibrated values (single L40S, measured): a game ≈ 18 worker requests; a replica
saturates near ~8 concurrent games (~1.6 req/s) → `target_qps_per_replica ≈ 1.3`.

## Cost knobs

g6e.xlarge (1× L40S) is the most price-efficient unit (~$1.86/GPU-hr); multi-GPU
boxes bundle unused CPU/RAM at a higher per-GPU price. Scale `max_replicas` for
the event ceiling and `num_overprovision` for warm-spare headroom.

## Operational gotchas (learned in validation)

- **`num_overprovision` only applies on a fresh `sky serve up`, NOT `sky serve update`.**
  A rolling update keeps the *previous* overprovision value in the running
  autoscaler even though the launch banner echoes the new one. To change warm-spare
  count, tear the service down and bring it back up.
- **Misleading log line:** the controller logs `Final target number of replicas: N`
  using the pre-overprovision count. The real provisioning target is
  `N + num_overprovision` — trust `sky serve status` (e.g. `0/9`) over that log field.
- **Warm-spare math:** at idle (qps 0), target = max(min_replicas, ceil(qps/target_qps))
  = min_replicas, then + num_overprovision. So `min_replicas: 1` + `num_overprovision: 8`
  = 9 always-warm replicas; scales up as demand pushes the QPS target above min.
- **Cross-region AMIs:** `any_of` fallback regions each need their own AMI copy
  (AMIs are region-scoped). Re-copy + update the spec's region image_id after any re-bake.
- **FSR is per-snapshot:** after a re-bake, enable Fast Snapshot Restore on the NEW
  snapshot and disable it on the old one (it bills per-AZ while armed).
