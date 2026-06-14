# Cue-n-Woo serving: architecture, decisions, gotchas, runbook

This is the single source of truth for the autoscaling GPU worker fleet that
serves the cue-n-woo FLAS/Gemma worker for high-concurrency events (e.g. game of
the week). Read this before touching the fleet.

---

## 1. Architecture in one picture

```
game (llm_worker_url)
      │  small JSON: prompts in, generated text / choice-logprobs out
      ▼
SkyServe load balancer  ◀── single frontend, one stable endpoint
      │  round-robin across READY replicas; health-gated by /ready
      ├──► replica 1 (g6e.xlarge, 1× L40S)  — worker serves from local disk
      ├──► replica 2
      └──► replica N        (autoscaled 1…16, +warm spares)
            ▲
  SkyServe controller (cheap c6i.xlarge CPU box):
    - load balancer process (routes requests)
    - autoscaler process (adds/removes replicas on 60s QPS window)
    - holds the AWS credentials (replicas hold none)
```

- **One worker = one L40S = one model copy.** The workload is GPU-compute-bound
  (autoregressive decode), so scaling = more single-GPU replicas, each batching
  internally. `g6e.xlarge` is the most price-efficient unit (~$1.86/GPU-hr).
- **The game is unchanged**: it already targets a single `llm_worker_url`; that
  just becomes the SkyServe endpoint (or a stable DNS fronting it).

---

## 2. Why each non-obvious decision was made

| Decision | Why |
|---|---|
| **SkyServe** (vs raw EC2 fleet) | Gives LB + autoscaler + health-gated routing + replica lifecycle for free; we'd otherwise hand-build all of it. |
| **Single-L40S replicas** (not 8×L40S boxes) | Most price-efficient per GPU; multi-GPU boxes bundle unused CPU/RAM at higher $/GPU. Finest autoscaling granularity. |
| **Baked AMI** (model in the image) | Avoids re-downloading ~18GB weights from HF on every boot. Boot = load-from-local-disk only. |
| **Minimal 40GB AMI on plain Ubuntu** (not the Deep Learning AMI) | The DLAMI ships ~41GB of system CUDA toolkits we don't need — torch bundles its own CUDA, only the NVIDIA *driver* is required. Trimmed 200GB→40GB. |
| **Fast Snapshot Restore (FSR)** | Fresh EBS volumes from a snapshot lazy-load blocks from S3 at ~19 MB/s, so the first 18GB read *hangs* (replicas never become ready). FSR pre-warms the snapshot so volumes launch at full speed. Billed per-AZ while armed. |
| **`/ready` endpoint** (separate from `/health`) | `/health` is liveness (200 if process up). `/ready` is 200 only when the model is loaded, so the LB never routes to a cold/loading replica. Avoids `FAILED_INITIAL_DELAY` and request hangs. |
| **SSO creds on the controller, none on workers** | No identity in this account has `iam:PassRole`, so no instance profile can be attached. The controller needs creds to launch replicas; workers make no AWS calls, so they get none (blast-radius limit). |
| **SkyPilot patch (skip instance profile)** | SkyPilot always attaches its `skypilot-v1` instance profile; with no `PassRole`, every launch fails. The patch skips the attach (gated by a marker file). |
| **min 1 + num_overprovision** | Keep warm spares ahead of demand so a surge hits already-booted GPUs instead of a multi-minute cold start. |

---

## 3. The credentials situation (most fragile piece)

- **Workers: zero AWS credentials** (serve a model from local disk).
- **Controller: the operator's SSO credentials**, placed manually:
  `~/.aws/config` (softmax profile + a mirrored `[default]` so boto resolves with
  no `AWS_PROFILE`), and `~/.aws/sso/cache/*.json` (access token + refresh token).
- **It self-refreshes unattended** (boto uses the refresh token) until the SSO
  *client registration* expires (~16 days from setup). Verified by forcing an
  expired access token and watching boto re-mint it.
- Full detail + the re-copy procedure if it lapses: `controller_credentials.md`.

---

## 4. GOTCHAS (the expensive-to-relearn ones)

1. **`num_overprovision` only applies on a fresh `sky serve up`, NOT `sky serve
   update`.** A rolling update keeps the previous overprovision value in the live
   autoscaler even though the launch banner echoes the new one. To change warm
   spares: tear down and bring up.

2. **Misleading controller log:** `Final target number of replicas: N` prints the
   *pre-overprovision* count. The real provisioning target is `N +
   num_overprovision`. Trust `sky serve status` (e.g. `0/9`) over that log line.

3. **FSR is per-snapshot and per-AZ.** After any re-bake: enable FSR on the NEW
   snapshot in every AZ you launch in, and disable it on the old snapshot (it
   bills per-AZ while armed). A replica launched before FSR reaches `enabled`
   will hang on lazy-load.

4. **AMIs are region-scoped.** The `any_of` multi-region fallback needs a separate
   AMI copy per region; update each region's `image_id` in the spec after a
   re-bake. (us-east-2 has **0** G-instance quota — do not list it.)

5. **The SkyServe controller must never autostop.** Default is 10-min idle
   autostop, which kills the fleet. Disabled via `~/.sky/config.yaml`
   (`serve.controller.resources.autostop: false`) for new controllers, and on a
   running controller via skylet `set_autostop(-1)` (the CLI blocks autostop
   changes on controllers).

6. **The SkyPilot patch + marker + creds do NOT survive a controller rebuild.**
   If the controller is recreated, re-apply: patch (`skypilot_patch/apply_patch.py`
   — though the code syncs from the launching box), `touch ~/.sky/skip_instance_profile`,
   and re-copy the SSO creds. See the runbook below.

7. **Fresh GPU boxes need ~1-2 min for the NVIDIA driver to self-initialize**
   after boot (and the driver only loads after the box has booted, not at AMI
   bake time on a non-rebooted source). `/ready` gating absorbs this naturally.

8. **The worker is a `BaseHTTPRequestHandler`.** It matched `self.path ==
   "/generate"` exactly — but the SkyServe LB forwards `/generate?` (trailing `?`
   from an empty query). Fixed with `route_path` (strips the query before
   matching). Any new route must use `route_path`, not `self.path`.

9. **Detaching a process over SSH is flaky** (nohup/`&`/`setsid` frequently drop
   the worker). Use `systemd-run --unit=... ` or run inside the SkyServe
   run-script's supervisor loop instead.

---

## 5. Tradeoffs accepted

- **FSR cost:** ~$0.75/hr per AZ per snapshot while armed (4 AZs ≈ $3/hr). Tiny
  next to GPU spend; buys instant full-speed boots and capacity resilience.
- **Warm spares cost:** `num_overprovision` idle GPUs run continuously (8 spares
  ≈ $15/hr idle). The price of zero cold-start during surges.
- **SSO-on-controller vs instance profile:** chosen because no `PassRole` is
  available. Tradeoff: a long-lived box holds an SSO session (PowerUser). Scoped
  to the controller only; workers stay credential-less. A scoped IAM role +
  `PassRole` grant would be cleaner if an admin provides it.
- **Disk trim required a full fresh bake** (~1hr): AWS won't shrink a volume below
  its source snapshot size, so 200GB→40GB meant rebuilding on a small root from
  scratch (re-download weights). One-time cost.
- **Boot speed depends on FSR being armed**, which takes ~up to 1hr after enabling
  on a new snapshot. Plan re-bakes ahead of an event.

---

## 6. Current state (as of this writing)

- **Worker AMI (us-east-1):** `ami-0cfa74772c613a2df` (minimal 40GB, `/ready` +
  rich `/health` + route_path fix). Snapshot `snap-0c0565304c794fac8`, **FSR
  enabled in us-east-1 a/b/c/d**.
- **Worker AMI (us-west-2):** `ami-0ae341a851559e7c7` (copy for multi-region
  fallback; FSR NOT yet enabled there).
- **Service `cue-workers-val`:** running on standby (`worker_service_standby.sky.yaml`,
  min 1 + 1 warm). The controller is `sky-serve-controller-e3640835`.
- **Quota:** G-instance On-Demand vCPU = 4096 (increase to 8192 requested);
  g6e.xlarge offered in us-east-1 a/b/c/d only.

### Validated end-to-end
- LB fans out across multiple replicas (proven 3-way, ~even round-robin).
- Zero-downtime through a replica termination (12/12 served by survivors).
- Auto-replacement of a killed replica back to full count.
- Warm-spare overprovision (min 1 + 8 → 9 warm at idle).
- FSR-warmed boot to serving in ~2 min; real `/generate` + `/choice-logprobs`
  return 200 through the LB.

---

## 7. Runbook

### Spec files (all in `v2/coworld/deploy/`)
- `worker_service_fast.sky.yaml` — **event spec**: min 1, max 16, overprovision 8,
  multi-region (us-east-1 + us-west-2), 40GB AMI, `/ready` probe, supervisor loop.
- `worker_service_standby.sky.yaml` — low-cost standby (min 1 + 1 warm).
- `worker_service_validate.sky.yaml` — single replica, for smoke tests.
- `worker.sky.yaml` / `bake_box.sky.yaml` — launch a box to benchmark / bake.
- `bench_batching.py` — throughput + cross-concept batching check.

### Bring the fleet up for an event
```bash
export AWS_PROFILE=softmax SKYPILOT_SKIP_INSTANCE_PROFILE=1 AWS_REGION=us-east-1
cd /home/kyleherndon/cue-n-woo
# (ensure us-west-2 FSR is enabled on that region's snapshot first if using it)
uv run --project /home/kyleherndon/metta sky serve up -n cue-n-woo-workers \
  v2/coworld/deploy/worker_service_fast.sky.yaml --yes
uv run --project /home/kyleherndon/metta sky serve status cue-n-woo-workers
```
If the controller is freshly created, immediately apply creds + marker (see
`controller_credentials.md`) so it can launch replicas.

### Change min/max replicas LIVE (no downtime)
`min_replicas` and `max_replicas` **hot-apply via `sky serve update`** (the
autoscaler re-reads them on a rolling update — unlike `num_overprovision`). Use
the helper so you don't hand-edit YAML mid-event:
```bash
export AWS_PROFILE=softmax SKYPILOT_SKIP_INSTANCE_PROFILE=1 AWS_REGION=us-east-1
v2/coworld/deploy/scale.sh 4 128      # min=4, max=128, applied live
v2/coworld/deploy/scale.sh 1 16       # back down
```
It writes a temp spec with the new min/max (leaving the source spec and
`num_overprovision` untouched) and runs `sky serve update --yes`. Effect is
near-immediate: the autoscaler clamps to the new bounds on its next decision tick.

### Change warm-spare count
Tear down and `serve up` again (update does NOT change `num_overprovision`).

### Watch fleet load (queries/min, replica count, queue depth, VRAM)
`fleet_metrics.py` runs **on the controller** (the only box that can reach every
replica). It polls each replica's `/health` (which reports a monotonic
`requests_served` counter, `queue_depth`, VRAM, RAM, uptime), derives
queries/min from the counter deltas, logs every sample to JSONL, and serves a
live auto-refreshing chart.
```bash
# copy to the controller and run (pick a port the controller SG already opens, 30001-30020):
scp v2/coworld/deploy/fleet_metrics.py <controller>:~/
ssh <controller> 'python3 ~/fleet_metrics.py --service cue-n-woo-workers --port 30010 --interval 15'
# then browse:  http://3.227.169.177:30010/   (EIP; /data?minutes=N for raw JSON)
```
Why poll `/health` and not SkyServe: `/autoscaler/info` exposes only
`{target,min,max}` replicas — the request-rate it uses for autoscaling is a
rolling 60s window that's never persisted. The worker's `requests_served`
counter gives a clean durable QPS series with zero worker changes. Use the
queries/min + queue-depth trend to decide whether to raise/lower min/max via
`scale.sh`.

### Re-bake the worker AMI (code/deps/model change)
See `bake_ami.md`. Then: enable FSR on the new snapshot (4 AZs), copy AMI to
us-west-2, update both `image_id`s in the spec, disable FSR on the old snapshot.

### Point the game at the fleet
Set the variant's `llm_worker_url` to the SkyServe endpoint (from `sky serve
status`), or repoint the stable DNS ingress (`worker-ingress.yaml`) at the LB so
the URL is durable.

### Tear everything down
```bash
uv run --project /home/kyleherndon/metta sky serve down cue-n-woo-workers --yes
# then disable FSR to stop per-AZ billing:
AWS_PROFILE=softmax aws ec2 disable-fast-snapshot-restores --region us-east-1 \
  --source-snapshot-ids snap-0c0565304c794fac8 \
  --availability-zones us-east-1a us-east-1b us-east-1c us-east-1d
```

---

## 8. The fleet URL (cutover target)

**https://cue-n-woo-fleet.softmax-research.net** fronts the autoscaling fleet,
parallel to the old single-box `https://cue-n-woo-worker.softmax-research.net`.

Path: DNS (Route53 alias) → nginx-ingress ELB → `cue-n-woo-fleet` Service/EndpointSlice
→ **EIP `3.227.169.177`** (on the SkyServe controller) → SkyServe LB `:30001` →
fleet replicas. Defined in `fleet-ingress.yaml` (kubectl-applied, not ArgoCD).

- TLS via cert-manager/letsencrypt (`cue-n-woo-fleet-tls`); DNS via ExternalDNS.
- **Same signing/priority token** as the old worker: the game signs with the
  private key; replicas verify with the committed public key (`signing.py`
  `DEFAULT_PUBLIC_KEY_B64`). Signed requests get tournament priority on the fleet
  too — no new token needed.
- **If the controller is replaced:** re-associate EIP `eipalloc-07b2fa3c5c37f0564`
  to the new controller instance; the URL and ingress stay unchanged.

### To cut the game over
Set the variant's `llm_worker_url` to `https://cue-n-woo-fleet.softmax-research.net`
(in `coworld_manifest.json` / `coworld_manifest_template.json`), rebuild + re-upload
the manifest. Roll back by reverting the URL to `cue-n-woo-worker...`.
