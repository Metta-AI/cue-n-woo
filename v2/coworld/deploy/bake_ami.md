# Baking the ready-to-serve worker AMI

New worker replicas must start serving in a couple of minutes, not an hour. The
slow part of a cold boot is **re-downloading ~18GB of Gemma-2-9b weights from
HuggingFace** plus reinstalling the Python deps. We avoid both by baking an AMI
that already has everything on disk, and pointing the SkyServe fleet at it
(`worker_service_fast.sky.yaml`).

## What the AMI must contain

- The project venv with all deps installed (`~/cue-n-woo/.venv`, ~5GB).
- The FLAS checkpoint (`~/cue-n-woo/checkpoints/...`, ~0.5GB).
- The base model weights in the HF cache (`~/.cache/huggingface`, ~18GB) — this
  is the expensive bit. The model must have been loaded at least once on the
  source box so the weights are cached locally.

## Procedure

1. Start from a fully-set-up, model-loaded worker box (e.g. the box launched from
   `worker.sky.yaml` after a successful `/load`). Confirm the cache is populated:

   ```bash
   ssh ubuntu@<box> 'du -sh ~/.cache/huggingface ~/cue-n-woo/.venv ~/cue-n-woo/checkpoints'
   ```

2. Create an AMI without rebooting (files are already flushed to disk):

   ```bash
   AWS_PROFILE=softmax aws ec2 create-image \
     --instance-id <i-...> \
     --name "cue-n-woo-worker-ready-$(date +%s)" \
     --no-reboot \
     --tag-specifications 'ResourceType=image,Tags=[{Key=project,Value=cue-n-woo}]' \
     --query ImageId --output text
   ```

3. Wait for the image to become `available`:

   ```bash
   AWS_PROFILE=softmax aws ec2 wait image-available --image-ids <ami-...>
   ```

4. Point `worker_service_fast.sky.yaml` at the new `image_id` and (re)launch the
   fleet. A replica then only loads the model from local disk before `/health`
   reports ready.

## Re-baking

Re-bake when the worker dependencies or the model change. A pure code change to
`v2/llm_worker.py` does NOT require a re-bake — `worker_service_fast.sky.yaml`
rsyncs the workdir code over the AMI's repo copy at boot, keeping the baked venv
and weights.
