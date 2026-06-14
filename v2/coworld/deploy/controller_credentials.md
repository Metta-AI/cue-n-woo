# Controller credentials (the SSO-on-controller design)

## Why the controller has credentials and the workers don't

SkyServe's controller launches and tears down GPU replicas, so it needs AWS
credentials. The workers serve a model from local disk and make **no** AWS calls,
so they get none (auth lives only on the controller — a deliberate blast-radius
limit). The controller cannot use an instance profile here (no identity in the
account has `iam:PassRole`; see `skypilot_patch/`), so instead we place the
operator's AWS SSO credentials on the controller directly.

## What was placed on the controller

- `~/.aws/config` with the `softmax` SSO profile **and** a mirrored `[default]`
  profile (so boto resolves with no `AWS_PROFILE` env var — the SkyServe
  provisioner sets none).
- `~/.aws/sso/cache/*.json` — the SSO access token **and** the client
  registration (which carries the refresh token + clientId/secret).

## Does it survive a 14-day event? Yes, unattended — within the SSO window.

SSO **access tokens are short-lived (~1 hour)**, but boto3 **auto-refreshes**
them using the refresh token in the cache, with no human in the loop. Verified on
the controller: forcing the access token to an expired timestamp and then making
an STS call transparently minted a fresh token and authenticated.

The hard ceiling is the **SSO client registration expiry**
(`registrationExpiresAt`, ~16 days out when this was set up). As long as that is
valid and the SSO session isn't revoked by an admin, the controller stays
authenticated. Re-run `softmax login` locally and re-copy `~/.aws/sso/cache/` +
`~/.aws/config` to the controller to extend before it lapses.

## If the controller credentials DO lapse mid-event

Symptom: replicas stop launching/reaping; controller logs show
`NoCredentialsError` or an SSO token error. Fix:

```bash
# locally
aws sso login --profile softmax   # or: softmax login
scp ~/.aws/config            sky-serve-controller-e3640835:~/.aws/config
scp ~/.aws/sso/cache/*.json  sky-serve-controller-e3640835:~/.aws/sso/cache/
# the [default] profile block must also be present on the controller (see above)
```

No controller restart is needed; boto re-reads the cache on the next call.

## Monitoring

Check liveness of the credential chain any time with:

```bash
ssh sky-serve-controller-e3640835 \
  '~/skypilot-runtime/bin/python -c "import boto3;print(boto3.client(\"sts\").get_caller_identity()[\"Arn\"])"'
```

A non-error ARN means the controller can still launch replicas.
