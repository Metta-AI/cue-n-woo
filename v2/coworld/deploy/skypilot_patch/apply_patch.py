#!/usr/bin/env python3
"""Idempotently patch the active SkyPilot install to allow launching with NO
instance profile (gated by SKYPILOT_SKIP_INSTANCE_PROFILE=1).

Why: in our AWS account no available identity has iam:PassRole, so SkyPilot
cannot attach its default `skypilot-v1` instance profile and every launch fails.
Our GPU workers make no AWS calls at runtime (they serve a model from local
disk) and the SkyPilot controller uses uploaded LOCAL_CREDENTIALS, so launching
with no instance profile works fine. This edits the one line in SkyPilot that
unconditionally attaches the profile.

This patch lives OUTSIDE version control of SkyPilot, so it is wiped whenever the
venv is rebuilt (`uv sync`/reinstall). Re-run this script after any such rebuild:

    python v2/coworld/deploy/skypilot_patch/apply_patch.py
    # then always launch with the env var set:
    SKYPILOT_SKIP_INSTANCE_PROFILE=1 AWS_PROFILE=softmax uv run sky serve up ...

Run with --check to verify the patch is present without modifying anything
(exit 0 = patched, exit 1 = not patched).
"""
from __future__ import annotations

import sys
from pathlib import Path

SENTINEL = "SKYPILOT_SKIP_INSTANCE_PROFILE"

ORIGINAL = """    # The head node needs to have an IAM role that allows it to create further
    # EC2 instances.
    if 'IamInstanceProfile' not in node_cfg:
        iam = aws.resource('iam', region_name=region, **aws_credentials)
        node_cfg['IamInstanceProfile'] = _configure_iam_role(iam)
"""

PATCHED = """    # The head node needs to have an IAM role that allows it to create further
    # EC2 instances.
    #
    # cue-n-woo PATCH (see cue-n-woo repo v2/coworld/deploy/skypilot_patch/):
    # In our AWS account, no available identity has iam:PassRole, so SkyPilot
    # cannot attach its default skypilot-v1 instance profile. Our workers need no
    # AWS perms at runtime (they serve a model from local disk) and the controller
    # uses uploaded LOCAL_CREDENTIALS, so launching with NO instance profile works.
    # Setting SKYPILOT_SKIP_INSTANCE_PROFILE=1 skips the attach entirely.
    import os as _os
    if _os.environ.get('SKYPILOT_SKIP_INSTANCE_PROFILE') == '1':
        pass
    elif 'IamInstanceProfile' not in node_cfg:
        iam = aws.resource('iam', region_name=region, **aws_credentials)
        node_cfg['IamInstanceProfile'] = _configure_iam_role(iam)
"""


def _target() -> Path:
    """Locate sky/provision/aws/config.py in the active environment."""
    try:
        import sky  # noqa: PLC0415
    except ImportError:
        sys.exit("ERROR: 'sky' (SkyPilot) is not importable in this environment.")
    path = Path(sky.__file__).parent / "provision" / "aws" / "config.py"
    if not path.exists():
        sys.exit(f"ERROR: expected SkyPilot file not found: {path}")
    return path


def main() -> None:
    check_only = "--check" in sys.argv
    path = _target()
    text = path.read_text()

    if SENTINEL in text:
        print(f"OK: already patched ({path})")
        sys.exit(0)
    if check_only:
        print(f"NOT PATCHED: {path}")
        sys.exit(1)
    if ORIGINAL not in text:
        sys.exit(
            "ERROR: could not find the exact block to patch. SkyPilot may have "
            f"changed; update this script against:\n  {path}"
        )
    path.write_text(text.replace(ORIGINAL, PATCHED, 1))
    print(f"PATCHED: {path}")


if __name__ == "__main__":
    main()
