#!/usr/bin/env bash
# Dynamically change the fleet's min/max replicas — live, no downtime.
#
# min_replicas and max_replicas hot-apply via `sky serve update` (unlike
# num_overprovision, which needs a fresh `sky serve up`). This script edits the
# event spec's min/max and rolls the update.
#
# Usage:
#   ./scale.sh <MIN> <MAX> [SERVICE] [SPEC]
# Examples:
#   ./scale.sh 4 128            # floor 4, ceiling 128
#   ./scale.sh 1 16             # back to a modest ceiling
#
# Requires (same env as all fleet ops):
#   export AWS_PROFILE=softmax SKYPILOT_SKIP_INSTANCE_PROFILE=1 AWS_REGION=us-east-1
set -euo pipefail

MIN="${1:?usage: scale.sh MIN MAX [SERVICE] [SPEC]}"
MAX="${2:?usage: scale.sh MIN MAX [SERVICE] [SPEC]}"
SERVICE="${3:-cue-n-woo-workers}"
SPEC="${4:-$(dirname "$0")/worker_service_fast.sky.yaml}"
SKY="${SKY:-/home/kyleherndon/metta/.venv/bin/sky}"

[ -f "$SPEC" ] || { echo "spec not found: $SPEC" >&2; exit 1; }

# Write a temp spec with the new min/max so we never mutate the source-of-truth
# spec unexpectedly (and num_overprovision stays whatever the spec has).
TMP="$(mktemp /tmp/cue-scale-XXXX.yaml)"
trap 'rm -f "$TMP"' EXIT
sed -E "s/^( *min_replicas:).*/\1 ${MIN}/; s/^( *max_replicas:).*/\1 ${MAX}/" "$SPEC" > "$TMP"

echo "Updating ${SERVICE}: min_replicas=${MIN}, max_replicas=${MAX}"
grep -E "min_replicas|max_replicas|num_overprovision" "$TMP" | sed 's/^/  /'
echo
"$SKY" serve update "$SERVICE" "$TMP" --yes
echo
echo "Done. Check: $SKY serve status $SERVICE"
echo "NOTE: this changes MIN/MAX only. To change num_overprovision you must tear"
echo "down and 'sky serve up' fresh (it is not re-read on update)."
