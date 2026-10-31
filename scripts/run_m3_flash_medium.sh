#!/usr/bin/env bash
set -euo pipefail
export EXPERIMENT_ID=m3_flash_medium_20261031
export EXPERIMENT_ACCOUNT=m3-flash-medium-20261031
export EPISODIC_WORKERS=4
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_m3_flash.sh" "$@"
