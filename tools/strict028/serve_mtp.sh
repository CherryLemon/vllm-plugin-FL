#!/usr/bin/env bash
# DSpark MTP uses the original checkpoint's three draft layers and five tokens.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/serve.sh" \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' "$@"
