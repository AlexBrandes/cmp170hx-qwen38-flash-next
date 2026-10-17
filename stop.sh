#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && . ./.env
NAME=${NAME:-cmp170hx-qwen38-flash-next}
docker stop -t 45 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME"
