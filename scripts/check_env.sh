#!/usr/bin/env bash
# 環境の点検。必要なものがそろっているかを、一覧で出す。   scripts/check_env.sh [--smoke] [--port 8000]
#   --smoke   合成した音声で、実際に生成を 1 回 (反復 50 回) 通す (約 20 秒)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$WORK"
exec "$PYTHON" "$REPO/server/check_env.py" "$@"
