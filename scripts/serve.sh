#!/usr/bin/env bash
# サイト (site/) と、敵対的音声を作る API を、1 つのサーバで配信する。
#   scripts/serve.sh [ポート番号]    既定 8000。HTTPS は、前段 (Cloudflare Tunnel など) に任せる
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO"
exec "$PYTHON" "$REPO/server/app.py" "$@"
