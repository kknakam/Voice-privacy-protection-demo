#!/usr/bin/env bash
# サーバの状態 (起動しているか、順番待ちの数) を見る。   scripts/status.sh [ポート番号]
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
PORT="${1:-8000}"
PIDFILE="$WORK/serve.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "プロセス: $(cat "$PIDFILE") (scripts/start.sh で起動)"; else echo "プロセス: scripts/start.sh で起動したものは、ありません"; fi
# 自分への接続は、プロキシ (http_proxy) を通さない (--noproxy)
if out="$(curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null)"; then echo "応答: $out (ポート $PORT)"; else echo "応答: ポート $PORT から、返事がありません"; exit 1; fi
