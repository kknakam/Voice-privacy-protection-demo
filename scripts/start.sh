#!/usr/bin/env bash
# サーバ (サイトと API) を、バックグラウンドで起動する。systemd が使えない環境向け (使える環境は deploy/voice-demo.service)。
#   scripts/start.sh [ポート番号]     既定 8000。ログは server/work/serve.log、プロセス番号は server/work/serve.pid
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
PORT="${1:-8000}"
PIDFILE="$WORK/serve.pid"
LOG="$WORK/serve.log"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "すでに起動しています (プロセス $(cat "$PIDFILE"))。止めるには scripts/stop.sh"; exit 0
fi
[ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 5242880 ] && mv -f "$LOG" "$LOG.1"      # 5 MiB を超えたら、1 世代だけ残して、新しくする
cd "$REPO"
nohup setsid "$REPO/scripts/serve.sh" "$PORT" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "起動しました: http://127.0.0.1:$PORT/ (プロセス $(cat "$PIDFILE")、ログ $LOG)"; exit 0
  fi
  sleep 0.5
done
echo "起動を確認できませんでした。ログを見てください: $LOG" >&2; exit 1
