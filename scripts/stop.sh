#!/usr/bin/env bash
# scripts/start.sh で起動したサーバを止める。処理中の生成も、止める。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
PIDFILE="$WORK/serve.pid"
if [ ! -f "$PIDFILE" ]; then echo "起動していません"; exit 0; fi
PID="$(cat "$PIDFILE")"
# 別のプロセスに、同じ番号が回っていた場合に、巻き込まないよう、サーバのプロセスだけを止める
if kill -0 "$PID" 2>/dev/null && ps -p "$PID" -o args= | grep -q "server/app.py\|scripts/serve.sh"; then
  kill -- "-$(ps -o pgid= -p "$PID" | tr -d ' ')" 2>/dev/null || kill "$PID"
  echo "止めました (プロセス $PID)"
else
  echo "プロセス $PID は、すでにありません"
fi
rm -f "$PIDFILE"
