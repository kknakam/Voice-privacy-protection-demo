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
# 別のプロセスが、すでにそのポートを使っていると、その応答を、このサーバの起動と取り違えるので、先に確かめる
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "ポート $PORT は、すでに別のプロセスが使っています。別のポートを指定してください (例: scripts/start.sh 8765)。使っているプロセスは: ss -ltnp | grep ':$PORT '" >&2; exit 1
fi
OFFSET="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"     # 失敗したときに、今回の起動で追加された分だけを見せるため
nohup setsid "$REPO/scripts/serve.sh" "$PORT" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
WAIT="${START_WAIT:-120}"      # 待つ秒数。初回は、PyTorch などの読み込みが遅い機械 (ネットワークのファイルシステムなど) がある
for _ in $(seq 1 $((WAIT * 2))); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PIDFILE"
    echo "起動に失敗しました (プロセスが終了しました)。今回のログの末尾 ($LOG):" >&2
    tail -c +$((OFFSET + 1)) "$LOG" | tr '\r' '\n' | grep -v 'Loading weights' | tail -n 15 | cut -c1-300 >&2; exit 1
  fi
  if curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then      # 自分への接続は、プロキシ (http_proxy) を通さない
    echo "起動しました: http://127.0.0.1:$PORT/ (プロセス $PID、ログ $LOG)"
    echo "音声合成のモデルの読み込みは、このあとも続きます (約 20 秒)。scripts/status.sh の応答が \"state\": \"ready\" になれば、Step 3 も使えます"; exit 0
  fi
  sleep 0.5
done
echo "$WAIT 秒待っても、応答がありません。プロセス $PID は動いています。scripts/status.sh $PORT で、状態を確かめてください (ログ: $LOG)" >&2; exit 1
