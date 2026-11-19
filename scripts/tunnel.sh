#!/usr/bin/env bash
# Cloudflare Tunnel で、サイトを HTTPS で公開する (マイクは、HTTPS でしか使えない)。cloudflared が要る。
#   scripts/tunnel.sh quick [ポート]    アカウント不要の、一時的な URL (https://....trycloudflare.com)。試験用。
#                                       URL を知っている人は、誰でも、この処理 (GPU) を使える。確認が終わったら、Ctrl-C で閉じる
#   scripts/tunnel.sh run <トンネル名>   名前つきのトンネル (Cloudflare のアカウントとドメインが要る。設定は deploy/cloudflared-config.example.yml)
# cloudflared は、環境変数 CLOUDFLARED、server/work/bin/cloudflared、PATH の順に探す。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
BIN="${CLOUDFLARED:-}"
[ -z "$BIN" ] && [ -x "$WORK/bin/cloudflared" ] && BIN="$WORK/bin/cloudflared"
[ -z "$BIN" ] && BIN="$(command -v cloudflared || true)"
[ -z "$BIN" ] && { echo "cloudflared が見つかりません (https://github.com/cloudflare/cloudflared/releases)" >&2; exit 1; }
case "${1:-}" in
  quick) exec "$BIN" tunnel --no-autoupdate --url "http://127.0.0.1:${2:-8000}" ;;
  run)   [ -n "${2:-}" ] || { echo "トンネル名を指定してください" >&2; exit 1; }; exec "$BIN" tunnel --no-autoupdate run "$2" ;;
  *)     sed -n 2,8p "$0"; exit 1 ;;
esac
