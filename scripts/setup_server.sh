#!/usr/bin/env bash
# 研究室のサーバへ、このデモを置く作業を、自動で行う (仮想環境、パッケージ、重み、設定ファイル、点検)。
#   scripts/setup_server.sh --mask-root /path/to/AntiCloning_mask --from-hf 元:/path/to/hf --from-hub 元:/path/to/torch_hub --smoke
#   scripts/setup_server.sh --dry-run ...     何を実行するかだけを表示する
# 詳しくは、server/setup_server.py の先頭と、README の「公開する」。仮想環境の前に動くので、server/env.local には、依存しない。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for p in python3.10 python3 python3.11 python3.12 python3.9; do
  if command -v "$p" >/dev/null 2>&1; then exec "$p" "$REPO/server/setup_server.py" "$@"; fi
done
echo "python3 が見つかりません (Ubuntu なら、python3.10 が標準です)" >&2
exit 1
