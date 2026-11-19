#!/usr/bin/env bash
# ターゲット話者の候補群 (server/data/pool.npz) と、参照音声 (site/assets/targets/*.wav) を作る。
#   scripts/build_pool.sh [--n 24]    (LIBRITTS_PARQUET_DIR などは、server/env.local に書く)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$WORK"
exec "$PYTHON" "$REPO/server/build_pool.py" "$@"
