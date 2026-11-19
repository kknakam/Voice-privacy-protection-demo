#!/usr/bin/env bash
# 共通の環境設定 (他のスクリプトから source する)。server/env.local があれば読む。書き方は server/env.example を参照
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO/server/env.local" ]; then
  # shellcheck disable=SC1091
  source "$REPO/server/env.local"
fi
PYTHON="${PYTHON:-python3}"
export ANTICLONING_MASK_ROOT="${ANTICLONING_MASK_ROOT:-}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-}"
[ -n "${HF_HOME:-}" ] && export HF_HOME
export LIBRITTS_PARQUET_DIR="${LIBRITTS_PARQUET_DIR:-}" PYARROW_PATH="${PYARROW_PATH:-}"     # 候補群を作るときだけ使う
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # 重みは、あらかじめ置いておく (実行時にネットへ取りに行かない)
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export PYTHONUNBUFFERED=1
# speechbrain が、作業用のフォルダ (pretrained_models/) を、実行した場所に作る。リポジトリを汚さないよう、専用の場所で動かす
WORK="$REPO/server/work"
mkdir -p "$WORK"
