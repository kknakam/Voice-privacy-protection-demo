#!/usr/bin/env bash
# 録音から敵対的音声を作る (最適化の本体は server/attack_cli.py。Python)。
#   scripts/run_attack.sh --input 録音.wav --out-dir 出力先 [--steps 50|100|200] [--device cuda]
# Web サーバ (server/app.py) も、利用者の録音ごとに、このスクリプトを呼ぶ。引数は、サーバが決めた値だけを渡す。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$WORK"
exec "$PYTHON" "$REPO/server/attack_cli.py" "$@"
