#!/usr/bin/env bash
# 重み (x-vector、英語の音声合成、UTMOS) を、コピー元から、このサーバの置き場へコピーし、SHA-256 で確かめる。
#   scripts/copy_weights.sh --from-hf <HF_HOME のコピー元> --from-hub <TORCH_HUB_DIR のコピー元> [--to-hf DIR] [--to-hub DIR] [--dry-run]
#   scripts/copy_weights.sh --verify-only [--to-hf DIR] [--to-hub DIR]      コピーはせず、コピー先を、一覧と照合する
#
# コピー元は、ローカルのパス、または ホスト:パス (ssh で取りに行く。ユーザー@ホスト:パス も可)。
# コピー先の既定は、server/env.local の HF_HOME と TORCH_HUB_DIR。
# コピーするのは、次の 2 組だけ (ほかのものは、コピーしない。消すこともしない):
#   HF_HOME の下 (約 1.2 GB)
#     <HF_HOME>/hub/models--speechbrain--spkrec-xvect-voxceleb/      x-vector (約 32 MB)
#     <HF_HOME>/hub/models--microsoft--speecht5_tts/                 英語の音声合成 SpeechT5 (約 1.1 GB。使うのは、このうち約 590 MB)
#     <HF_HOME>/hub/models--microsoft--speecht5_hifigan/             英語のボコーダ (約 97 MB。使うのは、このうち約 50 MB)
#   TORCH_HUB_DIR の下 (約 411 MB)
#     <TORCH_HUB_DIR>/tarepan_SpeechMOS_v1.2.0/ と <TORCH_HUB_DIR>/checkpoints/utmos22_strong_step7459_v1.pt    UTMOS
# 日本語の音声合成のモデルは、リポジトリの中 (server/models_ja/) にあるので、コピーしない。
# コピー先に、正しい (SHA-256 が一致する) ものがすでにあれば、その組は飛ばす。照合に使う一覧は server/weights/*.sha256。
# 必要なのは tar と sha256sum (どこにでもある)。コピー元が別のマシンなら ssh も要る。rsync は要らない。
# 終了コード: 0 = 照合まで成功、1 = 失敗 (照合が合わない、コピーできない)、2 = 使い方の誤り
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

HF_DIRS=("hub/models--speechbrain--spkrec-xvect-voxceleb" "hub/models--microsoft--speecht5_tts" "hub/models--microsoft--speecht5_hifigan")
SM="tarepan_SpeechMOS_v1.2.0"
CK="checkpoints/utmos22_strong_step7459_v1.pt"
MAN="$REPO/server/weights"
FROM_HF=""; FROM_HUB=""; TO_HF="${HF_HOME:-}"; TO_HUB="${TORCH_HUB_DIR:-}"; DRY=0; VERIFY_ONLY=0

usage() { sed -n 2,6p "$0" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --from-hf)     FROM_HF="${2:-}"; shift 2 ;;
    --from-hub)    FROM_HUB="${2:-}"; shift 2 ;;
    --to-hf)       TO_HF="${2:-}"; shift 2 ;;
    --to-hub)      TO_HUB="${2:-}"; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)     usage ;;
    *) echo "不明な引数: $1" >&2; usage ;;
  esac
done
[ -n "$TO_HF" ] && [ -n "$TO_HUB" ] || { echo "コピー先が未設定です。server/env.local に HF_HOME と TORCH_HUB_DIR を書くか、--to-hf と --to-hub で指定してください" >&2; exit 2; }

is_remote() { [[ "$1" =~ ^[^/]+: ]]; }         # ホスト:パス の形か (最初の / より前に : がある)

# verify <置き場> <一覧>: 一覧のファイルが、すべて、置き場にあって、SHA-256 が一致すれば 0
verify() { [ -d "$1" ] && (cd "$1" && sha256sum -c --quiet "$2" >/dev/null 2>&1); }

# pack <コピー元> <パス...>: コピー元の下の、指定したパスを、tar にして標準出力へ (ローカルでも、ssh の先でも同じ)
pack() {
  local src="$1"; shift
  if is_remote "$src"; then
    local host="${src%%:*}" path="${src#*:}" rest
    printf -v rest '%q ' "$@"
    ssh "$host" "tar -C $(printf '%q' "$path") --exclude=__pycache__ -cf - $rest"
  else
    tar -C "$src" --exclude=__pycache__ -cf - "$@"
  fi
}

# copy_group <名前> <コピー元> <コピー先> <一覧> <パス...>
copy_group() {
  local name="$1" src="$2" dst="$3" man="$4"; shift 4
  echo "== $name: $src -> $dst"
  if verify "$dst" "$man"; then echo "   コピー済みです (SHA-256 が一致)。飛ばします"; return 0; fi
  if ! is_remote "$src" && [ "$(realpath -m "$src")" = "$(realpath -m "$dst")" ]; then
    echo "   NG: コピー元とコピー先が同じ場所なのに、SHA-256 が一致しません" >&2; return 1
  fi
  if [ "$DRY" = 1 ]; then echo "   (dry-run) コピーするもの: $*"; return 0; fi
  mkdir -p "$dst"
  pack "$src" "$@" | tar -C "$dst" -xf - --no-same-owner
  if verify "$dst" "$man"; then
    echo "   コピーしました。SHA-256 が一致しました"
  else
    echo "   NG: コピーしたあとの SHA-256 が一致しません (一致しないもの):" >&2
    (cd "$dst" && sha256sum -c "$man" 2>&1 | grep -v ': OK$' | head -5) >&2 || true
    return 1
  fi
}

check_group() {      # check_group <名前> <置き場> <一覧>
  if verify "$2" "$3"; then echo "OK  $1: $2 (SHA-256 が一致)"; else echo "NG  $1: $2 (ないか、SHA-256 が一致しません)"; return 1; fi
}

fail=0
if [ "$VERIFY_ONLY" = 1 ]; then
  check_group "HF_HOME (x-vector、英語の SpeechT5)" "$TO_HF" "$MAN/hf_home.sha256" || fail=1
  check_group "TORCH_HUB_DIR (UTMOS)" "$TO_HUB" "$MAN/torch_hub.sha256" || fail=1
  exit "$fail"
fi

[ -n "$FROM_HF" ] && [ -n "$FROM_HUB" ] || { echo "コピー元を、--from-hf と --from-hub で指定してください" >&2; usage; }
copy_group "HF_HOME (x-vector、英語の SpeechT5)" "$FROM_HF" "$TO_HF" "$MAN/hf_home.sha256" "${HF_DIRS[@]}" || fail=1
copy_group "TORCH_HUB_DIR (UTMOS)" "$FROM_HUB" "$TO_HUB" "$MAN/torch_hub.sha256" "$SM" "$CK" || fail=1
if [ "$fail" = 0 ] && [ "$DRY" = 0 ]; then
  echo
  echo "次: server/env.local に ANTICLONING_MASK_ROOT と PYTHON を書き、scripts/check_env.sh --smoke で、全体を点検してください"
fi
exit "$fail"
