"""録音から、敵対的音声を作る (コマンドライン)。scripts/run_attack.sh から呼ぶ。

  scripts/run_attack.sh --input 録音.wav --out-dir 出力先 --steps 100

出力先に result.json (結果。音声、統計、埋め込み、スペクトログラム) を書く。
進み具合は、標準出力に 1 行 1 件の JSON で出す。失敗は、{"kind": "error", "message": ...} を出して、終了コード 1 で終わる。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import core  # noqa: E402


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=100, choices=core.STEPS_ALLOWED)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    try:
        wave = core.read_wav(args.input)
        result = core.run(wave, args.steps, emit, args.device)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        emit({"kind": "done"})
        return 0
    except Exception as e:  # noqa: BLE001  (利用者に返す文を、1 行にまとめる)
        emit({"kind": "error", "message": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
