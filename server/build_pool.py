"""ターゲット話者の候補群を作る (開発用。scripts/build_pool.sh から呼ぶ)。

LibriTTS の話者ごとに、x-vector の平均を求め、互いにできるだけ離れた N 人を選ぶ (最遠点サンプリング)。
録音の話者から「いちばん遠い話者」を選ぶとき、候補が、x-vector の空間に広がっているほど、遠い話者が見つかる。

出力:
  server/data/pool.npz              話者の ID と、平均の x-vector (N x 512)
  site/assets/targets/<話者>.wav    その話者の参照音声 (16 kHz、モノラル、約 6 秒、音量を調整)

入力: LibriTTS (CC BY 4.0) の parquet (Hugging Face の mythicinfinity/libritts)。環境変数 LIBRITTS_PARQUET_DIR に、
*.parquet を含むフォルダを指定する。parquet を読むための pyarrow が、実行する Python に無ければ、
環境変数 PYARROW_PATH に、pyarrow を含むフォルダを指定する。
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import core  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REF_SEC = 6.0
TARGET_RMS_DB = -23.0


def to16k(wave: np.ndarray, sr: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != core.SAMPLE_RATE:
        g = gcd(core.SAMPLE_RATE, sr)
        wave = resample_poly(wave, core.SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wave.astype(np.float32)


def collect(parquet_dir: Path, per_speaker: int) -> dict:
    """話者ごとに、3 秒以上の発話を per_speaker 件まで集める ({話者: [16 kHz の波形, ...]})"""
    if os.environ.get("PYARROW_PATH"):
        sys.path.insert(0, os.environ["PYARROW_PATH"])
    import pyarrow.parquet as pq
    import soundfile as sf
    got: dict = {}
    for f in sorted(parquet_dir.rglob("*.parquet")):
        pf = pq.ParquetFile(str(f))
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=["audio", "speaker_id"]).to_pylist():
                spk = str(row["speaker_id"])
                if len(got.get(spk, [])) >= per_speaker:
                    continue
                wave, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                wave = to16k(wave, sr)
                if len(wave) >= 3 * core.SAMPLE_RATE:
                    got.setdefault(spk, []).append(wave)
    return got


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24, help="候補の人数")
    p.add_argument("--utts", type=int, default=3, help="話者ごとに、平均する発話の数")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    root = Path(os.environ.get("LIBRITTS_PARQUET_DIR", ""))
    if not root.is_dir():
        raise SystemExit("環境変数 LIBRITTS_PARQUET_DIR に、LibriTTS の parquet があるフォルダを指定してください")

    import soundfile as sf
    import torch
    core.use_research_code()
    from anticloning_mask.speaker.encoders import build_encoder
    enc = build_encoder("xvector", args.device)

    got = collect(root, args.utts + 2)
    ids, embs = [], []
    for spk, waves in sorted(got.items()):
        if len(waves) < args.utts:
            continue
        es = []
        for w in waves[: args.utts]:
            with torch.no_grad():
                e = enc.embed(torch.from_numpy(w[: int(core.SAMPLE_RATE * core.MAX_SEC)]).to(args.device))[0].cpu().numpy()
            es.append(e / np.linalg.norm(e))
        m = np.mean(es, axis=0)
        ids.append(spk)
        embs.append(m / np.linalg.norm(m))
    E = np.stack(embs)
    print(f"候補にできる話者: {len(ids)} 人", flush=True)

    # 最遠点サンプリング: 全体の平均から最も遠い話者から始め、選んだ話者たちから最も遠い話者を、順に足す
    first = int(np.argmin(E @ (E.mean(axis=0) / np.linalg.norm(E.mean(axis=0)))))
    chosen = [first]
    best = E @ E[first]                       # 選んだ話者たちとの、最大のコサイン類似度
    while len(chosen) < args.n:
        k = int(np.argmin(best))
        chosen.append(k)
        best = np.maximum(best, E @ E[k])
    chosen.sort(key=lambda i: ids[i])
    pool_ids = [ids[i] for i in chosen]
    pool_emb = E[chosen].astype(np.float32)
    core.POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(core.POOL_FILE, ids=np.array(pool_ids), emb=pool_emb)
    sims = pool_emb @ pool_emb.T
    off = sims[~np.eye(len(chosen), dtype=bool)]
    print(f"選んだ {len(chosen)} 人の、話者どうしの類似度: 平均 {off.mean():.3f}、最大 {off.max():.3f}、最小 {off.min():.3f}", flush=True)

    out = REPO / "site" / "assets" / "targets"
    out.mkdir(parents=True, exist_ok=True)
    for i in chosen:
        spk = ids[i]
        ref = next((w for w in got[spk] if len(w) >= REF_SEC * core.SAMPLE_RATE), got[spk][0])
        ref = ref[: int(core.SAMPLE_RATE * REF_SEC)]
        rms = float(np.sqrt(np.mean(ref ** 2)) + 1e-9)
        ref = ref * (10 ** (TARGET_RMS_DB / 20) / rms)
        peak = float(np.abs(ref).max())
        if peak > 0.95:
            ref = ref * (0.95 / peak)
        sf.write(str(out / f"{spk}.wav"), ref, core.SAMPLE_RATE, subtype="PCM_16")
    print(f"参照音声を {out} に書きました", flush=True)


if __name__ == "__main__":
    main()
