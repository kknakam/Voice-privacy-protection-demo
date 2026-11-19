"""サーバを動かす環境の点検 (scripts/check_env.sh から呼ぶ)。必要なものを一覧にして、足りないものを知らせる。

  scripts/check_env.sh [--smoke] [--port 8000]
    --smoke  合成した音声で、敵対的音声の生成を 1 回 (反復 50 回)、続けて、クローンの合成 (日本語と英語) を 1 回ずつ、実際に通す
終了コードは、足りないもの (NG) があれば 1。
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

rows: list = []


def add(name: str, status: str, detail: str = "") -> None:
    rows.append((name, status, detail))


def check_python() -> None:
    from server import tts
    tts.add_extra_paths()
    for mod in ("numpy", "scipy", "soundfile", "torch", "torchaudio", "speechbrain", "transformers", "onnxruntime", "pyopenjtalk"):
        try:
            m = importlib.import_module(mod)
            add(f"Python: {mod}", "OK", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            add(f"Python: {mod}", "NG", f"{type(e).__name__}: {e}"[:80])
    try:
        import torch
        if torch.cuda.is_available():
            add("GPU (CUDA)", "OK", torch.cuda.get_device_name(0))
        else:
            add("GPU (CUDA)", "注意", "GPU が見えません。生成は、CPU では、非常に遅くなります (scripts/run_attack.sh --device cpu)")
    except Exception:  # noqa: BLE001
        pass


def check_research() -> None:
    root = os.environ.get("ANTICLONING_MASK_ROOT", "")
    ok = bool(root) and (Path(root) / "anticloning_mask" / "optimize.py").is_file()
    add("研究のリポジトリ", "OK" if ok else "NG", root or "ANTICLONING_MASK_ROOT が未設定 (server/env.local)")
    if ok:
        from server import core
        try:
            core.use_research_code()
            for mod in ("anticloning_mask.optimize", "anticloning_mask.speaker.encoders", "anticloning_mask.metrics", "anticloning_mask.audio_utils"):
                importlib.import_module(mod)
            add("研究のコードの import", "OK", "optimize / speaker.encoders / metrics / audio_utils")
        except Exception as e:  # noqa: BLE001
            add("研究のコードの import", "NG", f"{type(e).__name__}: {e}"[:100])


def check_weights() -> None:
    hf = os.environ.get("HF_HOME", "")
    xv = glob.glob(os.path.join(hf, "hub", "models--speechbrain--spkrec-xvect-voxceleb", "snapshots", "*", "embedding_model.ckpt")) if hf else []
    add("重み: x-vector", "OK" if xv else "NG", xv[0] if xv else "HF_HOME の下の hub/models--speechbrain--spkrec-xvect-voxceleb/ に置く")
    hub = os.environ.get("TORCH_HUB_DIR", "")
    code = Path(hub) / "tarepan_SpeechMOS_v1.2.0" / "hubconf.py"
    ckpt = Path(hub) / "checkpoints" / "utmos22_strong_step7459_v1.pt"
    add("重み: UTMOS (コード)", "OK" if code.is_file() else "NG", str(code) if code.is_file() else "TORCH_HUB_DIR の下の tarepan_SpeechMOS_v1.2.0/ に置く")
    add("重み: UTMOS (checkpoint)", "OK" if ckpt.is_file() else "NG", str(ckpt) if ckpt.is_file() else "TORCH_HUB_DIR の下の checkpoints/ に置く")
    for label, repo, need in (("英語の音声合成 (SpeechT5)", "models--microsoft--speecht5_tts", "pytorch_model.bin"), ("英語のボコーダ (HiFi-GAN)", "models--microsoft--speecht5_hifigan", "pytorch_model.bin")):
        hit = glob.glob(os.path.join(hf, "hub", repo, "snapshots", "*", need)) + glob.glob(os.path.join(hf, "hub", repo, "snapshots", "*", "model.safetensors")) if hf else []
        add(f"重み: {label}", "OK" if hit else "NG", hit[0] if hit else f"HF_HOME の下の hub/{repo}/ に置く (scripts/copy_weights.sh)")
    add("オフライン実行", "OK" if os.environ.get("HF_HUB_OFFLINE") == "1" else "注意", "HF_HUB_OFFLINE=1 (実行時にネットへ取りに行かない)")


def check_site() -> None:
    from server import core
    import numpy as np
    pool = REPO / "server" / "data" / "pool.npz"
    if pool.is_file():
        ids, emb = core.load_pool()
        wavs = {p.stem for p in (REPO / "site" / "assets" / "targets").glob("*.wav")}
        ok = set(ids) == wavs and emb.shape == (len(ids), 512)
        add("ターゲット話者の候補群", "OK" if ok else "NG", f"{len(ids)} 人、参照音声と{'一致' if ok else '食い違い'}")
    else:
        add("ターゲット話者の候補群", "NG", "server/data/pool.npz がありません (scripts/build_pool.sh)")
    ja_dir = Path(os.environ.get("TTS_JA_DIR", REPO / "server" / "models_ja"))
    manifest = ja_dir / "manifest.json"
    if manifest.is_file():
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", [])
        missing = [p for f in files for p in f["parts"] if not (ja_dir / p).is_file()]
        add("日本語の音声合成のモデル", "NG" if missing else "OK", f"{len(files)} ファイル、分割ファイルの欠け {len(missing)} 件 ({ja_dir})")
    else:
        add("日本語の音声合成のモデル", "注意", f"{manifest} がありません (日本語の読み上げは、使えません)")
    for name in ("index.html", "_headers"):
        add(f"サイト: {name}", "OK" if (REPO / "site" / name).is_file() else "NG")


def check_machine(port: int) -> None:
    work = REPO / "server" / "work"
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free / 2**30
    add("空き容量 (作業の場所)", "OK" if free >= 2 else ("注意" if free >= 0.5 else "NG"), f"{free:.1f} GiB (処理中の録音は、1 MiB 未満。処理後に消す)")
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", port)) == 0
    add(f"ポート {port}", "注意" if busy else "OK", "すでに使われています (起動中のサーバがあるか、別のプログラムです)" if busy else "空いています")
    cf = os.environ.get("CLOUDFLARED") or (str(REPO / "server" / "work" / "bin" / "cloudflared") if (REPO / "server" / "work" / "bin" / "cloudflared").is_file() else shutil.which("cloudflared"))
    add("cloudflared (HTTPS で公開するため)", "OK" if cf else "注意", cf or "見つかりません (server/work/bin/cloudflared に置くか、PATH に通す)")
    secure = "HTTPS でしか使えません (マイクは、HTTPS か localhost のみ)。scripts/tunnel.sh か、前段の HTTPS を使う"
    add("公開 (HTTPS)", "注意", secure)


def smoke() -> None:
    import numpy as np
    import soundfile as sf
    rng = np.random.default_rng(0)
    t = np.arange(3 * 16000) / 16000
    wave = (0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.normal(size=t.size)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.wav"
        sf.write(str(src), wave, 16000, subtype="PCM_16")
        p = subprocess.run(["bash", str(REPO / "scripts" / "run_attack.sh"), "--input", str(src), "--out-dir", str(Path(d) / "out"), "--steps", "50"], capture_output=True, text=True, timeout=300)
        out = Path(d) / "out" / "result.json"
        if p.returncode == 0 and out.is_file():
            r = json.loads(out.read_text(encoding="utf-8"))
            add("通しの生成 (50 反復)", "OK", f"{r['seconds']} 秒、ターゲット {r['target']['id']}、SNR {r['stats']['snr']:.1f} dB、UTMOS {r['stats']['utmosAdv']:.2f}")
        else:
            tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-1:] or [""]
            add("通しの生成 (50 反復)", "NG", tail[0][:100])


def smoke_tts() -> None:
    """音声合成を、実際に通す: モデルを読み込み、日本語と英語を 1 文ずつ合成する (試験データの x-vector を使う)"""
    import time
    from server import tts
    emb = json.loads((REPO / "server" / "testdata" / "emb_fixture.json").read_text(encoding="utf-8"))
    t0 = time.time()
    svc = tts.Service(log=lambda m: None)
    svc.start()
    svc.ready.wait()
    add("音声合成の読み込み", "OK" if (svc.en and svc.ja) else "NG", f"{time.time() - t0:.0f} 秒、英語 {'可' if svc.en else '不可'}、日本語 {'可' if svc.ja else '不可'}" + (f" ({svc.errors})" if svc.errors else ""))
    for lang, text in (("ja", "今日は、いい天気です。"), ("en", "My voice belongs to me.")):
        if not getattr(svc, lang):
            continue
        try:
            t = time.time()
            r = svc.synthesize(text, emb, lambda *a: None, lambda: False)
            add(f"通しの合成 ({'日本語' if lang == 'ja' else '英語'})", "OK", f"{time.time() - t:.1f} 秒、保護なし 本人 {r['clean']['cosSrc']:.3f}、保護あり ターゲット {r['protected']['cosTgt']:.3f}")
        except Exception as e:  # noqa: BLE001
            add(f"通しの合成 ({'日本語' if lang == 'ja' else '英語'})", "NG", f"{type(e).__name__}: {e}"[:100])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    check_python()
    check_research()
    check_weights()
    check_site()
    check_machine(args.port)
    if args.smoke:
        smoke()
        smoke_tts()
    width = max(len(n) for n, _, _ in rows)
    for name, status, detail in rows:
        print(f"[{status}] {name.ljust(width)}  {detail}")
    ng = sum(1 for _, s, _ in rows if s == "NG")
    note = sum(1 for _, s, _ in rows if s == "注意")
    print(f"\n{len(rows)} 項目: NG {ng}、注意 {note}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
