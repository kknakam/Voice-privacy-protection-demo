"""敵対的音声の生成 (サーバ側の処理の本体)。

最適化の中心 (マスク、損失、Adam の更新) は、研究のリポジトリ (anticloning_mask) のコードを import して使う。
このリポジトリには、研究のコードを含めない。場所は、環境変数 ANTICLONING_MASK_ROOT で指定する (server/env.example を参照)。

入口は server/attack_cli.py (scripts/run_attack.sh から呼ぶ)。
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SAMPLE_RATE = 16000
MAX_SEC = 8.0
N_FFT = 1024
HOP = 256
STEPS_ALLOWED = (50, 100, 200)
POOL_FILE = Path(__file__).resolve().parent / "data" / "pool.npz"

# 攻撃の設定。以前、ブラウザで動かしていたときの値と同じ (Beta の β だけ、Beta 探索の採用値 1.0)
ATTACK = dict(lr=0.05, grad_clip=1.0, init_std=1.0, beta_alpha=5.0, beta_beta=1.0, w_beta=10.0,
              w_dist=5.0, w_percept=5.0, w_away=30.0, w_toward=30.0, tau=0.9)


def use_research_code() -> None:
    """研究のリポジトリを import できるようにし、重みの置き場 (torch.hub) を設定する"""
    root = os.environ.get("ANTICLONING_MASK_ROOT")
    if not root or not (Path(root) / "anticloning_mask").is_dir():
        raise RuntimeError("環境変数 ANTICLONING_MASK_ROOT に、研究のリポジトリ (anticloning_mask を含む場所) を指定してください")
    if root not in sys.path:
        sys.path.insert(0, root)
    hub = os.environ.get("TORCH_HUB_DIR")
    if hub:
        import torch
        torch.hub.set_dir(hub)


def read_wav(path: str) -> np.ndarray:
    """16 kHz・モノラル・16 bit の WAV だけを受け付ける (ブラウザが、この形で送る)。最長 MAX_SEC 秒に切る"""
    import soundfile as sf
    with sf.SoundFile(path) as f:
        if f.samplerate != SAMPLE_RATE or f.channels != 1 or f.subtype != "PCM_16":
            raise ValueError("16 kHz・モノラル・16 bit の WAV だけを受け付けます")
        if f.frames < SAMPLE_RATE // 2:
            raise ValueError("音声が短すぎます (0.5 秒以上が必要です)")
        wave = f.read(int(SAMPLE_RATE * MAX_SEC), dtype="float32")
    return wave


def load_pool() -> tuple:
    data = np.load(POOL_FILE, allow_pickle=False)
    return [str(i) for i in data["ids"]], data["emb"].astype(np.float32)


def farthest_target(e_src: np.ndarray, ids: list, emb: np.ndarray) -> tuple:
    """録音の話者から、コサイン類似度がいちばん低い話者 (= いちばん遠い話者) を、候補群から選ぶ"""
    e = e_src / np.linalg.norm(e_src)
    p = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = p @ e
    k = int(np.argmin(sims))
    return ids[k], emb[k], float(sims[k])


def _quantize(a: np.ndarray, lo: float | None = None, hi: float | None = None) -> dict:
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi
    q = np.clip(np.round((a - lo) / max(hi - lo, 1e-9) * 255), 0, 255).astype(np.uint8)
    return {"lo": lo, "hi": hi, "q": base64.b64encode(q.tobytes()).decode("ascii")}


def run(wave: np.ndarray, steps: int, emit, device: str = "cuda") -> dict:
    """録音 wave (float32、16 kHz) から、敵対的音声を作る。emit(dict) で、進み具合を知らせる。結果は JSON にできる辞書"""
    import time
    import torch
    import torch.nn.functional as F

    use_research_code()
    from anticloning_mask.audio_utils import snr_db
    from anticloning_mask.metrics import compute_utmos
    from anticloning_mask.optimize import run_mask_attack
    from anticloning_mask.speaker.encoders import build_encoder

    started = time.time()
    emit({"kind": "loading"})
    enc = build_encoder("xvector", device)
    x = torch.from_numpy(wave).to(device)

    with torch.no_grad():
        e_src = enc.embed(x)                                           # [1, 512]
    ids, pool = load_pool()
    target_id, target_emb, target_cos = farthest_target(e_src[0].cpu().numpy(), ids, pool)
    e_tgt = torch.from_numpy(target_emb).to(device).unsqueeze(0)

    a = ATTACK
    cfg = SimpleNamespace(
        device=device,
        optim=SimpleNamespace(optimizer="adam", lr=a["lr"], grad_clip_norm=a["grad_clip"], mask_init_std=a["init_std"], num_iters=steps),
        stft=SimpleNamespace(n_fft=N_FFT, hop_length=HOP),
        mask=SimpleNamespace(use_prior=True, beta_alpha=a["beta_alpha"], beta_beta=a["beta_beta"], beta_weight=a["w_beta"]),
        quality=SimpleNamespace(dist_weight=a["w_dist"], percept_weight=a["w_percept"]),
    )
    counter = {"n": 0}

    def task_loss(x_adv, mask):
        # L_speaker = w_away * max(cos(e_src, e') - tau, 0) + w_toward * (-cos(e', e_tgt))  (研究の speaker/attack.py と同じ式)
        emb = enc.embed(x_adv)
        cos_away = F.cosine_similarity(e_src, emb, dim=-1).mean()
        loss_away = torch.clamp(cos_away - a["tau"], min=0.0)
        loss_toward = -F.cosine_similarity(emb, e_tgt, dim=-1).mean()
        counter["n"] += 1
        emit({"kind": "step", "step": counter["n"], "steps": steps})
        return a["w_away"] * loss_away + a["w_toward"] * loss_toward, {"loss_away": float(loss_away.item()), "loss_toward": float(loss_toward.item())}

    res = run_mask_attack(x, cfg, task_loss, num_iters=steps)
    adv = res.protected_audio.detach()[: x.shape[-1]]
    if adv.shape[-1] < x.shape[-1]:
        adv = F.pad(adv, (0, x.shape[-1] - adv.shape[-1]))
    adv = adv.clamp(-1.0, 1.0)

    emit({"kind": "measuring"})
    window = torch.hann_window(N_FFT, device=device)
    with torch.no_grad():
        e_adv = enc.embed(adv)
        spec = torch.stft(x.unsqueeze(0), n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)[0]     # [F, T]
        spec_adv = torch.stft(adv.unsqueeze(0), n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)[0]
        mask = res.final_mask[:, : spec.shape[1]]
        clean_db = (10 * torch.log10(spec.abs() ** 2 + 1e-10)).T.contiguous().cpu().numpy()          # [T, F]
        adv_db = (10 * torch.log10(spec_adv.abs() ** 2 + 1e-10)).T.contiguous().cpu().numpy()
        mask_np = mask.T.contiguous().cpu().numpy()
    snr = snr_db(x, adv)
    utmos_clean = compute_utmos(x.cpu().numpy(), SAMPLE_RATE, device)
    utmos_adv = compute_utmos(adv.cpu().numpy(), SAMPLE_RATE, device)
    cos = lambda p, q: float(F.cosine_similarity(p, q, dim=-1).mean().item())

    T, Fb = clean_db.shape
    lo = float(min(clean_db.min(), adv_db.min()))
    hi = float(max(clean_db.max(), adv_db.max()))
    pcm = (adv.cpu().numpy() * 32767.0).round().astype(np.int16)
    return {
        "target": {"id": target_id, "cosToRecording": target_cos},
        "steps": steps,
        "seconds": round(time.time() - started, 1),
        "stats": {"snr": float(snr), "utmosClean": float(utmos_clean), "utmosAdv": float(utmos_adv), "maskMean": float(mask_np.mean())},
        "cos": {"srcToAdv": cos(e_src, e_adv), "srcToTarget": cos(e_src, e_tgt), "advToTarget": cos(e_adv, e_tgt)},
        "emb": {"src": e_src[0].cpu().tolist(), "adv": e_adv[0].cpu().tolist(), "tgt": target_emb.tolist()},
        "adv": {"sampleRate": SAMPLE_RATE, "pcm16": base64.b64encode(pcm.tobytes()).decode("ascii")},
        "heat": {"F": Fb, "T": T, "clean": _quantize(clean_db, lo, hi), "adv": _quantize(adv_db, lo, hi), "mask": _quantize(mask_np, 0.0, 1.0)},
    }
