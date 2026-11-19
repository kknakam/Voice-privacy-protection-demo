"""音声合成 (Step 3: クローンの合成) をサーバで行う。server/app.py から使う。

入力した文章から、言語を決め (ひらがな・カタカナ・漢字・半角カナが 1 字でもあれば日本語、なければ英語)、
本人の録音の x-vector と、敵対的音声の x-vector で、2 本の音声を合成して、本人とターゲットへの類似度を測る。

  英語    microsoft/speecht5_tts と speecht5_hifigan (PyTorch)。GPU があれば GPU で動かす (CPU では遅い)。
          文章の正規化 (数字を英単語に展開するなど) は、これまでブラウザで行っていた規則と同じ。
  日本語  SpeechT5 を日本語に追加学習したモデル (ONNX、server/models_ja/。ONNX Runtime、CPU)。
          音素化は、pyopenjtalk-plus の素の版 (use_vanilla=True) で、音素と韻律記号にする。
          文の終わりの検出 (止め方) と、長い文の分割 (入力長 100 以上を、ポーズの位置で 60 記号以内の区間に分ける) を行う。

モデルは、サーバの起動時に、別のスレッドで 1 回だけ読み込む (Service.start)。言語ごとに独立で、読み込めなかった言語は使えない。
このモジュールは、重いライブラリ (torch、onnxruntime、transformers、pyopenjtalk) を、読み込みのスレッドの中で import する。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSTANTS = json.loads((REPO / "server" / "data" / "tts.json").read_text(encoding="utf-8"))
SAMPLE_RATE = 16000
EN_MAX_CHARS = CONSTANTS["en"]["maxChars"]
JA_MAX_CHARS = CONSTANTS["ja"]["maxChars"]
JAPANESE = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")

# 日本語の生成の設定 (評価した値)
END_TAIL, END_MARGIN = 2, 8           # 文の終わりの検出: 注意が入力の末尾 END_TAIL 記号以内に届いたら、END_MARGIN ステップ後に止める
SPLIT_FROM, SPLIT_MAX, GAP_SEC = 100, 60, 0.1     # 長い文の分割: 入力長が SPLIT_FROM 以上なら、SPLIT_MAX 記号以内の区間に分け、GAP_SEC 秒の無音を挟んでつなぐ
LAYERS, HEADS, HEAD_DIM = 6, 12, 64
DEADLINE_SEC = 180                    # 1 回の合成 (2 本分) の上限。超えたら中止する


class Cancelled(Exception):
    pass


def js_len(text: str) -> int:
    """JavaScript の String.length と同じ数え方 (UTF-16 の単位)。画面の文字数の表示と合わせる"""
    return len(text.encode("utf-16-le")) // 2


def detect_lang(text: str) -> str:
    return "ja" if JAPANESE.search(text) else "en"


def add_extra_paths() -> None:
    """環境変数 EXTRA_PYTHONPATH (: 区切り) の場所を、import の探索の末尾に足す。venv に入れられないパッケージ (pyopenjtalk-plus など) の置き場。
    末尾に足すので、venv のパッケージを隠さない。"""
    for p in filter(None, os.environ.get("EXTRA_PYTHONPATH", "").split(os.pathsep)):
        if p not in sys.path:
            sys.path.append(p)


# ---- 英語の文章の正規化 (これまで、ブラウザの tts.js で行っていた規則と同じ) ---------------------------------

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion"]
ORDINAL_FIX = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def _below_thousand(n: int) -> str:
    words = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        t = TENS[n // 10]
        words.append(f"{t}-{ONES[n % 10]}" if n % 10 else t)
    elif n > 0:
        words.append(ONES[n])
    return " ".join(words)


def integer_to_words(digits: str) -> str:
    clean = re.sub(r"^0+(?=[0-9])", "", digits)
    if clean == "0":
        return "zero"
    if len(clean) > 15:
        return " ".join(ONES[int(d)] for d in clean)
    groups = []
    end = len(clean)
    while end > 0:
        groups.append(int(clean[max(0, end - 3):end]))
        end -= 3
    words = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i]:
            words.append(_below_thousand(groups[i]) + (f" {SCALES[i]}" if SCALES[i] else ""))
    return " ".join(words)


def _number_to_words(token: str) -> str:
    whole, _, frac = token.replace(",", "").partition(".")
    head = integer_to_words(whole or "0")
    return f"{head} point {' '.join(ONES[int(d)] for d in frac)}" if frac else head


def _ordinal_words(digits: str) -> str:
    words = integer_to_words(digits)
    m = re.match(r"^(.*?)([a-z]+)$", words)
    prefix, last = m.group(1), m.group(2)
    if last in ORDINAL_FIX:
        return prefix + ORDINAL_FIX[last]
    return prefix + (f"{last[:-1]}ieth" if last.endswith("y") else f"{last}th")


def _is_latin_letter(c: str) -> bool:
    return unicodedata.name(c, "").startswith("LATIN")


def normalize_en(raw: str, vocab: dict | None = None) -> tuple[str | None, str | None]:
    """入力文を、語彙に含まれる文字だけの文にする。(文, None) または (None, エラーの文) を返す"""
    vocab = vocab if vocab is not None else CONSTANTS["en"]["vocab"]
    s = unicodedata.normalize("NFKC", raw).strip()
    if not s:
        return None, "文章を入力してください。"
    if js_len(raw) > EN_MAX_CHARS:
        return None, f"{EN_MAX_CHARS} 文字以内で入力してください。"
    foreign = next((c for c in s if unicodedata.category(c).startswith("L") and not _is_latin_letter(c)), None)
    if foreign:
        return None, f"SpeechT5 は英語専用のため、英語で入力してください（使用できない文字: {foreign}）。"
    s = re.sub(r"[‘’ʼ]", "'", s)
    s = re.sub(r"[“”]", '"', s).replace("…", "...")
    s = re.sub(r"[–−]", "-", s)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.category(c).startswith("M"))
    s = re.sub(r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)", r"\1 dollars", s)
    s = re.sub(r"([0-9])\s?%", r"\1 percent", s)
    s = re.sub(r"\b([0-9]+)(?:st|nd|rd|th)\b", lambda m: _ordinal_words(m.group(1)), s, flags=re.IGNORECASE)
    s = re.sub(r"[0-9][0-9,]*(?:\.[0-9]+)?", lambda m: _number_to_words(m.group(0)), s)
    s = s.replace("&", " and ").replace("+", " plus ").replace("=", " equals ").replace("@", " at ")
    s = "".join(c for c in s if c.isspace() or (c in vocab and len(c) == 1 and c != "▁"))
    s = re.sub(r"\s+", " ", s).strip()
    if not any(unicodedata.category(c).startswith("L") for c in s):
        return None, "読み上げられる英単語が含まれていません。"
    if not re.search(r"[.!?]$", s):
        s += "."
    return s, None


def encode_en(text: str, vocab: dict | None = None) -> list[int]:
    vocab = vocab if vocab is not None else CONSTANTS["en"]["vocab"]
    space, eos = vocab["▁"], vocab["</s>"]
    return [space] + [space if c == " " else vocab[c] for c in text] + [eos]


# ---- 日本語の音素化 (OpenJTalk のフルコンテキストラベル -> 音素と韻律記号) ----------------------------------

def _num(regex: str, s: str) -> int:
    m = re.search(regex, s)
    return -50 if m is None else int(m.group(1))      # 値が "xx" の場合は -50


def labels_to_symbols(labels: list[str]) -> list[str]:
    """栗原らの方式 (Prosodic features control by symbols as input of sequence-to-sequence acoustic modeling for neural TTS)。
    記号: ^ 文頭、$ 平叙文の文末、? 疑問文の文末、_ ポーズ、# アクセント句の境界、[ ピッチの上昇、] ピッチの下降。学習時の音素化と同じ"""
    out: list[str] = []
    n_labels = len(labels)
    for n, lab in enumerate(labels):
        p3 = re.search(r"\-(.*?)\+", lab).group(1)
        if p3 in ("A", "I", "U", "E", "O"):               # 無声化母音は通常の母音として扱う
            p3 = p3.lower()
        if p3 == "sil":
            out.append("^" if n == 0 else ("?" if _num(r"!(\d+)_", lab) == 1 else "$"))
            continue
        if p3 == "pau":
            out.append("_")
            continue
        out.append(p3)
        if n + 1 >= n_labels:
            continue
        a1 = _num(r"/A:([0-9\-]+)\+", lab)                # アクセント核からの相対位置
        a2 = _num(r"\+(\d+)\+", lab)                      # アクセント句内の前からのモーラ位置
        a3 = _num(r"\+(\d+)/", lab)                       # アクセント句内の後ろからのモーラ位置
        f1 = _num(r"/F:(\d+)_", lab)                      # アクセント句のモーラ数
        a2_next = _num(r"\+(\d+)\+", labels[n + 1])
        if a3 == 1 and a2_next == 1 and p3 in ("a", "i", "u", "e", "o", "N", "cl"):
            out.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            out.append("]")
        elif a2 == 1 and a2_next == 2:
            out.append("[")
    return out


JA_SYMBOLS: list[str] = CONSTANTS["ja"]["symbols"]
JA_BASE: int = CONSTANTS["ja"]["baseVocab"]
JA_EOS: int = CONSTANTS["ja"]["eos"]
_SYMBOL_INDEX = {s: i for i, s in enumerate(JA_SYMBOLS)}
JA_CONVERT_ERROR = "この文章は読み上げに変換できませんでした。"


def symbols_to_ids(symbols: list[str]) -> list[int] | None:
    """記号 -> モデルの入力 ID (末尾に </s>)。表にない記号があれば None"""
    try:
        return [JA_BASE + _SYMBOL_INDEX[s] for s in symbols] + [JA_EOS]
    except KeyError:
        return None


def split_ids(ids: list[int], start: int = SPLIT_FROM, limit: int = SPLIT_MAX) -> list[list[int]]:
    """入力長 (末尾の </s> を含む) が start 以上なら、ポーズ (_) の位置で、limit 記号以内の区間に分ける (分けない場合は [ids])。
    各区間は ^ で始まり、$ (最後の区間は元の文末記号) で終わる。短すぎる末尾 (20 記号未満) は、手前の区間につなぐ"""
    if len(ids) < start:
        return [ids]
    body = [JA_SYMBOLS[i - JA_BASE] for i in ids[:-1]]
    end = body[-1]
    if body[0] != "^" or end not in ("$", "?"):
        return [ids]
    clauses: list[list[str]] = []
    cur: list[str] = []
    for s in body[1:-1]:
        if s == "_":
            clauses.append(cur)
            cur = []
        else:
            cur.append(s)
    clauses.append(cur)
    segs: list[list[str]] = []
    cur = []
    for c in clauses:
        if cur and len(cur) + 1 + len(c) > limit:
            segs.append(cur)
            cur = []
        cur = cur + (["_"] if cur else []) + c
    segs.append(cur)
    if len(segs) > 1 and len(segs[-1]) < 20:
        last = segs.pop()
        segs[-1] = segs[-1] + ["_"] + last
    if len(segs) == 1:
        return [ids]
    out = [symbols_to_ids(["^"] + sg + [end if i == len(segs) - 1 else "$"]) for i, sg in enumerate(segs)]
    return out if all(out) else [ids]


class JaFrontend:
    """pyopenjtalk-plus の素の版 (use_vanilla=True) で、日本語の文を、モデルの入力 ID にする (評価したサイトの音素化と同じ)"""

    def __init__(self):
        import pyopenjtalk
        self.jt = pyopenjtalk
        self.lock = threading.Lock()
        self.jt.extract_fullcontext("テスト", use_vanilla=True)     # 辞書の読み込みを済ませておく

    def ids(self, text: str) -> list[int]:
        try:
            with self.lock:
                labels = self.jt.extract_fullcontext(text, use_vanilla=True)
        except Exception as e:  # noqa: BLE001
            raise ValueError(JA_CONVERT_ERROR) from e
        if not labels or len(labels) < 3:
            raise ValueError(JA_CONVERT_ERROR)
        ids = symbols_to_ids(labels_to_symbols(labels))
        if ids is None:
            raise ValueError(JA_CONVERT_ERROR)
        return ids


# ---- 日本語の合成 (ONNX Runtime) -----------------------------------------------------------------------

class JaSynth:
    def __init__(self, models_dir: Path, threads: int):
        import onnxruntime as ort
        self.np = __import__("numpy")
        manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sessions = {}
        for f in manifest["files"]:
            data = b"".join((models_dir / p).read_bytes() for p in f["parts"])       # 20 MiB ごとの分割ファイルを、つなぐ
            if len(data) != f["bytes"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
                raise RuntimeError(f"{f['file']} の内容が、manifest.json と合いません")
            self.sessions[f["key"]] = ort.InferenceSession(data, so, providers=["CPUExecutionProvider"])
        self.version = manifest.get("version")

    def encode(self, ids: list[int]) -> dict:
        np = self.np
        r = self.sessions["encoder"].run(None, {"input_ids": np.array([ids], dtype=np.int64)})
        out = dict(zip([o.name for o in self.sessions["encoder"].get_outputs()], r))
        cross = {}
        for i in range(LAYERS):
            cross[f"past_key_values.{i}.encoder.key"] = out[f"present.{i}.encoder.key"]
            cross[f"past_key_values.{i}.encoder.value"] = out[f"present.{i}.encoder.value"]
        return {"mask": out["encoder_attention_mask"], "cross": cross, "length": len(ids)}

    def decode(self, enc: dict, speaker, on_step, deadline: float, cancelled) -> "object":
        """自己回帰の生成。停止条件は、transformers の generate_speech と同じ (停止確率の和が 0.5 以上、上限は入力長の 5 倍) に、文の終わりの検出を加える"""
        np = self.np
        dec = self.sessions["decoder"]
        names = [o.name for o in dec.get_outputs()]
        past = dict(enc["cross"])
        for i in range(LAYERS):
            past[f"past_key_values.{i}.decoder.key"] = np.zeros((1, HEADS, 0, HEAD_DIM), np.float32)
            past[f"past_key_values.{i}.decoder.value"] = np.zeros((1, HEADS, 0, HEAD_DIM), np.float32)
        seq = np.zeros((1, 1, 80), np.float32)
        spk = np.asarray(speaker, dtype=np.float32).reshape(1, 512)
        max_steps = enc["length"] * 5
        parts, reached = [], 0
        step = 0
        while True:
            step += 1
            if cancelled():
                raise Cancelled()
            if time.time() > deadline:
                raise TimeoutError("合成に時間がかかりすぎたため、中止しました。")
            r = dict(zip(names, dec.run(None, {"speaker_embeddings": spk, "encoder_attention_mask": enc["mask"], "output_sequence": seq, **past})))
            for i in range(LAYERS):
                past[f"past_key_values.{i}.decoder.key"] = r[f"present.{i}.decoder.key"]
                past[f"past_key_values.{i}.decoder.value"] = r[f"present.{i}.decoder.value"]
            seq = r["output_sequence_out"]
            parts.append(np.asarray(r["spectrum"], dtype=np.float32).reshape(-1, 80))
            on_step(step)
            align = r["align"].reshape(-1)
            if reached == 0 and int(align.argmax()) >= align.shape[0] - 1 - END_TAIL:
                reached = step
            if float(r["prob"].sum()) >= 0.5 or step >= max_steps or (reached > 0 and step - reached >= END_MARGIN):
                break
        return np.concatenate(parts, axis=0)

    def vocode(self, spectrogram):
        out = self.sessions["vocoder"].run(None, {"spectrogram": self.np.ascontiguousarray(spectrogram, dtype=self.np.float32)})
        return self.np.asarray(out[0], dtype=self.np.float32).reshape(-1)

    def synthesize(self, pieces: list[list[int]], speaker, on_progress, deadline: float, cancelled):
        """区間ごとに合成して、無音を挟んでつなぐ。on_progress(これまでのステップ数)"""
        np = self.np
        waves, done = [], 0
        for ids in pieces:
            enc = self.encode(ids)
            last = [0]

            def on_step(step, base=done):
                last[0] = step
                on_progress(base + step)

            waves.append(self.vocode(self.decode(enc, speaker, on_step, deadline, cancelled)))
            done += last[0]
        if len(waves) == 1:
            return waves[0]
        gap = np.zeros(round(GAP_SEC * SAMPLE_RATE), dtype=np.float32)
        out = []
        for i, w in enumerate(waves):
            out += [w] + ([gap] if i < len(waves) - 1 else [])
        return np.concatenate(out)


# ---- 英語の合成 (PyTorch) -------------------------------------------------------------------------------

class EnSynth:
    def __init__(self, device: str):
        import torch
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.torch, self.device = torch, device
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").eval().to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval().to(device)

    def synthesize(self, ids: list[int], speaker):
        torch = self.torch
        n = len(ids)
        max_steps = min(n * 10, n * 6 + 100)       # 止まらない場合に備えた上限 (ブラウザで使っていた値と同じ)
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            s = torch.as_tensor(speaker, dtype=torch.float32, device=self.device).reshape(1, 512)
            wave = self.model.generate_speech(x, s, vocoder=self.vocoder, threshold=0.5, maxlenratio=max_steps * 2 / n)
        return wave.float().cpu().numpy().reshape(-1)


# ---- サービス --------------------------------------------------------------------------------------------

def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if v.shape[0] != 512 or not np.isfinite(n) or n == 0:
        raise ValueError("話者の特徴 (x-vector) が正しくありません。")
    return v / n


def _pcm16_b64(wave) -> str:
    import numpy as np
    return base64.b64encode((np.clip(wave, -1.0, 1.0) * 32767.0).round().astype(np.int16).tobytes()).decode("ascii")


class Service:
    """音声合成のモデルを持つ。start() で、別のスレッドに読み込みを始める。1 回に 1 件ずつ合成する (呼ぶ側が直列にする)"""

    def __init__(self, log=print):
        self.log = log
        self.state = "idle"          # idle -> loading -> ready
        self.en = False
        self.ja = False
        self.errors: dict = {}
        self.ja_front: JaFrontend | None = None
        self.ja_synth: JaSynth | None = None
        self.en_synth: EnSynth | None = None
        self.embedder = None
        self.ready = threading.Event()

    def start(self) -> None:
        self.state = "loading"
        threading.Thread(target=self._load, daemon=True).start()

    def info(self) -> dict:
        return {"state": self.state, "en": self.en, "ja": self.ja}

    def _load(self) -> None:
        t0 = time.time()
        try:
            add_extra_paths()
            import torch
            from server import core
            core.use_research_code()
            device = os.environ.get("TTS_DEVICE", "auto")
            device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
            threads = int(os.environ.get("TTS_THREADS", str(min(4, os.cpu_count() or 1))))
            torch.set_num_threads(threads)
            from anticloning_mask.speaker.encoders import build_encoder
            enc = build_encoder("xvector", device)

            def embed(wave):
                with torch.no_grad():
                    e = enc.embed(torch.from_numpy(wave).to(device))
                return _unit(e.reshape(-1).float().cpu().numpy())

            self.embedder = embed
            try:
                self.en_synth = EnSynth(device)
                self.en = True
            except Exception as e:  # noqa: BLE001
                self.errors["en"] = f"{type(e).__name__}: {e}"[:200]
            try:
                self.ja_front = JaFrontend()
                self.ja_synth = JaSynth(Path(os.environ.get("TTS_JA_DIR", REPO / "server" / "models_ja")), threads)
                self.ja = True
            except Exception as e:  # noqa: BLE001
                self.errors["ja"] = f"{type(e).__name__}: {e}"[:200]
            self.log(f"音声合成の準備ができました: 英語={'可' if self.en else '不可'} 日本語={'可' if self.ja else '不可'} 装置={device} ({time.time() - t0:.0f}s)")
            for k, v in self.errors.items():
                self.log(f"  {k} を使えません: {v}")
        except Exception as e:  # noqa: BLE001
            self.errors["all"] = f"{type(e).__name__}: {e}"[:200]
            self.log(f"音声合成の準備に失敗しました: {self.errors['all']}")
        finally:
            self.state = "ready"
            self.ready.set()

    # 入力中の案内 (モデルは使わない)。{lang, text} または {lang, error}
    def normalize(self, text: str) -> dict:
        lang = detect_lang(text)
        if lang == "ja":
            if self.state == "ready" and not self.ja:
                return {"lang": lang, "error": "この環境では、日本語の読み上げを用意していません。英語で入力してください。"}
            trimmed = text.strip()
            if not trimmed:
                return {"lang": lang, "error": "文章を入力してください。"}
            if js_len(text) > JA_MAX_CHARS:
                return {"lang": lang, "error": f"{JA_MAX_CHARS} 文字以内で入力してください。"}
            return {"lang": lang, "text": trimmed}
        shown, err = normalize_en(text)
        return {"lang": lang, "error": err} if err else {"lang": lang, "text": shown}

    def synthesize(self, text: str, emb: dict, progress, cancelled) -> dict:
        """本人の録音 (src) と、敵対的音声 (adv) の x-vector で、2 本合成する。
        progress(variant, step, steps)、cancelled() -> bool。{text, lang, segments, clean, protected} を返す。エラーは ValueError (利用者に見せてよい文)"""
        self.ready.wait()
        norm = self.normalize(text)
        if norm.get("error"):
            raise ValueError(norm["error"])
        lang, shown = norm["lang"], norm["text"]
        if lang == "ja" and not self.ja:
            raise ValueError("日本語の音声合成を用意できていません。")
        if lang == "en" and not self.en:
            raise ValueError("英語の音声合成を用意できていません。")
        src, adv, tgt = _unit(emb["src"]), _unit(emb["adv"]), _unit(emb["tgt"])
        if lang == "ja":
            pieces = split_ids(self.ja_front.ids(shown))
            expected = sum(len(p) * 2 for p in pieces)           # 実測で 1 記号あたり約 2 ステップ
        else:
            pieces = [encode_en(shown)]
            expected = 100                                          # 英語は、ステップごとの報告がないので、区間の始めと終わりだけ知らせる
        deadline = time.time() + DEADLINE_SEC
        out = {"text": shown, "lang": lang, "segments": len(pieces)}
        for variant, speaker in (("clean", src), ("protected", adv)):
            progress(variant, 0, expected)
            if lang == "ja":
                wave = self.ja_synth.synthesize(pieces, speaker, lambda s, v=variant: progress(v, min(s, expected), expected), deadline, cancelled)
            else:
                if cancelled():
                    raise Cancelled()
                wave = self.en_synth.synthesize(pieces[0], speaker)
            progress(variant, expected, expected)
            e = self.embedder(wave)
            out[variant] = {"cosSrc": float(e @ src), "cosTgt": float(e @ tgt), "sampleRate": SAMPLE_RATE, "pcm16": _pcm16_b64(wave)}
        return out
