"""サイト (site/) と、敵対的音声を作る API を、1 つのサーバで配信する。scripts/serve.sh から呼ぶ。

  scripts/serve.sh [ポート番号]      既定 8000。環境変数 HOST (既定 127.0.0.1)、MAX_QUEUE (順番待ちの上限、既定 5)

静的ファイルは、tools/serve.py と同じく、site/_headers のヘッダを付けて返す。API は、すべて同じ出どころ (同じ URL の下) にある。

  GET    /api/health            {"ok": true, "queued": 順番待ちの数, "tts": {"state": loading|ready, "en": 可否, "ja": 可否}}
  POST   /api/jobs?steps=100    本体は、録音の WAV (16 kHz、モノラル、16 bit)。202 {"id": ...}   (Step 2: 敵対的音声の生成)
  POST   /api/clones            本体は、JSON {"text": 文章, "emb": {"src": [512], "adv": [512], "tgt": [512]}}。202 {"id": ...}   (Step 3: クローンの合成)
  POST   /api/normalize         本体は、JSON {"text": 文章}。{"lang", "text"} または {"lang", "error"}  (入力中の案内。モデルは使わない)
  GET    /api/jobs/<id>         {"state": queued|running|done|error|cancelled, "position", "step", "steps", "variant", "message"}
  GET    /api/jobs/<id>/result  完了した結果 (JSON)。1 回取ると、サーバから消える
  DELETE /api/jobs/<id>         中止

録音と入力した文章は、処理が終わるとすぐに、サーバから消す (録音はディスクから。結果は、取りに来るまで、メモリにだけ置く。10 分で捨てる)。
敵対的音声の生成は、1 件ずつ、scripts/run_attack.sh を、別のプロセスで呼んで行う (GPU を 1 件ずつ使う)。
クローンの合成は、このプロセスの中で、1 件ずつ行う (モデルは、起動時に 1 回だけ読み込む。server/tts.py)。2 つの処理は、順番待ちが別で、同時に動ける。
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from server import core, tts  # noqa: E402

SCRIPT = REPO / "scripts" / "run_attack.sh"
JOBS_DIR = REPO / "server" / "work" / "jobs"
MAX_UPLOAD = 1 << 20          # 録音の WAV (最長 8 秒で、約 256 KB) の上限
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "5"))
MAX_JSON = 1 << 15            # クローンの依頼 (文章と、3 つの x-vector) の JSON の上限。x-vector 3 本で約 10 KB
MAX_NORMALIZE = 1 << 13       # 入力中の案内の依頼の上限
JOB_TIMEOUT = 300             # 敵対的音声の生成 1 件の上限 (秒)
KEEP_RESULT = 600             # 結果を、取りに来るまで置いておく時間 (秒)

spec = importlib.util.spec_from_file_location("static_serve", REPO / "tools" / "serve.py")
static_serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_serve)


class Job:
    def __init__(self, wav: bytes | None, steps: int, owner: str, kind: str = "attack", payload: dict | None = None):
        self.id = secrets.token_urlsafe(12)
        self.wav, self.steps, self.owner = wav, steps, owner
        self.kind, self.payload, self.variant = kind, payload, ""      # kind: attack (敵対的音声の生成) | clone (クローンの合成)
        self.state, self.step, self.message = "queued", 0, ""
        self.result: bytes | None = None
        self.created = self.finished = time.time()
        self.proc: subprocess.Popen | None = None
        self.cancelled = False


LOCK = threading.Lock()
JOBS: dict = {}
ORDER: deque = deque()          # 敵対的音声の生成の、順番待ちの Job
ORDER_CLONE: deque = deque()    # クローンの合成の、順番待ちの Job
WAKE = threading.Condition(LOCK)
WAKE_CLONE = threading.Condition(LOCK)
TTS = tts.Service(log=lambda m: log(m))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_job(job: Job) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="job-", dir=JOBS_DIR))
    started = time.time()
    last_error = ""
    try:
        (work / "in.wav").write_bytes(job.wav)
        job.wav = None
        cmd = ["bash", str(SCRIPT), "--input", str(work / "in.wav"), "--out-dir", str(work / "out"), "--steps", str(job.steps)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        job.proc = proc
        timer = threading.Timer(JOB_TIMEOUT, lambda: os.killpg(proc.pid, signal.SIGKILL))
        timer.start()
        try:
            for line in proc.stdout:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue                      # 警告などの、JSON ではない行は読み飛ばす
                if o.get("kind") == "loading":
                    job.message = "モデルを読み込んでいます"
                elif o.get("kind") == "step":
                    job.step, job.message = o["step"], ""
                elif o.get("kind") == "measuring":
                    job.message = "音質を測っています"
                elif o.get("kind") == "error":
                    last_error = str(o.get("message", ""))
            rc = proc.wait()
        finally:
            timer.cancel()
        out = work / "out" / "result.json"
        if job.cancelled:
            job.state, job.message = "cancelled", "中止しました"
        elif rc == 0 and out.exists():
            job.result = out.read_bytes()
            job.state, job.message = "done", ""
        else:
            job.state = "error"
            job.message = last_error.split(": ", 1)[-1] if last_error.startswith("ValueError") else "敵対的音声の生成に失敗しました"
            log(f"job {job.id[:6]} 失敗 rc={rc} {last_error[:200]}")
        log(f"job {job.id[:6]} {job.state} steps={job.steps} {time.time() - started:.1f}s")
    except Exception as e:  # noqa: BLE001
        job.state, job.message = "error", "敵対的音声の生成に失敗しました"
        log(f"job {job.id[:6]} 例外 {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)       # 録音と、作った音声のファイルを、すぐに消す
        job.proc, job.wav, job.finished = None, None, time.time()


def sweep() -> None:
    """取りに来なかった古い結果を捨てる (LOCK を持って呼ぶ)"""
    now = time.time()
    for jid in [j.id for j in JOBS.values() if j.state in ("done", "error", "cancelled") and now - j.finished > KEEP_RESULT]:
        JOBS.pop(jid, None)


def run_clone(job: Job) -> None:
    started = time.time()
    payload = job.payload
    lang, chars = tts.detect_lang(payload["text"]), len(payload["text"])
    try:
        def progress(variant: str, step: int, steps: int) -> None:
            job.variant, job.step, job.steps = variant, step, steps

        if TTS.state != "ready":
            job.message = "音声合成のモデルを読み込んでいます"        # 起動の直後だけ。読み込みが終わるまで待つ
        result = TTS.synthesize(payload["text"], payload["emb"], progress, lambda: job.cancelled)
        result["seconds"] = round(time.time() - started, 1)
        job.result = json.dumps(result, ensure_ascii=False).encode("utf-8")
        job.state, job.message = ("cancelled", "中止しました") if job.cancelled else ("done", "")
    except tts.Cancelled:
        job.state, job.message = "cancelled", "中止しました"
    except (ValueError, TimeoutError) as e:
        job.state, job.message = "error", str(e)
    except Exception as e:  # noqa: BLE001
        job.state, job.message = "error", "音声の合成に失敗しました"
        log(f"clone {job.id[:6]} 例外 {type(e).__name__}")           # 例外の文面には、入力した文章の断片が入ることがあるので、型名だけ記録する
    finally:
        job.payload, job.finished = None, time.time()            # 文章を、すぐに捨てる
    log(f"clone {job.id[:6]} {job.state} lang={lang} chars={chars} {time.time() - started:.1f}s")    # 文章そのものは、記録しない


def worker() -> None:
    while True:
        with WAKE:
            while not ORDER:
                WAKE.wait(timeout=30)
                sweep()
            job = ORDER.popleft()
            if job.cancelled:
                continue
            job.state = "running"
        run_job(job)


def worker_clone() -> None:
    while True:
        with WAKE_CLONE:
            while not ORDER_CLONE:
                WAKE_CLONE.wait(timeout=30)
                sweep()
            job = ORDER_CLONE.popleft()
            if job.cancelled:
                continue
            job.state = "running"
        run_clone(job)


def position(job: Job) -> int:
    with LOCK:
        if job.state == "queued":
            queue = ORDER if job.kind == "attack" else ORDER_CLONE
            return list(queue).index(job) + 1 + (1 if any(j.state == "running" and j.kind == job.kind for j in JOBS.values()) else 0)
        return 0


class Handler(static_serve.Handler):
    server_version = "VoiceDemo"

    def client_ip(self) -> str:
        peer = self.client_address[0]
        if peer in ("127.0.0.1", "::1"):               # 前段 (Cloudflare Tunnel など) から来た場合は、元の接続元を使う
            fwd = self.headers.get("CF-Connecting-IP") or (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            return fwd or peer
        return peer

    def send_json(self, code: int, obj, raw: bytes | None = None) -> None:
        body = raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        http.server.BaseHTTPRequestHandler.end_headers(self)      # _headers の規則は、付けない (静的ファイル用)
        self.wfile.write(body)

    def is_api(self) -> bool:
        return urlparse(self.path).path.startswith("/api/")

    def do_GET(self):
        if not self.is_api():
            return super().do_GET()
        path = urlparse(self.path).path
        if path == "/api/health":
            with LOCK:
                return self.send_json(200, {"ok": True, "queued": len(ORDER) + len(ORDER_CLONE), "tts": TTS.info()})
        m = re.fullmatch(r"/api/jobs/([\w-]+)(/result)?", path)
        job = JOBS.get(m.group(1)) if m else None
        if not job:
            return self.send_json(404, {"error": "見つかりません"})
        if m.group(2):
            if job.state != "done" or job.result is None:
                return self.send_json(409, {"error": "まだ完了していません"})
            raw, job.result = job.result, None
            with LOCK:
                JOBS.pop(job.id, None)                    # 1 回取ったら、サーバから消す
            return self.send_json(200, None, raw)
        self.send_json(200, {"state": job.state, "position": position(job), "step": job.step, "steps": job.steps, "variant": job.variant, "message": job.message})

    def read_json(self, limit: int):
        """本体を JSON として読む。(値, None) または (None, (HTTP の状態, メッセージ))"""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > limit:
            return None, (413, {"error": "依頼のデータの大きさが、範囲外です"})
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None, (400, {"error": "依頼の形式が正しくありません"})
        if not isinstance(body, dict):
            return None, (400, {"error": "依頼の形式が正しくありません"})
        return body, None

    def enqueue(self, job: Job, order: deque, wake: threading.Condition) -> None:
        """同じ接続元から同時に 1 件まで (種類をまたいで)、順番待ちは MAX_QUEUE 件まで"""
        with LOCK:
            if any(j.owner == job.owner and j.state in ("queued", "running") for j in JOBS.values()):
                code, body = 429, {"error": "前の処理が終わってから、もう一度お試しください"}
            elif len(order) >= MAX_QUEUE:
                code, body = 503, {"error": "混み合っています。しばらくしてから、もう一度お試しください"}
            else:
                JOBS[job.id] = job
                order.append(job)
                wake.notify()
                code, body = 202, {"id": job.id}
        self.send_json(code, body)

    def do_POST(self):
        if not self.is_api():
            return self.send_json(405, {"error": "使えない操作です"})
        url = urlparse(self.path)
        if url.path == "/api/jobs":
            return self.post_attack(url)
        if url.path == "/api/clones":
            return self.post_clone()
        if url.path == "/api/normalize":
            return self.post_normalize()
        self.send_json(404, {"error": "見つかりません"})

    def post_normalize(self):
        body, err = self.read_json(MAX_NORMALIZE)
        if err:
            return self.send_json(*err)
        text = body.get("text")
        if not isinstance(text, str):
            return self.send_json(400, {"error": "依頼の形式が正しくありません"})
        self.send_json(200, TTS.normalize(text))

    def post_clone(self):
        body, err = self.read_json(MAX_JSON)
        if err:
            return self.send_json(*err)
        text, emb = body.get("text"), body.get("emb")
        if not isinstance(text, str) or not isinstance(emb, dict):
            return self.send_json(400, {"error": "依頼の形式が正しくありません"})
        try:
            for k in ("src", "adv", "tgt"):
                tts._unit(emb.get(k))                      # 長さ 512 の、有限で、0 でない数列か
        except (ValueError, TypeError):
            return self.send_json(400, {"error": "話者の特徴 (x-vector) が正しくありません"})
        norm = TTS.normalize(text)
        if norm.get("error"):
            return self.send_json(400, {"error": norm["error"]})
        if TTS.state == "ready" and not getattr(TTS, norm["lang"]):
            return self.send_json(503, {"error": "日本語の音声合成を用意できていません" if norm["lang"] == "ja" else "英語の音声合成を用意できていません"})
        job = Job(None, 0, self.client_ip(), kind="clone", payload={"text": text, "emb": {k: emb[k] for k in ("src", "adv", "tgt")}})
        self.enqueue(job, ORDER_CLONE, WAKE_CLONE)

    def post_attack(self, url):
        try:
            steps = int(parse_qs(url.query).get("steps", ["100"])[0])
        except ValueError:
            steps = -1
        if steps not in core.STEPS_ALLOWED:
            return self.send_json(400, {"error": "反復回数は、50・100・200 のどれかです"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            return self.send_json(413, {"error": "録音のデータの大きさが、範囲外です"})
        wav = self.rfile.read(length)
        if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            return self.send_json(400, {"error": "WAV の録音を送ってください"})
        self.enqueue(Job(wav, steps, self.client_ip()), ORDER, WAKE)

    def do_DELETE(self):
        m = re.fullmatch(r"/api/jobs/([\w-]+)", urlparse(self.path).path)
        job = JOBS.get(m.group(1)) if m else None
        if not job:
            return self.send_json(404, {"error": "見つかりません"})
        job.cancelled = True
        with LOCK:
            if job.state == "queued":
                job.state, job.message, job.finished = "cancelled", "中止しました", time.time()
                for order in (ORDER, ORDER_CLONE):
                    if job in order:
                        order.remove(job)
                job.wav = job.payload = None
        if job.proc and job.proc.poll() is None:
            try:
                os.killpg(job.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.send_json(200, {"state": "cancelled"})

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    # 接続の待ち行列。標準は 5 で、前段 (Cloudflare Tunnel など) が、ページの資源ごとに接続を同時に張ると、あふれた接続が、
    # 1 秒、3 秒、7 秒と、間隔を空けて再試行されて、読み込みが止まる (同時 200 接続で、最大約 30 秒止まるのを確かめた)
    request_queue_size = 128


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    host = os.environ.get("HOST", "127.0.0.1")
    shutil.rmtree(JOBS_DIR, ignore_errors=True)       # 前回の異常終了で残った、録音の作業フォルダを消す
    work = JOBS_DIR.parent
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)                                    # speechbrain が、作業用のフォルダ (pretrained_models/) を、実行した場所に作る。リポジトリを汚さない
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=worker_clone, daemon=True).start()
    if os.environ.get("TTS_DISABLE") == "1":          # 音声合成を使わない (モデルを読み込まない。試験用)
        TTS.state = "ready"
        TTS.ready.set()
    else:
        TTS.start()                                   # モデルの読み込みは、別のスレッド。読み込み中も、サイトと Step 2 は使える
    log(f"http://{host}:{port}/  (順番待ちの上限 {MAX_QUEUE} 件ずつ。敵対的音声の生成と、クローンの合成で、別の順番待ち)")
    Server((host, port), partial(Handler, directory=str(static_serve.SITE))).serve_forever()


if __name__ == "__main__":
    main()
