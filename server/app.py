"""サイト (site/) と、敵対的音声を作る API を、1 つのサーバで配信する。scripts/serve.sh から呼ぶ。

  scripts/serve.sh [ポート番号]      既定 8000。環境変数 HOST (既定 127.0.0.1)、MAX_QUEUE (順番待ちの上限、既定 5)

静的ファイルは、tools/serve.py と同じく、site/_headers のヘッダを付けて返す。API は、すべて同じ出どころ (同じ URL の下) にある。

  GET    /api/health            {"ok": true, "queued": 順番待ちの数}
  POST   /api/jobs?steps=100    本体は、録音の WAV (16 kHz、モノラル、16 bit)。202 {"id": ...}
  GET    /api/jobs/<id>         {"state": queued|running|done|error|cancelled, "position", "step", "steps", "message"}
  GET    /api/jobs/<id>/result  完了した結果 (JSON)。1 回取ると、サーバから消える
  DELETE /api/jobs/<id>         中止

録音は、処理が終わるとすぐに、サーバのディスクから消す (結果は、取りに来るまで、メモリにだけ置く。10 分で捨てる)。
最適化は、1 件ずつ、scripts/run_attack.sh を、別のプロセスで呼んで行う (GPU を 1 件ずつ使う)。
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
from server import core  # noqa: E402

SCRIPT = REPO / "scripts" / "run_attack.sh"
JOBS_DIR = REPO / "server" / "work" / "jobs"
MAX_UPLOAD = 1 << 20          # 録音の WAV (最長 8 秒で、約 256 KB) の上限
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "5"))
JOB_TIMEOUT = 300             # 1 件の上限 (秒)
KEEP_RESULT = 600             # 結果を、取りに来るまで置いておく時間 (秒)

spec = importlib.util.spec_from_file_location("static_serve", REPO / "tools" / "serve.py")
static_serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_serve)


class Job:
    def __init__(self, wav: bytes, steps: int, owner: str):
        self.id = secrets.token_urlsafe(12)
        self.wav, self.steps, self.owner = wav, steps, owner
        self.state, self.step, self.message = "queued", 0, ""
        self.result: bytes | None = None
        self.created = self.finished = time.time()
        self.proc: subprocess.Popen | None = None
        self.cancelled = False


LOCK = threading.Lock()
JOBS: dict = {}
ORDER: deque = deque()          # 順番待ちの Job
WAKE = threading.Condition(LOCK)


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


def worker() -> None:
    while True:
        with WAKE:
            while not ORDER:
                WAKE.wait(timeout=30)
                now = time.time()                      # ついでに、古い結果を捨てる
                for jid in [j.id for j in JOBS.values() if j.state in ("done", "error", "cancelled") and now - j.finished > KEEP_RESULT]:
                    JOBS.pop(jid, None)
            job = ORDER.popleft()
            if job.cancelled:
                continue
            job.state = "running"
        run_job(job)


def position(job: Job) -> int:
    with LOCK:
        if job.state == "queued":
            return list(ORDER).index(job) + 1 + (1 if any(j.state == "running" for j in JOBS.values()) else 0)
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
                return self.send_json(200, {"ok": True, "queued": len(ORDER)})
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
        self.send_json(200, {"state": job.state, "position": position(job), "step": job.step, "steps": job.steps, "message": job.message})

    def do_POST(self):
        if not self.is_api():
            return self.send_json(405, {"error": "使えない操作です"})
        url = urlparse(self.path)
        if url.path != "/api/jobs":
            return self.send_json(404, {"error": "見つかりません"})
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
        owner = self.client_ip()
        with LOCK:
            if any(j.owner == owner and j.state in ("queued", "running") for j in JOBS.values()):
                return self.send_json(429, {"error": "前の処理が終わってから、もう一度お試しください"})
            if len(ORDER) >= MAX_QUEUE:
                return self.send_json(503, {"error": "混み合っています。しばらくしてから、もう一度お試しください"})
            job = Job(wav, steps, owner)
            JOBS[job.id] = job
            ORDER.append(job)
            WAKE.notify()
        self.send_json(202, {"id": job.id})

    def do_DELETE(self):
        m = re.fullmatch(r"/api/jobs/([\w-]+)", urlparse(self.path).path)
        job = JOBS.get(m.group(1)) if m else None
        if not job:
            return self.send_json(404, {"error": "見つかりません"})
        job.cancelled = True
        with LOCK:
            if job.state == "queued":
                job.state, job.message, job.finished = "cancelled", "中止しました", time.time()
                if job in ORDER:
                    ORDER.remove(job)
                job.wav = None
        if job.proc and job.proc.poll() is None:
            try:
                os.killpg(job.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.send_json(200, {"state": "cancelled"})

    def log_message(self, *args):
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    host = os.environ.get("HOST", "127.0.0.1")
    shutil.rmtree(JOBS_DIR, ignore_errors=True)       # 前回の異常終了で残った、録音の作業フォルダを消す
    threading.Thread(target=worker, daemon=True).start()
    log(f"http://{host}:{port}/  (順番待ちの上限 {MAX_QUEUE} 件)")
    http.server.ThreadingHTTPServer((host, port), partial(Handler, directory=str(static_serve.SITE))).serve_forever()


if __name__ == "__main__":
    main()
