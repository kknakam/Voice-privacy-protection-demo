"""
このフォルダを、Cloudflare Pages と同じ _headers を付けてローカル配信する。

マイクは localhost でも使える。COOP/COEP が付くのでマルチスレッドの動作も確認できる。

実行コマンド: python serve.py [ポート番号]  (既定 8000)
"""
from __future__ import annotations

import fnmatch
import http.server
import sys
from functools import partial
from pathlib import Path

SITE = Path(__file__).resolve().parent


def parse_headers(path: Path) -> list:
    rules, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            current = (line.strip(), [])
            rules.append(current)
        else:
            key, value = line.strip().split(":", 1)
            current[1].append((key.strip(), value.strip()))
    return rules


RULES = parse_headers(SITE / "_headers")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".woff2": "font/woff2", ".svg": "image/svg+xml", ".onnx": "application/octet-stream"}

    def end_headers(self):
        url = self.path.split("?")[0]
        for pattern, headers in RULES:
            if fnmatch.fnmatch(url, pattern):
                for key, value in headers:
                    self.send_header(key, value)
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://localhost:{port}/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(SITE))).serve_forever()
