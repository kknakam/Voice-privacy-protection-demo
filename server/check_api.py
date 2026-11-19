"""起動中のサーバの、クローンの合成の API (Step 3) の通し試験。モデルを実際に使う (GPU と重みが要る)。

  scripts/start.sh 8100            # 先に、サーバを起動する (音声合成のモデルの読み込みに、約 20 秒かかる)
  python server/check_api.py [http://127.0.0.1:8100] [--log server/work/serve.log]

確かめること: 状態、入力中の案内、不正な依頼の拒否、日本語と英語の合成、結果の取得が 1 回だけであること、
同じ接続元から同時に 2 件を受けないこと、中止、文章がログに残らないこと。終了コードは、NG があれば 1。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMB = json.loads((HERE / "testdata" / "emb_fixture.json").read_text(encoding="utf-8"))
EMB = {k: EMB[k] for k in ("src", "adv", "tgt")}
fail = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global fail
    print(("OK  " if ok else "NG  ") + name, extra)
    fail += not ok


def call(base: str, method: str, path: str, body=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def wait_done(base: str, jid: str, timeout: float = 120):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        code, s = call(base, "GET", f"/api/jobs/{jid}")
        last = s
        if s.get("state") in ("done", "error", "cancelled"):
            return s
        time.sleep(0.3)
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8100")
    ap.add_argument("--log", default=str(HERE / "work" / "serve.log"))
    a = ap.parse_args()
    b = a.base.rstrip("/")

    code, h = call(b, "GET", "/api/health")
    check("状態: API がある", code == 200 and h.get("ok") is True, json.dumps(h.get("tts")))
    for _ in range(120):                                              # モデルの読み込みが終わるまで待つ
        if call(b, "GET", "/api/health")[1]["tts"]["state"] == "ready":
            break
        time.sleep(1)
    tts = call(b, "GET", "/api/health")[1]["tts"]
    check("状態: 音声合成の準備ができ、英語と日本語が使える", tts == {"state": "ready", "en": True, "ja": True}, json.dumps(tts))

    n = lambda t: call(b, "POST", "/api/normalize", {"text": t})[1]  # noqa: E731
    check("案内: 英語は、数字を英単語に展開した文を返す", n("Hello 12 world") == {"lang": "en", "text": "Hello twelve world."}, json.dumps(n("Hello 12 world"), ensure_ascii=False))
    check("案内: 日本語は、日本語と判定する", n("こんにちは。") == {"lang": "ja", "text": "こんにちは。"})
    check("案内: 日本語 61 字は、上限のエラー", "60 文字以内" in n("あ" * 61).get("error", ""))
    check("案内: 空は、エラー", "error" in n("   "))
    check("案内: 本体が JSON でない依頼は、400", call(b, "POST", "/api/normalize", raw=b"xx")[0] == 400)

    ok_body = {"text": "今日は、いい天気です。", "emb": EMB}
    bad = [("本体が JSON でない", dict(raw=b"not json")), ("JSON が配列", dict(raw=b"[1,2]")), ("文章がない", dict(body={"emb": EMB})), ("emb がない", dict(body={"text": "Hi."})),
           ("emb の長さが違う", dict(body={"text": "Hi.", "emb": {**EMB, "src": [0.1] * 511}})), ("emb が 0", dict(body={"text": "Hi.", "emb": {**EMB, "adv": [0.0] * 512}})),
           ("emb に文字列", dict(body={"text": "Hi.", "emb": {**EMB, "tgt": ["a"] * 512}})), ("日本語 61 字", dict(body={"text": "あ" * 61, "emb": EMB})),
           ("英語 201 字", dict(body={"text": "a" * 201, "emb": EMB})), ("記号だけ", dict(body={"text": "!!!", "emb": EMB}))]
    for name, kw in bad:
        code, body = call(b, "POST", "/api/clones", **kw)
        check(f"拒否: {name} -> 400", code == 400 and "error" in body, f"{code} {body.get('error', '')[:40]}")
    code, _ = call(b, "POST", "/api/clones", raw=b"x" * (1 << 16))
    check("拒否: 大きすぎる依頼 -> 413", code == 413)

    secret = "ひみつの文章ですよ、ログに残してはいけません。"
    for label, text, lang in (("日本語", secret, "ja"), ("英語", "My voice belongs to me, thanks.", "en")):
        code, body = call(b, "POST", "/api/clones", {"text": text, "emb": EMB})
        check(f"{label}: 依頼を受け付ける (202)", code == 202 and "id" in body, str(code))
        jid = body["id"]
        if label == "日本語":
            code2, _ = call(b, "POST", "/api/clones", {"text": "second.", "emb": EMB})
            check("同じ接続元から、同時に 2 件目は、429", code2 == 429)
        s = wait_done(b, jid)
        check(f"{label}: 完了する", s and s["state"] == "done", json.dumps(s)[:100])
        code, r = call(b, "GET", f"/api/jobs/{jid}/result")
        check(f"{label}: 結果を取れる", code == 200 and r["lang"] == lang and r["text"] and r["segments"] >= 1, f"{code}")
        sec = [len(base64.b64decode(r[v]["pcm16"])) / 2 / r[v]["sampleRate"] for v in ("clean", "protected")]
        check(f"{label}: 2 本の音声の長さが妥当 (1〜15 秒)", all(1 < x < 15 for x in sec), f"{sec[0]:.2f} / {sec[1]:.2f} 秒")
        c, p = r["clean"], r["protected"]
        check(f"{label}: 保護なしは本人に近く、保護ありはターゲットに近づく", c["cosSrc"] > c["cosTgt"] and p["cosTgt"] > p["cosSrc"] and p["cosSrc"] < c["cosSrc"],
              f"clean {c['cosSrc']:.3f}/{c['cosTgt']:.3f}  protected {p['cosSrc']:.3f}/{p['cosTgt']:.3f}")
        check(f"{label}: 結果は 1 回で消える (2 回目は 404)", call(b, "GET", f"/api/jobs/{jid}/result")[0] == 404)

    long_text = "友人から誕生日にもらった手紙を、何度も読み返しているうちに、遠くに住んでいても、心はつながっていると感じました。"
    code, body = call(b, "POST", "/api/clones", {"text": long_text, "emb": EMB})
    s = wait_done(b, body["id"])
    r = call(b, "GET", f"/api/jobs/{body['id']}/result")[1]
    check("長い日本語: 区間に分けて合成する (3 区間)", s["state"] == "done" and r["segments"] >= 2, f"区間 {r.get('segments')}")

    code, body = call(b, "POST", "/api/clones", {"text": long_text, "emb": EMB})
    time.sleep(0.8)
    st = call(b, "GET", f"/api/jobs/{body['id']}")[1]
    check("進み具合: 実行中は、step と steps と variant を返す", st["state"] == "running" and st["steps"] > 0 and st["variant"] in ("clean", "protected"), json.dumps(st))
    call(b, "DELETE", f"/api/jobs/{body['id']}")
    s = wait_done(b, body["id"])
    check("中止: 実行中の合成を止められる", s["state"] == "cancelled", json.dumps(s))
    code, _ = call(b, "POST", "/api/clones", {"text": "after cancel.", "emb": EMB})
    check("中止のあと、すぐに、次の依頼を受け付ける", code == 202, str(code))
    wait_done(b, call(b, "POST", "/api/clones", {"text": "x", "emb": EMB})[1].get("id", ""), timeout=1) if False else None

    time.sleep(3)
    log = Path(a.log)
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        check("ログに、入力した文章が残らない", secret not in text and "ひみつ" not in text and "My voice belongs" not in text)
        check("ログに、結果の記録がある (文章の長さと言語だけ)", "clone " in text and "lang=" in text and "chars=" in text)
    else:
        print("SKIP ログの検査 (ログのファイルが見つかりません:", log, ")")
    code, _ = call(b, "GET", "/api/health")
    check("最後も、サーバが応答する", code == 200)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
