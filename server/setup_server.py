"""研究室のサーバへ、このデモを置く作業を、自動で行う。scripts/setup_server.sh から呼ぶ (標準ライブラリだけで動く。Python 3.8 以上)。

  scripts/setup_server.sh --mask-root /path/to/AntiCloning_mask \\
      --from-hf 元のホスト:/path/to/hf --from-hub 元のホスト:/path/to/torch_hub [--smoke] [--start]
  scripts/setup_server.sh --dry-run ...        何を実行するかだけを表示する (何も書き込まない)

行うこと (どの工程も、やり直せる。すでに済んでいれば、飛ばす):
  1. 事前の点検: OS、Python、空き容量、GPU とドライバ (GPU がなければ、理由を示して止まる)
  2. 仮想環境の作成 (venv が使えない環境では、get-pip.py で pip を入れる方法に切り替える)
  3. PyTorch (ドライバに合う CUDA の版) と、requirements.txt のインストール
  4. 研究のリポジトリの場所の確認
  5. 重みのコピーと SHA-256 の照合 (scripts/copy_weights.sh)
  6. server/env.local の作成 (すでにあれば、上書きしない)
  7. 通しの点検 (scripts/check_env.sh --smoke)
  8. systemd の設定ファイルの生成 (server/work/voice-demo.service。置くには sudo が要るので、コマンドを表示するだけ)
  9. (--start) サーバの起動
行わないこと: HTTPS での公開 (Cloudflare Tunnel、nginx)、sudo が要る作業、研究のリポジトリや重みの取得元の用意。

終了コード: 0 = 完了、1 = 失敗または未完了の工程がある、2 = 使い方の誤り
"""
from __future__ import annotations

import argparse
import getpass
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "server" / "work"
MIN_FREE_GIB = 15              # 仮想環境 (約 7 GiB) と、ダウンロードの一時領域 (約 4 GiB) と、余裕
MIN_GPU_MIB = 4096             # 敵対的音声の生成 約 1.4 GiB + 英語の音声合成の常駐 約 1.4 GiB + 余裕
TORCH_PIN = "torch==2.8.0"
AUDIO_PIN = "torchaudio==2.8.0"
PYTHON_PREFERENCE = ["python3.10", "python3.11", "python3.12", "python3.9", "python3"]    # 動作確認は 3.10 だけ

results: list = []


def say(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def info(msg: str) -> None:
    print(f"      {msg}", flush=True)


# ---- 判断の部分 (外部のコマンドを呼ばない。試験できる) ---------------------------------------------------

def parse_cuda_version(nvidia_smi_text: str):
    """nvidia-smi の見出しの「CUDA Version: 12.8」から (12, 8)。なければ None"""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi_text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_gpu_query(csv_text: str) -> list:
    """nvidia-smi --query-gpu=name,driver_version,memory.total[,compute_cap] --format=csv,noheader,nounits の出力"""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[2].replace(".", "").isdigit():
            continue
        cc = None
        if len(parts) >= 4:
            try:
                cc = float(parts[3])
            except ValueError:
                cc = None
        gpus.append({"name": parts[0], "driver": parts[1], "mem_mib": int(float(parts[2])), "cc": cc})
    return gpus


def choose_torch_build(cuda_version, compute_cap=None, override: str = "auto"):
    """ドライバが対応する CUDA の版 (nvidia-smi の表示) から、PyTorch 2.8.0 の、どの版を入れるかを決める。
    (タグ、追加の index-url または None、注意の文、エラーの文 または None) を返す。
    PyPI の既定の PyTorch 2.8.0 (Linux) は CUDA 12.8 版。それより古いドライバには、cu126 の版を使う。"""
    index = {"cu126": "https://download.pytorch.org/whl/cu126", "cu129": "https://download.pytorch.org/whl/cu129", "cu128": None}
    if override != "auto":
        return override, index[override], [f"--cuda {override} の指定に従います (ドライバとの適合は確かめません)"], None
    if cuda_version is None:
        return None, None, [], "ドライバが対応する CUDA の版を、nvidia-smi から読み取れませんでした。--cuda cu126 / cu128 / cu129 で指定してください"
    notes = []
    if cuda_version < (12, 0):
        return None, None, [], f"ドライバが対応する CUDA は {cuda_version[0]}.{cuda_version[1]} で、古すぎます (12.0 以上が要ります)。ドライバの更新が必要です"
    if cuda_version >= (12, 8):
        tag = "cu128"
    else:
        tag = "cu126"
        if cuda_version < (12, 6):
            notes.append(f"ドライバの CUDA が {cuda_version[0]}.{cuda_version[1]} で、cu126 より古いため、CUDA 12 の互換機能で動かします (未確認)。動かなければ、ドライバの更新が必要です")
    if compute_cap is not None and compute_cap >= 12.0 and tag != "cu128":
        return None, None, [], f"この GPU (compute capability {compute_cap}) には、CUDA 12.8 以上に対応したドライバが要ります (現在は {cuda_version[0]}.{cuda_version[1]})。ドライバの更新が必要です"
    return tag, index[tag], notes, None


def pick_python(versions: dict):
    """候補の Python (コマンド名 -> (主, 副)) から、使うものを選ぶ。動作確認は 3.10。3.9〜3.12 なら動く見込み (torch 2.8 の対応範囲)"""
    for name in PYTHON_PREFERENCE:
        v = versions.get(name)
        if v and (3, 9) <= v <= (3, 12):
            return name, v
    return None, None


def render_env(mask_root: Path, python: Path, hf_home: Path, hub_dir: Path) -> str:
    return (
        "# scripts/setup_server.sh が作った、この機械の設定 (server/env.example が、書き方の説明)。git には入れない\n"
        f"ANTICLONING_MASK_ROOT={mask_root}\n"
        f"PYTHON={python}\n"
        f"HF_HOME={hf_home}\n"
        f"TORCH_HUB_DIR={hub_dir}\n"
    )


def parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip("'\"")
    return out


def render_unit(template: str, user: str, repo: Path, port: int) -> str:
    out = template.replace("User=YOUR_USER", f"User={user}").replace("WorkingDirectory=/path/to/demo_site", f"WorkingDirectory={repo}")
    return out.replace("ExecStart=/path/to/demo_site/scripts/serve.sh 8000", f"ExecStart={repo}/scripts/serve.sh {port}")


# ---- 実行の部分 ----------------------------------------------------------------------------------------------

class Runner:
    def __init__(self, dry: bool):
        self.dry = dry

    def run(self, cmd, env=None, capture=True, log=None, cwd=None, probe=False):
        """(終了コード, 出力)。コマンドが見つからなければ (127, '')。
        --dry-run では、表示だけして、(0, '') を返す。ただし、probe=True (何も変更しない、読み取りだけの確認) は、実際に実行する"""
        shown = " ".join(str(c) for c in cmd)
        if self.dry and not probe:
            info(f"(dry-run) {shown}")
            return 0, ""
        try:
            if log:
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"\n$ {shown}\n")
                    f.flush()
                    p = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)
                return p.returncode, ""
            p = subprocess.run([str(c) for c in cmd], env=env, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None, text=True, cwd=cwd)
            return p.returncode, (p.stdout or "") if capture else ""
        except FileNotFoundError:
            return 127, ""


def python_version(r: Runner, cmd: str):
    rc, out = r.run([cmd, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"], probe=True)
    if rc != 0:
        return None
    try:
        return tuple(int(x) for x in out.strip().split("."))[:2]
    except ValueError:
        return None


def step_preflight(r: Runner, args) -> dict:
    ctx = {}
    ok = sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")
    say("OK" if ok else "NG", "OS", f"{platform.system()} {platform.machine()}" + ("" if ok else " (Linux の x86_64 だけを確かめています)"))
    # Python
    if args.python:
        v = python_version(r, args.python)
        name = args.python
    else:
        versions = {n: python_version(r, n) for n in PYTHON_PREFERENCE if shutil.which(n)}
        name, v = pick_python({k: x for k, x in versions.items() if x})
    if name and v:
        say("OK" if v == (3, 10) else "注意", "Python", f"{name} ({v[0]}.{v[1]})" + ("" if v == (3, 10) else "  動作確認は 3.10 だけです (3.9〜3.12 なら、動く見込み)"))
    else:
        say("NG", "Python", "3.9〜3.12 の python3 が見つかりません。--python で指定してください (Ubuntu 22.04 なら python3.10)")
    ctx["python"] = name
    # 空き容量
    probe = Path(args.venv).resolve().parent
    while not probe.exists() and probe != probe.parent:        # まだないフォルダは、いちばん近い、あるフォルダで測る (ここでは、何も作らない)
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 2**30
    say("OK" if free >= MIN_FREE_GIB else "NG", "空き容量 (仮想環境と、一時領域)", f"{free:.0f} GiB ({probe})。{MIN_FREE_GIB} GiB 以上が要ります")
    # GPU
    rc, out = r.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"], probe=True)
    if rc != 0 and "compute_cap" in out:                       # 古いドライバは、compute_cap を問い合わせられない
        rc, out = r.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], probe=True)
    gpus = parse_gpu_query(out) if rc == 0 else []
    rc2, header = r.run(["nvidia-smi"], probe=True)
    cuda = parse_cuda_version(header) if rc2 == 0 else None
    ctx["gpus"], ctx["cuda"] = gpus, cuda
    if not gpus:
        reason = "nvidia-smi が見つかりません" if rc == 127 else "nvidia-smi が失敗しました (GPU がないか、ドライバが入っていません)"
        if args.allow_no_gpu:
            say("注意", "GPU", f"{reason}。--allow-no-gpu のため続けますが、敵対的音声の生成 (Step 2) は、GPU がないと動きません")
        else:
            say("NG", "GPU", f"{reason}。このサーバでは、敵対的音声の生成 (Step 2) が動きません。GPU のある機械で実行してください (--allow-no-gpu で、続けることはできます)")
    else:
        g = gpus[0]
        small = g["mem_mib"] < MIN_GPU_MIB
        text = f"{g['name']} ({g['mem_mib']} MiB)、ドライバ {g['driver']}" + (f"、CUDA {cuda[0]}.{cuda[1]}" if cuda else "")
        say("注意" if small else "OK", "GPU", text + (f"  メモリが {MIN_GPU_MIB} MiB 未満です。他のジョブと共有すると足りない可能性があります" if small else ""))
        if len(gpus) > 1:
            info(f"GPU が {len(gpus)} 枚あります。サーバは、1 枚目 (cuda:0) を使います。別の 1 枚を使うなら、起動時に CUDA_VISIBLE_DEVICES を指定してください")
    return ctx


def step_venv(r: Runner, args, ctx) -> bool:
    venv = Path(args.venv).resolve()
    vpy = venv / "bin" / "python"
    if vpy.exists() and r.run([str(vpy), "-c", "import pip"], probe=True)[0] == 0:
        say("OK", "仮想環境", f"{venv} (すでにあります)")
        return True
    if not ctx.get("python"):
        say("NG", "仮想環境", "使える Python がありません")
        return False
    existed = venv.exists()
    py = ctx["python"]
    if r.run([py, "-m", "venv", str(venv)])[0] == 0 and (r.dry or r.run([str(vpy), "-c", "import pip"], probe=True)[0] == 0):
        say("OK", "仮想環境", f"{venv} を作りました")
        return True
    # venv が使えない環境 (python3-venv が入っていない): pip なしで作り、get-pip.py で pip を入れる
    info("python3 -m venv が失敗しました (python3-venv が入っていない環境です)。get-pip.py を使う方法に切り替えます")
    if not existed and venv.exists() and not r.dry:
        shutil.rmtree(venv)                                    # 今回、作りかけたものだけを消す (すでにあったフォルダは、消さない)
    if r.run([py, "-m", "venv", "--without-pip", str(venv)])[0] != 0:
        say("NG", "仮想環境", "作成できませんでした。管理者権限があれば、sudo apt install python3.10-venv を実行してください")
        return False
    WORK.mkdir(parents=True, exist_ok=True)
    getpip = WORK / "get-pip.py"
    if not r.dry:
        try:
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", str(getpip))
        except Exception as e:  # noqa: BLE001
            say("NG", "仮想環境", f"get-pip.py を取得できませんでした ({type(e).__name__})。ネットワークを確認するか、sudo apt install python3.10-venv を実行してください")
            return False
    rc = r.run([str(vpy), str(getpip)], log=WORK / "setup-pip.log")[0]
    if not r.dry and getpip.exists():
        getpip.unlink()
    if rc != 0:
        say("NG", "仮想環境", f"pip を入れられませんでした。ログ: {WORK / 'setup-pip.log'}")
        return False
    say("OK", "仮想環境", f"{venv} を作りました (get-pip.py で pip を入れました)")
    return True


def step_install(r: Runner, args, ctx) -> bool:
    venv = Path(args.venv).resolve()
    vpy = venv / "bin" / "python"
    log = WORK / "setup-pip.log"
    tmp = WORK / "tmp"
    if not r.dry:
        tmp.mkdir(parents=True, exist_ok=True)                 # pip の一時領域は、空きの大きい場所に置く (/tmp が小さい機械がある)
    env = dict(os.environ, TMPDIR=str(tmp), PIP_DISABLE_PIP_VERSION_CHECK="1")
    tag, index, notes, err = choose_torch_build(ctx.get("cuda"), (ctx["gpus"][0]["cc"] if ctx.get("gpus") else None), args.cuda)
    if err and (ctx.get("gpus") or args.cuda != "auto"):
        say("NG", "PyTorch の版", err)
        return False
    if err:                                                   # GPU がなく、--allow-no-gpu: 既定の版を入れる
        tag, index, notes = "cu128", None, ["GPU がないため、既定の PyTorch (CUDA 12.8 版) を入れます"]
    say("OK", "PyTorch の版", f"{tag}" + (f" ({index})" if index else " (PyPI の既定)"))
    for n in notes:
        info(n)
    info(f"インストールします (数 GB のダウンロードで、10〜30 分かかります)。経過は {log} に書きます")
    if index:
        rc = r.run([str(vpy), "-m", "pip", "install", "--no-cache-dir", TORCH_PIN, AUDIO_PIN, "--index-url", index], env=env, log=log)[0]
        if rc != 0:
            say("NG", "PyTorch のインストール", f"失敗しました。ログの末尾: tail -n 30 {log}")
            return False
    rc = r.run([str(vpy), "-m", "pip", "install", "--no-cache-dir", "-r", str(REPO / "requirements.txt")], env=env, log=log)[0]
    if rc != 0:
        say("NG", "パッケージのインストール", f"失敗しました。ログの末尾: tail -n 30 {log} (「No space left on device」なら、空き容量を確保してください)")
        return False
    if r.dry:
        say("OK", "パッケージのインストール", "(dry-run)")
        return True
    rc, out = r.run([str(vpy), "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"], probe=True)
    say("OK", "パッケージのインストール", f"torch {out.strip()}")
    if ctx.get("gpus") and not out.strip().endswith("True"):
        say("NG", "PyTorch が GPU を認識しません", "ドライバと、入れた CUDA の版が、合っていない可能性があります。--cuda cu126 (または cu128、cu129) を指定して、やり直してください。nvidia-smi の表示も確認してください")
        return False
    return True


def find_mask_root(args):
    cands = [args.mask_root] if args.mask_root else [os.environ.get("ANTICLONING_MASK_ROOT"), str(REPO.parent / "AntiCloning_mask"), str(REPO.parent.parent / "AntiCloning_mask"), str(Path.home() / "AntiCloning_mask")]
    for c in filter(None, cands):
        if (Path(c) / "anticloning_mask" / "optimize.py").is_file():
            return Path(c).resolve()
    return None


def step_mask_root(args):
    root = find_mask_root(args)
    if root:
        say("OK", "研究のリポジトリ", str(root))
    elif args.mask_root:
        say("NG", "研究のリポジトリ", f"{args.mask_root} に anticloning_mask/optimize.py がありません")
    else:
        say("NG", "研究のリポジトリ", "見つかりません。--mask-root で、anticloning_mask/ を含むフォルダを指定してください (研究のコードだけ、約 0.5 MB を、コピーすれば足ります)")
    return root


def step_weights(r: Runner, args, hf_home: Path, hub_dir: Path) -> bool:
    cmd = [str(REPO / "scripts" / "copy_weights.sh"), "--to-hf", str(hf_home), "--to-hub", str(hub_dir)]
    if args.skip_weights:
        say("注意", "重み", "--skip-weights のため、確認しません")
        return True
    rc, out = r.run(cmd + ["--verify-only"], probe=True)
    if rc == 0:
        say("OK", "重み", f"すでにそろっています (SHA-256 が一致)。{hf_home}、{hub_dir}")
        return True
    if not (args.from_hf and args.from_hub):
        say("NG", "重み", "まだありません。--from-hf と --from-hub で、コピー元 (パス、または ホスト:パス) を指定してください (README の「重みを置く」)")
        return False
    rc, out = r.run(cmd + ["--from-hf", args.from_hf, "--from-hub", args.from_hub], capture=False)
    say("OK" if rc == 0 else "NG", "重みのコピー", ("(dry-run) コピーして、SHA-256 を照合します" if r.dry else "SHA-256 が一致しました") if rc == 0 else "失敗しました (上の表示を確認してください)")
    return rc == 0


def step_env(args, mask_root: Path, hf_home: Path, hub_dir: Path, dry: bool) -> bool:
    env_file = REPO / "server" / "env.local"
    want = {"ANTICLONING_MASK_ROOT": str(mask_root), "PYTHON": str(Path(args.venv).resolve() / "bin" / "python"), "HF_HOME": str(hf_home), "TORCH_HUB_DIR": str(hub_dir)}
    text = render_env(mask_root, Path(want["PYTHON"]), hf_home, hub_dir)
    if env_file.exists():
        have = parse_env(env_file.read_text(encoding="utf-8"))
        diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        if not diff:
            say("OK", "server/env.local", "すでにあります (内容が一致)")
            return True
        if not args.force_env:
            say("注意", "server/env.local", "すでにあり、内容が違うため、上書きしません。違う項目: " + "、".join(diff) + " (上書きするには --force-env。元のファイルは、.bak に残します)")
            return True
        if not dry:
            shutil.copy2(env_file, env_file.with_name(f"env.local.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
    if dry:
        info(f"(dry-run) {env_file} に、次を書きます:\n" + "\n".join("        " + ln for ln in text.splitlines()))
    else:
        env_file.write_text(text, encoding="utf-8")
    say("OK", "server/env.local", "作りました")
    return True


def step_smoke(r: Runner, args) -> bool:
    if not args.smoke:
        say("注意", "通しの点検", "--smoke を付けると、敵対的音声の生成と、日本語・英語の合成を、実際に 1 回ずつ通します (約 1 分)")
        return True
    info("通しの点検を実行します (約 1 分)")
    rc, _ = r.run([str(REPO / "scripts" / "check_env.sh"), "--smoke"], capture=False)
    say("OK" if rc == 0 else "NG", "通しの点検 (check_env.sh --smoke)", "NG が 0 でした" if rc == 0 else "NG があります (上の表示を確認してください)")
    return rc == 0


def step_unit(args) -> None:
    template = (REPO / "deploy" / "voice-demo.service").read_text(encoding="utf-8")
    out = WORK / "voice-demo.service"
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_text(render_unit(template, getpass.getuser(), REPO, args.port), encoding="utf-8")
    say("OK", "systemd の設定ファイル", f"{out} (User={getpass.getuser()}、ポート {args.port})")


def step_start(r: Runner, args) -> bool:
    rc, _ = r.run([str(REPO / "scripts" / "start.sh"), str(args.port)], capture=False)
    say("OK" if rc == 0 else "NG", "サーバの起動", f"http://127.0.0.1:{args.port}/" if rc == 0 else "起動できませんでした (server/work/serve.log を確認してください)")
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="研究室のサーバへ、このデモを置く作業を、自動で行う (詳しくは、このファイルの先頭)")
    ap.add_argument("--venv", default=str(REPO / "venv"), help="仮想環境の場所 (既定: リポジトリの venv)")
    ap.add_argument("--python", help="仮想環境を作る Python (既定: python3.10 など、自動で選ぶ)")
    ap.add_argument("--mask-root", help="研究のリポジトリ (anticloning_mask/ を含むフォルダ)。省略時は、いくつかの場所を探す")
    ap.add_argument("--weights-dir", help="重みの置き場 (既定: リポジトリの 1 つ上の weights/。中に hf/ と torch_hub/ を作る)")
    ap.add_argument("--from-hf", help="HF_HOME のコピー元 (パス、または ホスト:パス)")
    ap.add_argument("--from-hub", help="TORCH_HUB_DIR のコピー元 (パス、または ホスト:パス)")
    ap.add_argument("--cuda", default="auto", choices=["auto", "cu126", "cu128", "cu129"], help="PyTorch の CUDA の版 (既定: ドライバから自動で選ぶ)")
    ap.add_argument("--allow-no-gpu", action="store_true", help="GPU がなくても続ける (Step 2 は動かない。試験用)")
    ap.add_argument("--skip-install", action="store_true", help="仮想環境と、パッケージのインストールを飛ばす")
    ap.add_argument("--skip-weights", action="store_true", help="重みの確認とコピーを飛ばす")
    ap.add_argument("--force-env", action="store_true", help="server/env.local が違う内容でも、上書きする (元のファイルは、.bak に残す)")
    ap.add_argument("--smoke", action="store_true", help="最後に、通しの点検 (check_env.sh --smoke) を実行する")
    ap.add_argument("--start", action="store_true", help="最後に、サーバを起動する (scripts/start.sh)")
    ap.add_argument("--port", type=int, default=8000, help="サーバのポート (既定 8000)")
    ap.add_argument("--dry-run", action="store_true", help="実行する内容だけを表示する")
    args = ap.parse_args()

    r = Runner(args.dry_run)
    weights = Path(args.weights_dir).resolve() if args.weights_dir else REPO.parent / "weights"
    hf_home, hub_dir = weights / "hf", weights / "torch_hub"
    print(f"== 研究室のサーバへの構築{' (dry-run)' if args.dry_run else ''}: {REPO}\n", flush=True)

    ctx = step_preflight(r, args)
    fatal_pre = [x for x in results if x[0] == "NG"]
    if fatal_pre and not args.skip_install:
        print("\n事前の点検で、NG があるため、ここで止めます。上の NG を解消して、もう一度実行してください。")
        return 1
    if not args.skip_install:
        if not step_venv(r, args, ctx) or not step_install(r, args, ctx):
            return finish(args)
    else:
        say("注意", "仮想環境とパッケージ", "--skip-install のため、飛ばします")
    mask_root = step_mask_root(args)
    weights_ok = step_weights(r, args, hf_home, hub_dir)
    if mask_root:
        step_env(args, mask_root, hf_home, hub_dir, args.dry_run)
    else:
        say("NG", "server/env.local", "研究のリポジトリが見つからないため、作れません")
    clean = not any(s == "NG" for s, _, _ in results)
    if args.dry_run:
        if args.smoke:
            say("OK" if clean else "注意", "通しの点検", "(dry-run) 実行します" if clean else "NG があるため、実行しません")
        say("OK", "systemd の設定ファイル", f"(dry-run) {WORK / 'voice-demo.service'} を作ります")
        if args.start:
            say("OK" if clean else "注意", "サーバの起動", f"(dry-run) scripts/start.sh {args.port} を実行します" if clean else "NG があるため、実行しません")
    else:
        if clean:
            step_smoke(r, args)
        elif args.smoke:
            say("注意", "通しの点検", "NG があるため、実行しません")
        step_unit(args)
        if args.start and not any(s == "NG" for s, _, _ in results):
            step_start(r, args)
    return finish(args)


def finish(args) -> int:
    ng = [x for x in results if x[0] == "NG"]
    note = [x for x in results if x[0] == "注意"]
    print(f"\n== 結果: {len(results)} 工程、NG {len(ng)}、注意 {len(note)}")
    for s, n, d in ng:
        print(f"   NG: {n}  {d}")
    if ng:
        print("\n未完了です。上の NG を解消して、同じコマンドを、もう一度実行してください (済んでいる工程は、飛ばされます)。")
        return 1
    if args.dry_run:
        print("\n(dry-run のため、何も実行していません)")
        return 0
    print(f"""
完了です。次の作業:
  1. 起動:  scripts/start.sh {args.port}          (止める: scripts/stop.sh、状態: scripts/status.sh)
     常駐:  sudo cp {WORK / 'voice-demo.service'} /etc/systemd/system/ && sudo systemctl enable --now voice-demo
  2. HTTPS で公開 (マイクは HTTPS でしか使えません。README の「公開する」の手順 8):
     Cloudflare Tunnel (scripts/tunnel.sh) か、研究室の nginx (deploy/nginx.example.conf)
  3. 公開の前に、プライバシーの文面と、アクセスの制限を、確認してください (README の「公開の前に確認すること」)""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
