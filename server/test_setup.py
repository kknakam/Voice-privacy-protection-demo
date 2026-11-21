"""研究室のサーバの構築を自動化するプログラム (server/setup_server.py) の、判断の部分の試験。GPU も、ネットワークも、pip も使わない。

  python -m unittest server.test_setup -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import setup_server as S  # noqa: E402

NVIDIA_SMI_HEADER = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.181                Driver Version: 570.181        CUDA Version: 12.8     |
+-----------------------------------------+------------------------+----------------------+
"""


class ParseNvidiaSmi(unittest.TestCase):
    def test_cuda_version(self):
        self.assertEqual(S.parse_cuda_version(NVIDIA_SMI_HEADER), (12, 8))
        self.assertEqual(S.parse_cuda_version("CUDA Version: 11.4"), (11, 4))
        self.assertIsNone(S.parse_cuda_version("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."))
        self.assertIsNone(S.parse_cuda_version(""))

    def test_gpu_query(self):
        one = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 570.181, 97887, 12.0\n"
        g = S.parse_gpu_query(one)
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0]["driver"], g[0]["mem_mib"], g[0]["cc"]), ("570.181", 97887, 12.0))
        self.assertIn("RTX PRO 6000", g[0]["name"])
        two = "NVIDIA A100-SXM4-40GB, 535.104.05, 40960, 8.0\nNVIDIA A100-SXM4-40GB, 535.104.05, 40960, 8.0\n"
        self.assertEqual(len(S.parse_gpu_query(two)), 2)

    def test_old_driver_without_compute_cap(self):
        g = S.parse_gpu_query("Tesla V100-PCIE-32GB, 470.82.01, 32510\n")
        self.assertEqual((g[0]["mem_mib"], g[0]["cc"]), (32510, None))

    def test_failure_output_gives_no_gpu(self):
        self.assertEqual(S.parse_gpu_query("Failed to initialize NVML: Driver/library version mismatch\n"), [])
        self.assertEqual(S.parse_gpu_query(""), [])


class ChooseTorchBuild(unittest.TestCase):
    def test_new_driver_uses_default_pypi_build(self):
        for v in [(12, 8), (12, 9), (13, 0)]:
            with self.subTest(v=v):
                tag, index, notes, err = S.choose_torch_build(v)
                self.assertEqual((tag, index, err), ("cu128", None, None))

    def test_driver_with_cuda_12_6_or_12_7_uses_cu126_index(self):
        for v in [(12, 6), (12, 7)]:
            with self.subTest(v=v):
                tag, index, notes, err = S.choose_torch_build(v)
                self.assertEqual((tag, err), ("cu126", None))
                self.assertTrue(index.endswith("/whl/cu126"))
                self.assertEqual(notes, [])

    def test_older_cuda_12_uses_cu126_with_a_note(self):
        tag, index, notes, err = S.choose_torch_build((12, 2))
        self.assertEqual((tag, err), ("cu126", None))
        self.assertTrue(notes)

    def test_too_old_or_unknown_driver_is_an_error(self):
        self.assertIsNotNone(S.choose_torch_build((11, 8))[3])
        self.assertIsNotNone(S.choose_torch_build(None)[3])

    def test_blackwell_needs_cuda_12_8(self):
        self.assertIsNotNone(S.choose_torch_build((12, 6), compute_cap=12.0)[3])
        self.assertEqual(S.choose_torch_build((12, 8), compute_cap=12.0)[:3][0], "cu128")
        self.assertIsNone(S.choose_torch_build((12, 6), compute_cap=9.0)[3])         # 古い世代は、cu126 でよい

    def test_override(self):
        tag, index, notes, err = S.choose_torch_build(None, override="cu129")
        self.assertEqual((tag, err), ("cu129", None))
        self.assertTrue(index.endswith("/whl/cu129"))
        self.assertIsNone(S.choose_torch_build((12, 2), override="cu128")[1])


class PickPython(unittest.TestCase):
    def test_prefers_3_10(self):
        self.assertEqual(S.pick_python({"python3.10": (3, 10), "python3": (3, 12)}), ("python3.10", (3, 10)))
        self.assertEqual(S.pick_python({"python3": (3, 10)}), ("python3", (3, 10)))
        self.assertEqual(S.pick_python({"python3.11": (3, 11), "python3.9": (3, 9)})[0], "python3.11")

    def test_rejects_unsupported(self):
        self.assertEqual(S.pick_python({"python3": (3, 13)}), (None, None))
        self.assertEqual(S.pick_python({"python3": (3, 8)}), (None, None))
        self.assertEqual(S.pick_python({}), (None, None))


class EnvFile(unittest.TestCase):
    def test_roundtrip(self):
        text = S.render_env(Path("/data/mask root"), Path("/data/venv/bin/python"), Path("/data/w/hf"), Path("/data/w/hub"))
        got = S.parse_env(text)
        self.assertEqual(got, {"ANTICLONING_MASK_ROOT": "/data/mask root", "PYTHON": "/data/venv/bin/python", "HF_HOME": "/data/w/hf", "TORCH_HUB_DIR": "/data/w/hub"})

    def test_parse_ignores_comments_and_quotes(self):
        got = S.parse_env("# 説明\nA=1\n\n  B = 'x y' \n#C=3\nbroken line\n")
        self.assertEqual(got, {"A": "1", "B": "x y"})

    def test_unit_file(self):
        template = (S.REPO / "deploy" / "voice-demo.service").read_text(encoding="utf-8")
        out = S.render_unit(template, "alice", Path("/data/demo_site"), 8010)
        self.assertIn("User=alice", out)
        self.assertIn("WorkingDirectory=/data/demo_site", out)
        self.assertIn("ExecStart=/data/demo_site/scripts/serve.sh 8010", out)
        self.assertNotIn("YOUR_USER", out)
        self.assertNotIn("/path/to/", out)


class FakeRunner(S.Runner):
    """venv が使えない環境 (python3-venv が入っていない) を、模擬する。実際には、何も実行しない"""

    def __init__(self, tmp: Path, venv: Path):
        super().__init__(dry=False)
        self.tmp, self.venv, self.calls = tmp, venv, []

    def run(self, cmd, env=None, capture=True, log=None, cwd=None, probe=False):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[1:3] == ["-m", "venv"] and "--without-pip" not in cmd:
            (self.venv / "bin").mkdir(parents=True, exist_ok=True)           # 作りかけて、失敗する (ensurepip がない)
            return 1, "ensurepip is not available"
        if cmd[1:3] == ["-m", "venv"] and "--without-pip" in cmd:
            (self.venv / "bin").mkdir(parents=True, exist_ok=True)
            (self.venv / "bin" / "python").write_text("#!/bin/sh\n")
            return 0, ""
        if cmd[0].endswith("/bin/python") and cmd[1:3] == ["-c", "import pip"]:
            return (0, "") if (self.venv / "lib_pip_installed").exists() else (1, "")
        if cmd[0].endswith("/bin/python") and cmd[1].endswith("get-pip.py"):
            (self.venv / "lib_pip_installed").write_text("")
            return 0, ""
        return 0, ""


class VenvFallback(unittest.TestCase):
    def args(self, venv: Path):
        return mock.Mock(venv=str(venv))

    def test_falls_back_to_get_pip_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            venv = tmp / "venv"
            r = FakeRunner(tmp, venv)

            def fake_urlretrieve(url, dest):
                self.assertEqual(url, "https://bootstrap.pypa.io/get-pip.py")
                Path(dest).write_text("# get-pip")

            S.results.clear()
            with mock.patch.object(S, "WORK", tmp / "work"), mock.patch.object(S.urllib.request, "urlretrieve", fake_urlretrieve):
                ok = S.step_venv(r, self.args(venv), {"python": "python3"})
            self.assertTrue(ok)
            self.assertEqual(S.results[-1][0], "OK")
            self.assertIn("get-pip", S.results[-1][2])
            flat = [" ".join(c) for c in r.calls]
            self.assertTrue(any("-m venv " in c and "--without-pip" not in c for c in flat))     # まず、普通に作ろうとする
            self.assertTrue(any("--without-pip" in c for c in flat))                              # 失敗したら、pip なしで作る
            self.assertFalse((tmp / "work" / "get-pip.py").exists())                              # 使い終わった get-pip.py は、消す

    def test_does_not_delete_a_preexisting_directory(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            venv = tmp / "venv"
            (venv / "keep").mkdir(parents=True)
            (venv / "keep" / "file.txt").write_text("x")
            r = FakeRunner(tmp, venv)
            S.results.clear()
            with mock.patch.object(S, "WORK", tmp / "work"), mock.patch.object(S.urllib.request, "urlretrieve", lambda u, dest: Path(dest).write_text("")):
                S.step_venv(r, self.args(venv), {"python": "python3"})
            self.assertTrue((venv / "keep" / "file.txt").exists())                                # すでにあったものは、消さない

    def test_existing_working_venv_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            venv = tmp / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("")
            (venv / "lib_pip_installed").write_text("")
            r = FakeRunner(tmp, venv)
            S.results.clear()
            self.assertTrue(S.step_venv(r, self.args(venv), {"python": "python3"}))
            self.assertIn("すでにあります", S.results[-1][2])
            self.assertEqual([c for c in r.calls if "venv" in c[1:3]], [])                        # 作り直さない


if __name__ == "__main__":
    unittest.main()
