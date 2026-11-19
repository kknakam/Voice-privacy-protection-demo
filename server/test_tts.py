"""音声合成 (server/tts.py) の、モデルを使わない部分の試験。GPU も、重みも要らない。

  python -m unittest server.test_tts -v

正解データ (server/testdata/) は、これまでブラウザで動かして評価した実装の結果。
  normalize_en.json   ブラウザの英語の文章の正規化 (tts.js) の実行結果 66 件
  ja_frontend.json    日本語の音素化 (OpenJTalk の素の版) の、文章と記号列。サイトで評価した音素化と同じ
  split.json          長い文の分割 (開始 100、最大 60) の結果。評価に使った実装の結果
日本語の音素化の試験は、pyopenjtalk が import できるときだけ動く (EXTRA_PYTHONPATH に、pyopenjtalk-plus のある場所を書いてもよい)。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import tts  # noqa: E402

DATA = Path(__file__).resolve().parent / "testdata"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def ids_of(symbols: list[str]) -> list[int]:
    ids = tts.symbols_to_ids(symbols)
    assert ids is not None
    return ids


def symbols_of(ids: list[int]) -> list[str]:
    return [tts.JA_SYMBOLS[i - tts.JA_BASE] for i in ids[:-1]]


class NormalizeEnglish(unittest.TestCase):
    def test_matches_browser_implementation(self):
        cases = load("normalize_en.json")["cases"]
        self.assertGreaterEqual(len(cases), 60)
        for c in cases:
            with self.subTest(input=c["input"][:40]):
                text, err = tts.normalize_en(c["input"])
                self.assertEqual((text, err), (c.get("text"), c.get("error")))

    def test_numbers_and_limits(self):
        self.assertEqual(tts.normalize_en("Hello 12 world")[0], "Hello twelve world.")
        self.assertEqual(tts.normalize_en("the 3rd item costs $5")[0], "the third item costs five dollars.")
        self.assertEqual(tts.normalize_en("a" * 201)[1], "200 文字以内で入力してください。")
        self.assertEqual(tts.normalize_en("日本語")[1][:20], "SpeechT5 は英語専用のため、英語で入力して"[:20])
        self.assertEqual(tts.normalize_en("!!!")[1], "読み上げられる英単語が含まれていません。")

    def test_encode(self):
        v = tts.CONSTANTS["en"]["vocab"]
        ids = tts.encode_en("a b.")
        self.assertEqual(ids[0], v["▁"])
        self.assertEqual(ids[-1], v["</s>"])
        self.assertEqual(ids[2], v["▁"])                      # 空白は ▁
        self.assertEqual(len(ids), 6)


class Language(unittest.TestCase):
    def test_detect(self):
        for text, want in [("こんにちは", "ja"), ("漢字", "ja"), ("ｱｲｳ", "ja"), ("ー", "ja"), ("Hello 世界", "ja"), ("ABC 123 !?", "en"), ("ＡＢＣ", "en"), ("", "en")]:
            with self.subTest(text=text):
                self.assertEqual(tts.detect_lang(text), want)

    def test_js_len_counts_utf16_units(self):
        self.assertEqual(tts.js_len("abc"), 3)
        self.assertEqual(tts.js_len("😀"), 2)                  # JavaScript の length と同じ。画面の文字数の表示と合わせる


class JapaneseSymbols(unittest.TestCase):
    def test_labels_to_symbols(self):
        for i, c in enumerate(load("ja_frontend.json")["labels"]):
            with self.subTest(i=i):
                self.assertEqual(tts.labels_to_symbols(c["labels"]), c["symbols"])

    def test_symbols_to_ids(self):
        ids = ids_of(["^", "a", "$"])
        self.assertEqual(ids, [tts.JA_BASE + tts.JA_SYMBOLS.index(s) for s in ("^", "a", "$")] + [tts.JA_EOS])
        self.assertIsNone(tts.symbols_to_ids(["^", "存在しない記号"]))

    def test_frontend_with_pyopenjtalk(self):
        tts.add_extra_paths()
        try:
            import pyopenjtalk  # noqa: F401
        except ImportError:
            self.skipTest("pyopenjtalk が import できません")
        fe = tts.JaFrontend()
        for c in load("ja_frontend.json")["sentences"]:
            with self.subTest(text=c["text"][:20]):
                self.assertEqual(symbols_of(fe.ids(c["text"])), c["symbols"])
        for bad in ["！！！", "", "   "]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fe.ids(bad)


class SplitIds(unittest.TestCase):
    def test_matches_validated_implementation(self):
        cases = load("split.json")["cases"]
        self.assertTrue(any(c["segments"] for c in cases) and any(not c["segments"] for c in cases))
        for i, c in enumerate(cases):
            with self.subTest(i=i):
                got = [symbols_of(p) for p in tts.split_ids(ids_of(c["symbols"]))]
                self.assertEqual(got, c["segments"] or [c["symbols"]])

    def test_properties(self):
        for c in load("split.json")["cases"]:
            if not c["segments"]:
                continue
            parts = [symbols_of(p) for p in tts.split_ids(ids_of(c["symbols"]))]
            self.assertGreater(len(parts), 1)
            for k, p in enumerate(parts):
                self.assertEqual(p[0], "^")
                self.assertEqual(p[-1], c["symbols"][-1] if k == len(parts) - 1 else "$")
            # 区間をつなぎ直すと (^、$、区間の境目のポーズを除いて)、元の音素と韻律記号が、順に並ぶ
            body = [s for p in parts for s in p if s not in ("^", "$", "?", "_")]
            self.assertEqual(body, [s for s in c["symbols"] if s not in ("^", "$", "?", "_")])

    def test_short_and_pauseless_are_not_split(self):
        short = ids_of(["^"] + ["a"] * 50 + ["$"])
        self.assertEqual(tts.split_ids(short), [short])
        pauseless = ids_of(["^"] + ["a"] * 150 + ["$"])
        self.assertEqual(tts.split_ids(pauseless), [pauseless])


class ServiceWithoutModels(unittest.TestCase):
    def test_normalize(self):
        s = tts.Service(log=lambda m: None)
        self.assertEqual(s.normalize("Hello 3")["text"], "Hello three.")
        self.assertEqual(s.normalize("こんにちは。")["text"], "こんにちは。")
        self.assertIn("60 文字以内", s.normalize("あ" * 61)["error"])
        self.assertEqual(s.normalize("   ")["error"], "文章を入力してください。")      # 空白だけ: 英語として扱われ、同じ文面
        s.state = "ready"                                                              # 読み込みが終わって、日本語が使えない場合
        self.assertIn("用意していません", s.normalize("こんにちは")["error"])
        s.ja = True
        self.assertEqual(s.normalize("こんにちは")["lang"], "ja")

    def test_embedding_validation(self):
        for bad in ([0.0] * 512, [1.0] * 511, [float("nan")] * 512):
            with self.subTest(n=len(bad)), self.assertRaises(ValueError):
                tts._unit(bad)
        v = tts._unit([3.0] * 512)
        self.assertAlmostEqual(float((v * v).sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
