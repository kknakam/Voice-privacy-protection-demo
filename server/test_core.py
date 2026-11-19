"""server/core.py の、GPU を使わない部分の試験。

  python -m unittest server.test_core -v     (リポジトリの直下で実行。soundfile と numpy が要る)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from server import core

SITE_TARGETS = Path(__file__).resolve().parent.parent / "site" / "assets" / "targets"


class Pool(unittest.TestCase):
    def test_pool_matches_reference_audio(self):
        ids, emb = core.load_pool()
        self.assertEqual(emb.shape, (len(ids), 512))
        self.assertEqual(len(set(ids)), len(ids))
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
        # 候補の話者ごとに、参照音声がある。逆に、候補にない参照音声は置かない
        self.assertEqual(sorted(p.stem for p in SITE_TARGETS.glob("*.wav")), sorted(ids))
        for i in ids:
            info = sf.info(str(SITE_TARGETS / f"{i}.wav"))
            self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))
            self.assertTrue(3.0 <= info.duration <= 6.5, (i, info.duration))   # 3 秒以上の発話から作る。6 秒を超える分は切る

    def test_farthest_target_is_lowest_cosine(self):
        ids, emb = core.load_pool()
        rng = np.random.default_rng(0)
        for _ in range(20):
            e = rng.normal(size=512).astype(np.float32)
            tid, temb, cos = core.farthest_target(e, ids, emb)
            sims = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ (e / np.linalg.norm(e))
            self.assertEqual(tid, ids[int(np.argmin(sims))])
            self.assertAlmostEqual(cos, float(sims.min()), places=5)
            np.testing.assert_array_equal(temb, emb[ids.index(tid)])

    def test_pool_member_is_never_its_own_farthest(self):
        ids, emb = core.load_pool()
        for k in range(len(ids)):
            self.assertNotEqual(core.farthest_target(emb[k], ids, emb)[0], ids[k])


class ReadWav(unittest.TestCase):
    def write(self, name, wave, sr=16000, subtype="PCM_16"):
        path = Path(self.tmp.name) / name
        sf.write(str(path), wave, sr, subtype=subtype)
        return str(path)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_accepts_16k_mono_pcm16(self):
        w = core.read_wav(self.write("ok.wav", np.zeros(16000, np.float32)))
        self.assertEqual((w.dtype, w.shape), (np.float32, (16000,)))

    def test_truncates_to_max_sec(self):
        w = core.read_wav(self.write("long.wav", np.zeros(16000 * 12, np.float32)))
        self.assertEqual(len(w), int(core.SAMPLE_RATE * core.MAX_SEC))

    def test_rejects_other_formats(self):
        for name, wave, sr, sub in [("sr.wav", np.zeros(48000, np.float32), 48000, "PCM_16"),
                                    ("stereo.wav", np.zeros((16000, 2), np.float32), 16000, "PCM_16"),
                                    ("float.wav", np.zeros(16000, np.float32), 16000, "FLOAT"),
                                    ("short.wav", np.zeros(1000, np.float32), 16000, "PCM_16")]:
            with self.assertRaises(ValueError, msg=name):
                core.read_wav(self.write(name, wave, sr, sub))


class Quantize(unittest.TestCase):
    def test_roundtrip_error_is_half_a_step(self):
        a = np.random.default_rng(1).uniform(-100, 35, size=(50, 40)).astype(np.float32)
        q = core._quantize(a, -100.0, 35.0)
        import base64
        back = np.frombuffer(base64.b64decode(q["q"]), np.uint8).astype(np.float32).reshape(a.shape) * (q["hi"] - q["lo"]) / 255 + q["lo"]
        self.assertLessEqual(float(np.abs(back - a).max()), (q["hi"] - q["lo"]) / 255 / 2 + 1e-4)


if __name__ == "__main__":
    unittest.main()
