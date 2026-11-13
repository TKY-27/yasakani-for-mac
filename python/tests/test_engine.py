# SPDX-License-Identifier: Apache-2.0
import threading
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import support
from yasakani._native import YasakaniError
from yasakani.engine import Engine, Sampling
from yasakani.runtime import EOS_IDS, IM_START

EOS = EOS_IDS[0]


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.e = support.open_engine(Path(self.tmp.name), self.stub)

    def tearDown(self):
        self.e.close()
        self.tmp.cleanup()

    def test_basic_generate(self):
        self.stub.push([11, 12, 13, EOS])
        r = self.e.generate([5, 6, 7 + 1], 100)
        self.assertEqual(r.ids, [11, 12, 13, EOS])
        self.assertEqual(r.finish, "stop")
        self.assertEqual((r.prompt_tokens, r.cached_tokens, r.snap_mode), (3, 0, 0))
        self.assertGreater(r.passes, 0)
        self.assertEqual(self.e.position(), 3 + 4 - 1)          # prompt + generated ids, the last one not yet processed

    def test_length_limit_and_first_id(self):
        self.stub.push(list(range(100, 200)))
        r = self.e.generate([5, 6], 10)
        self.assertEqual(r.ids, list(range(100, 110)))          # the first id comes from the prompt pass: 10 ids in all
        self.assertEqual(r.finish, "length")

    def test_max_new_one(self):
        self.stub.push([21, 22, 23])
        r = self.e.generate([5, 6], 1)
        self.assertEqual(r.ids, [21])

    def test_streaming_progress(self):
        self.stub.push(list(range(100, 160)) + [EOS])
        self.stub.delay_us(3000)
        seen = []
        r = self.e.generate([5, 6], 500, on_tokens=lambda ids: seen.append(len(ids)))
        self.assertGreater(len(seen), 5)                        # progress was reported while the engine ran
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], len(r.ids))
        self.assertEqual(r.ids[-1], EOS)

    def test_cancel_from_another_thread(self):
        self.stub.push([100, 101, 102, 103], loop_from=0)       # never ends by itself
        self.stub.delay_us(1000)
        threading.Timer(0.05, self.e.cancel).start()
        t0 = time.time()
        r = self.e.generate([5, 6], 1_000_000 - 100)
        self.assertLess(time.time() - t0, 5)
        self.assertEqual(r.finish, "cancelled")
        self.assertGreater(len(r.ids), 3)
        self.stub.push([31, 32, EOS])                           # the engine is usable afterwards
        self.assertEqual(self.e.generate([5, 6], 50).ids, [31, 32, EOS])

    def test_consumer_error_stops_engine_and_is_raised(self):
        self.stub.push([100, 101, 102, 103], loop_from=0)
        self.stub.delay_us(1000)

        def boom(ids):
            raise RuntimeError("consumer failed")
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            self.e.generate([5, 6], 500_000, on_tokens=boom)
        self.stub.push([41, EOS])
        self.assertEqual(self.e.generate([5, 6], 50).ids, [41, EOS])

    def test_error_text_comes_from_the_worker_thread(self):
        with self.assertRaisesRegex(YasakaniError, "injected failure"):
            self.e.generate([7, 6], 10)

    def test_prompt_too_long(self):
        with self.assertRaises(ValueError):
            self.e.generate(list(range(1, 4100)), 10)
        with self.assertRaises(ValueError):
            self.e.generate([], 10)

    def test_ignore_stop_ids(self):
        self.stub.push([1, 2, EOS, 3, 4, 5])
        r = self.e.generate([9, 9], 5, stop_ids=None)
        self.assertEqual(r.ids, [1, 2, EOS, 3, 4])
        self.assertEqual(r.finish, "length")
        cfg = self.stub.last_cfg()
        self.assertEqual((cfg[20], cfg[21]), (0x7FFFFFFF, 0x7FFFFFFF))

    def test_stop_ids_reach_the_engine(self):
        self.stub.push([1, 2, 3])
        self.e.generate([9, 9], 5, stop_ids=(EOS, 248044))
        cfg = self.stub.last_cfg()
        self.assertEqual(len(cfg), 44)
        self.assertEqual((cfg[20], cfg[21], cfg[41], cfg[42], cfg[43]), (EOS, 248044, 1, 1, IM_START))

    def test_state_settings_are_not_per_request(self):
        with self.assertRaises(ValueError):
            self.e.generate([1, 2], 5, spec={"snap": 0})
        with self.assertRaises(ValueError):
            self.e.generate([1, 2], 5, spec={"nope": 0})
        self.stub.push([1, EOS])
        self.e.generate([1, 2], 5, spec={"gamma": 5})
        self.assertEqual(self.stub.last_cfg()[0], 5)

    def test_prompt_reuse_modes(self):
        marker = [IM_START, 50, 51]
        self.stub.push([60, 61, 62, EOS])
        p1 = [10, 11] + marker
        r1 = self.e.generate(p1, 50)
        self.assertEqual((r1.snap_mode, r1.cached_tokens), (0, 0))
        # next turn extends the live state (prompt + answer without its last id)
        p2 = p1 + r1.ids[:-1] + [70, 71]
        self.stub.push([80, EOS])
        r2 = self.e.generate(p2, 50)
        self.assertEqual(r2.snap_mode, 1)
        self.assertEqual(r2.cached_tokens, len(p1) + len(r1.ids) - 1)
        # a regeneration shares the prompt up to the snapshot: the snapshot is restored
        p3 = [10, 11, IM_START, 99, 98]
        self.stub.push([81, EOS])
        r3 = self.e.generate(p3, 50)
        self.assertEqual(r3.snap_mode, 2)

    def test_sampling_options(self):
        s = self.e.set_sampling(Sampling(0.6, 0.9, 40, 0.05, 123))
        self.assertEqual(s.top_k, 20)                            # capped at 20
        got = [self.stub.opt(self.e, k) for k in (11, 12, 13, 10, 16, 29)]
        self.assertEqual(got, [900, 20, 123, 600, 3, 50])
        self.e.set_sampling(Sampling(0.0))
        self.assertEqual(self.stub.opt(self.e, 10), 0)

    def test_state_hash_and_reset(self):
        self.stub.push([1, 2, EOS])
        self.e.generate([5, 6], 10)
        h1 = self.e.state_hash()
        self.e.reset()
        self.assertEqual(self.e.position(), 0)
        self.assertNotEqual(self.e.state_hash().tolist(), h1.tolist())

    def test_open_failure_hint_and_bad_drafter(self):
        with self.assertRaisesRegex(YasakaniError, "GPU memory"):
            support.open_engine(Path(self.tmp.name), self.stub, name="fail")
        root = Path(self.tmp.name) / "bad"
        support.make_pack(Path(self.tmp.name), "bad")
        (root / "bdraft2_k8.yskn").rename(root / "bdraft_bad.yskn")
        (root / "yasakani.json").write_text('{"files": {"draft": "bdraft_bad.yskn"}}')
        from yasakani.runtime import ModelPack
        with self.assertRaisesRegex(YasakaniError, "bad container"):
            Engine(ModelPack.load(root), max_ctx=4096, lib=self.stub.lib)

    def test_plain_paths_do_not_touch_the_locale(self):
        import sys
        from yasakani.engine import _narrow_paths_as_utf8
        self.assertFalse(_narrow_paths_as_utf8(["C:/models/qwen/x.yskn", "D:\\a b\\c.bin"]))
        if sys.platform != "win32":                                  # the switch exists for Windows only
            self.assertFalse(_narrow_paths_as_utf8(["/tmp/\u30e2\u30c7\u30eb/x.yskn"]))

    def test_closed_engine(self):
        self.e.close()
        with self.assertRaises(YasakaniError):
            self.e.reset()


if __name__ == "__main__":
    unittest.main()
