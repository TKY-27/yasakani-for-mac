# SPDX-License-Identifier: Apache-2.0
import random
import unittest

import numpy as np

from tests import support
from yasakani.runtime import EOS_IDS
from yasakani.textio import ChatFormatter, OutputStream, StreamDecoder

HAVE_TOK = support.tokenizer_dir() is not None

TEXTS = [
    "Hello, world! This is plain ASCII text.\n\tIndented line with a tab.\n",
    "日本語のテキストです。漢字、ひらがな、カタカナ、そして記号「」（）も含みます。",
    "Emoji: 😀 👍🏽 👨‍👩‍👧‍👦 🇯🇵 and a rare one 𠮷 plus accents é ñ ü.",
    "def f(x):\n    return [i**2 for i in range(x) if i % 2]  # code\n\n\n    trailing spaces   \n",
    "Mixed 日本語 and English, 123456789 digits, symbols <>&\"' and a very long run: " + "あ" * 40,
    "​  zero width and nbsp 　 ideographic space",
]


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class TextioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def test_template_strings(self):
        f = self.fmt
        u = [{"role": "user", "content": "hi"}]
        t = f.render(u, enable_thinking=False)
        self.assertEqual(t, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        t = f.render(u)                                             # thinking on: the effort instruction opens the conversation
        self.assertTrue(t.startswith("<|im_start|>system\nReasoning effort is set to xhigh."), t[:80])
        self.assertTrue(t.endswith("<|im_start|>assistant\n<think>\n"))
        t = f.render(u, reasoning_effort="medium")                  # medium has no extra sentence
        self.assertTrue(t.startswith("<|im_start|>user\nhi<|im_end|>"))
        t = f.render([{"role": "system", "content": "Be brief."}] + u, reasoning_effort="low")
        self.assertIn("Reasoning effort is set to low.", t)
        self.assertIn("\n\nBe brief.<|im_end|>\n", t)
        with self.assertRaises(Exception):
            f.render(u, reasoning_effort="ultra")
        with self.assertRaises(Exception):
            f.render([])
        with self.assertRaises(Exception):
            f.render([{"role": "system", "content": "only a system message"}])

    def test_history_keeps_the_earlier_thinking(self):
        m = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, {"role": "user", "content": "q2"}]
        t = self.fmt.render(m, enable_thinking=False)
        self.assertIn("<|im_start|>assistant\n<think>\nr1\n</think>\n\na1<|im_end|>\n", t)
        m[1].pop("reasoning_content")
        self.assertIn("<|im_start|>assistant\n<think>\n\n</think>\n\na1<|im_end|>\n", self.fmt.render(m, enable_thinking=False))

    def test_prompt_reports_whether_the_thinking_is_open(self):
        u = [{"role": "user", "content": "hi"}]
        ids, open_ = self.fmt.prompt(u)
        self.assertTrue(open_)
        self.assertEqual(int(ids[-2]), self.fmt.think_start)         # ... <think> \n
        ids2, open2 = self.fmt.prompt(u, enable_thinking=False)
        self.assertFalse(open2)
        self.assertEqual(int(ids2[-2]), self.fmt.think_end)

    def test_stream_decoder_matches_full_decode(self):
        rnd = random.Random(1234)
        for text in TEXTS:
            ids = self.fmt.encode(text).tolist()
            full = self.fmt.decode(ids)
            for trial in range(12):
                d = StreamDecoder(self.fmt.tok)
                out, i = [], 0
                while i < len(ids):
                    n = 1 if trial == 0 else rnd.randint(1, 7)
                    piece = d.push(ids[i:i + n])
                    self.assertNotIn("�", piece, (text[:20], trial))
                    out.append(piece)
                    i += n
                out.append(d.flush())
                self.assertEqual("".join(out), full, (text[:20], trial))

    def test_stream_decoder_work_stays_bounded(self):
        ids = self.fmt.encode("あいうえお漢字" * 3000).tolist()
        d = StreamDecoder(self.fmt.tok)
        calls = []
        orig = d._dec
        d._dec = lambda a, b=None: (calls.append((len(d.ids) if b is None else b) - a), orig(a, b))[1]
        for k in range(0, len(ids), 3):
            d.push(ids[k:k + 3])
        self.assertLess(max(calls), 40)                              # the window decoded again never grows with the text

    def test_output_stream_splits_thinking_and_answer(self):
        f = self.fmt
        reasoning = f.encode("Let me think about it.").tolist()
        answer = f.encode("\n\nThe answer is 42.").tolist()
        ids = np.array(reasoning + [f.think_end] + answer + [EOS_IDS[0]], np.int32)
        for chunk in (1, 2, 5, 1000):
            s = OutputStream(f, True)
            ev = []
            for k in range(chunk, len(ids) + chunk, chunk):
                ev += s.feed(ids[:min(k, len(ids))])
            ev += s.feed(ids, final=True)
            r = "".join(t for kind, t in ev if kind == "reasoning")
            a = "".join(t for kind, t in ev if kind == "content")
            self.assertEqual((r, a), ("Let me think about it.", "The answer is 42."), chunk)
            self.assertEqual(s.think_end_at, len(reasoning))
        s = OutputStream(f, False)                                   # thinking disabled: everything is the answer
        ev = s.feed(np.array(answer + [EOS_IDS[1]], np.int32), final=True)
        self.assertEqual("".join(t for k, t in ev if k == "content"), "The answer is 42.")
        self.assertIsNone(s.think_end_at)

    def test_output_stream_gives_one_piece_per_id(self):
        f = self.fmt
        text = "The answer is 42, as everybody knows by now."
        answer = f.encode(text).tolist()
        ids = np.array(answer + [EOS_IDS[0]], np.int32)
        for chunk in (1, 3, 1000):                                   # however the ids arrive: one piece per id
            s = OutputStream(f, False)
            ev = []
            for k in range(chunk, len(ids) + chunk, chunk):
                ev += s.feed(ids[:min(k, len(ids))])
            ev += s.feed(ids, final=True)
            self.assertEqual(len(ev), len(answer), chunk)
            self.assertEqual("".join(t for _, t in ev), text)
        for text in TEXTS:                                           # a character split over several ids: fewer pieces, never a broken one
            ids = f.encode(text).tolist()
            s = OutputStream(f, False)
            ev = s.feed(np.array(ids, np.int32), final=True)
            self.assertLessEqual(len(ev), len(ids))
            self.assertEqual("".join(t for _, t in ev), f.decode(ids))
            self.assertTrue(all("\ufffd" not in t for _, t in ev))


if __name__ == "__main__":
    unittest.main()
