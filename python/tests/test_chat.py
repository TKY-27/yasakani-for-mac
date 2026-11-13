# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import support
from yasakani import loopguard
from yasakani.chat import ChatParams, ChatRunner, RequestError, StopFilter, clean_messages
from yasakani.runtime import EOS_IDS
from yasakani.textio import ChatFormatter

EOS = EOS_IDS[0]
HAVE_TOK = support.tokenizer_dir() is not None


class StopFilterTests(unittest.TestCase):
    def test_cut_and_holdback(self):
        f = StopFilter(["END"])
        out = f.push("hello E") + f.push("N") + f.push("D tail")
        self.assertEqual(out, "hello ")
        self.assertTrue(f.hit)
        self.assertEqual(f.push("more"), "")

    def test_partial_stop_is_released(self):
        f = StopFilter(["END"])
        out = f.push("abc EN") + f.push("x yz") + f.flush()
        self.assertEqual(out, "abc ENx yz")
        self.assertFalse(f.hit)

    def test_no_stops_passthrough(self):
        f = StopFilter([])
        self.assertEqual(f.push("abc"), "abc")

    def test_two_stops_earliest_wins(self):
        f = StopFilter(["bb", "a"])
        self.assertEqual(f.push("xxbba"), "xx")


class CleanMessageTests(unittest.TestCase):
    def test_ok_and_errors(self):
        m = clean_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"role": "assistant", "content": None}])
        self.assertEqual(m[0]["content"], [{"type": "text", "text": "hi"}])
        self.assertEqual(m[1]["content"], "")
        for bad in ([], None, [{"role": "tool", "content": "x"}], [{"role": "user", "content": 5}],
                    [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]):
            with self.assertRaises(RequestError):
                clean_messages(bad)


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.engine = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000)
        self.runner = ChatRunner(self.engine, self.fmt, seed0=100)

    def tearDown(self):
        self.engine.close()
        self.tmp.cleanup()

    def ids(self, text):
        return self.fmt.encode(text).tolist()

    def run_chat(self, **kw):
        ev = []
        kw.setdefault("messages", [{"role": "user", "content": "hi"}])
        u = self.runner.run(ChatParams(**kw), lambda k, t: ev.append((k, t)))
        text = {k: "".join(t for kk, t in ev if kk == k) for k in ("reasoning", "content")}
        return text, u

    def test_reasoning_then_answer(self):
        think = self.ids("Let me think.")
        script = think + [self.fmt.think_end] + self.ids("\n\nHello!") + [EOS]
        self.stub.push(script)
        text, u = self.run_chat()
        self.assertEqual(text, {"reasoning": "Let me think.", "content": "Hello!"})
        self.assertEqual(u.completion_tokens, len(script))
        self.assertEqual(u.reasoning_tokens, len(think))
        self.assertEqual(u.finish, "stop")
        self.assertGreater(u.prompt_tokens, 20)

    def test_thinking_disabled(self):
        self.stub.push(self.ids("Hello there.") + [EOS])
        text, u = self.run_chat(template_kwargs={"enable_thinking": False})
        self.assertEqual(text, {"reasoning": "", "content": "Hello there."})
        self.assertEqual(u.reasoning_tokens, 0)

    def test_stop_string(self):
        self.stub.push(self.ids("Hello STOP world") + [EOS])
        text, u = self.run_chat(template_kwargs={"enable_thinking": False}, stop=["STOP"])
        self.assertEqual(text["content"], "Hello ")
        self.assertEqual(u.finish, "stop")

    def test_length_finish(self):
        self.stub.push(self.ids("word " * 50) + [EOS])
        text, u = self.run_chat(template_kwargs={"enable_thinking": False}, max_tokens=5)
        self.assertEqual(u.finish, "length")
        self.assertEqual(u.completion_tokens, 5)

    def test_loop_guard_bridges_to_an_answer(self):
        prefix = self.ids("I will think about this problem carefully. ")
        unit = self.ids("and again and")[:5]
        self.stub.push(prefix + unit, loop_from=len(prefix))        # an endless repetition inside the thinking
        answer = self.ids("Final answer.") + [EOS]
        self.stub.push(answer)                                       # what the model writes after the bridge
        self.stub.delay_us(300)
        text, u = self.run_chat(max_tokens=100000)
        self.assertIn(loopguard.NOTICE.strip(), text["reasoning"])
        self.assertEqual(text["content"], "Final answer.")
        self.assertEqual(u.loop_guard["where"], "reasoning")
        self.assertEqual(u.loop_guard["kind"], "repeat")
        self.assertEqual(u.finish, "stop")
        first = u.loop_guard["tokens_before"]
        self.assertEqual(u.completion_tokens, first + len(self.runner.bridge) + len(answer))
        # the second call continued the same sequence: prompt + the first part + the bridge sentence
        self.assertEqual(self.stub.d.stub_last_n_prompt(), u.prompt_tokens + first + len(self.runner.bridge))
        self.assertEqual(u.loop_guard["answer_tokens"], len(answer))

    def test_loop_guard_off(self):
        prefix = self.ids("Thinking. ")
        self.stub.push(prefix + self.ids("ab ab ab")[:4], loop_from=len(prefix))
        text, u = self.run_chat(max_tokens=700, loop_guard=False)
        self.assertIsNone(u.loop_guard)
        self.assertEqual(u.finish, "length")
        self.assertEqual(u.completion_tokens, 700)

    def test_loop_in_the_answer_stops(self):
        script = self.ids("ok") + [self.fmt.think_end] + self.ids("\n\nstart ") + self.ids("x y")[:2] * 3
        self.stub.push(script, loop_from=len(script) - 2)
        text, u = self.run_chat(max_tokens=100000)
        self.assertEqual(u.loop_guard["where"], "answer")
        self.assertEqual(u.finish, "length")

    def test_prompt_too_long(self):
        with self.assertRaises(RequestError):
            self.run_chat(messages=[{"role": "user", "content": "word " * 30000}])

    def test_seed_counter_and_explicit_seed(self):
        self.stub.push([1000, EOS])
        self.run_chat()
        self.assertEqual(self.stub.opt(self.engine, 13), 100)
        self.stub.push([1000, EOS])
        self.run_chat()
        self.assertEqual(self.stub.opt(self.engine, 13), 101)
        self.stub.push([1000, EOS])
        self.run_chat(seed=7)
        self.assertEqual(self.stub.opt(self.engine, 13), 7)

    def test_emit_failure_stops_the_engine(self):
        self.stub.push(self.ids("a long answer " * 40), loop_from=0)
        self.stub.delay_us(500)

        def bad(kind, text):
            raise BrokenPipeError("client went away")
        with self.assertRaises(BrokenPipeError):
            self.runner.run(ChatParams(messages=[{"role": "user", "content": "hi"}], max_tokens=1_000_000,
                                       template_kwargs={"enable_thinking": False}), bad)
        self.stub.push([1000, EOS])
        text, u = self.run_chat()                                    # the engine is free again
        self.assertEqual(u.finish, "stop")


if __name__ == "__main__":
    unittest.main()
