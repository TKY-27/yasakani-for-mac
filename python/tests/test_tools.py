# SPDX-License-Identifier: Apache-2.0
"""Tool calling: the filter that finds the calls in the answer text, the request checks, and requests through the runner and the server."""
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from tests import support
from yasakani.chat import ChatParams, ChatRunner, RequestError, clean_messages, clean_tools
from yasakani.runtime import EOS_IDS
from yasakani.server import ApiState, ServerConfig, bind, parse_chat_request
from yasakani.textio import ChatFormatter, ToolCallFilter, convert_parameter, parse_tool_call, tool_schemas

EOS = EOS_IDS[0]
HAVE_TOK = support.tokenizer_dir() is not None

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                                                                      "options": {"type": "object"}, "overwrite": {"type": "boolean"}},
                                                     "required": ["path", "content"]}}},
]
CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
CALL2 = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
USER = [{"role": "user", "content": "How is the weather in Osaka?"}]


def feed(f, text, step=7):
    out = []
    for i in range(0, len(text), step):
        out.extend(f.push(text[i:i + step]))
    out.extend(f.flush())
    return out


def content_of(events):
    return "".join(item for kind, item in events if kind == "content")


def calls_of(events):
    return [item for kind, item in events if kind == "tool_call"]


class ToolCallFilterTests(unittest.TestCase):
    def test_plain_text_passes_unchanged(self):
        for step in (1, 3, 50):
            f = ToolCallFilter(TOOLS)
            text = "Hello,  world.\n\nA < B and <tool, done. "
            ev = feed(f, text, step)
            self.assertEqual(content_of(ev), text)
            self.assertEqual((calls_of(ev), f.calls), ([], 0))

    def test_one_call_after_some_text(self):
        for step in (1, 7, 500):
            f = ToolCallFilter(TOOLS)
            ev = feed(f, "I will check.\n\n" + CALL, step)
            self.assertEqual(content_of(ev), "I will check.")              # the white space before the call is not part of the text
            self.assertEqual(calls_of(ev), [{"name": "get_weather", "arguments": {"city": "Osaka", "days": 3}}])
            self.assertEqual(f.calls, 1)

    def test_two_calls(self):
        f = ToolCallFilter(TOOLS)
        ev = feed(f, CALL + "\n" + CALL2 + "\n")
        self.assertEqual(content_of(ev), "")
        self.assertEqual([c["arguments"]["city"] for c in calls_of(ev)], ["Osaka", "Tokyo"])

    def test_an_unfinished_call_comes_back_as_text(self):
        f = ToolCallFilter(TOOLS)
        ev = feed(f, "Text\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOsa")
        self.assertEqual(calls_of(ev), [])
        self.assertEqual(content_of(ev), "Text\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOsa")   # nothing the model wrote is lost

    def test_an_unreadable_block_comes_back_as_text(self):
        f = ToolCallFilter(TOOLS)
        block = "<tool_call>\nthis is not a call\n</tool_call>"
        ev = feed(f, block)
        self.assertEqual((content_of(ev), calls_of(ev)), (block, []))

    def test_the_json_form_is_read_too(self):
        f = ToolCallFilter(TOOLS)
        ev = feed(f, '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Osaka"}}\n</tool_call>')
        self.assertEqual(calls_of(ev), [{"name": "get_weather", "arguments": {"city": "Osaka"}}])

    def test_parameter_values(self):
        props = tool_schemas(TOOLS)["write_file"]
        text = 'line 1\n  "quoted" <b>tag</b>\n\n123\n'
        block = ("\n<function=write_file>\n<parameter=path>\nC:\\tmp\\a.txt\n</parameter>\n<parameter=content>\n" + text + "\n</parameter>\n"
                 "<parameter=options>\n{\"mode\": \"w\", \"n\": [1, 2]}\n</parameter>\n<parameter=overwrite>\ntrue\n</parameter>\n</function>\n")
        call = parse_tool_call(block, tool_schemas(TOOLS))
        self.assertEqual(call, {"name": "write_file", "arguments": {"path": "C:\\tmp\\a.txt", "content": text, "options": {"mode": "w", "n": [1, 2]},
                                                                    "overwrite": True}})
        self.assertEqual(convert_parameter("3", props["content"]), "3")          # declared as a string: stays text
        self.assertEqual(convert_parameter("3", {"type": "integer"}), 3)
        self.assertEqual(convert_parameter("3", None), 3)                        # not declared: JSON when it is JSON ...
        self.assertEqual(convert_parameter('"x"', None), '"x"')                  # ... but a quoted text stays as the model wrote it
        self.assertEqual(convert_parameter("not json", {"type": "object"}), "not json")
        self.assertIsNone(parse_tool_call("no function here", {}))


class ToolRequestTests(unittest.TestCase):
    def test_clean_tools(self):
        self.assertIsNone(clean_tools(None))
        self.assertIsNone(clean_tools([]))
        t = clean_tools([{"type": "function", "function": {"name": "f", "strict": True}}, {"function": {"name": "g", "description": "d",
                                                                                                    "parameters": {"type": "object"}}}])
        self.assertEqual(t[0], {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}})
        self.assertEqual(t[1]["function"], {"name": "g", "description": "d", "parameters": {"type": "object"}})
        for bad in ("x", [5], [{"type": "retrieval"}], [{"type": "function", "function": {}}], [{"type": "function", "function": {"name": "a<b"}}],
                    [{"type": "function", "function": {"name": "f", "parameters": []}}]):
            with self.assertRaises(RequestError):
                clean_tools(bad)

    def test_clean_messages_with_tool_calls(self):
        m = clean_messages([
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Osaka\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}])
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "tool"])
        self.assertEqual(m[2]["tool_calls"], [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Osaka"}}},
                                              {"type": "function", "function": {"name": "get_weather", "arguments": {}}}])
        for calls in ("x", [{"function": {"name": ""}}], [{"function": {"name": "f", "arguments": "{not json"}}],
                      [{"function": {"name": "f", "arguments": "[1]"}}]):
            with self.assertRaises(RequestError):
                clean_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "tool_calls": calls}])

    def test_parse_chat_request(self):
        cfg = ServerConfig(port=0, default_thinking_budget=0)
        p = parse_chat_request({"messages": USER, "tools": TOOLS, "parallel_tool_calls": True}, cfg)
        self.assertEqual([t["function"]["name"] for t in p.tools], ["get_weather", "write_file"])
        self.assertIsNone(parse_chat_request({"messages": USER, "tools": TOOLS, "tool_choice": "none"}, cfg).tools)
        self.assertIsNotNone(parse_chat_request({"messages": USER, "tools": TOOLS, "tool_choice": "auto"}, cfg).tools)
        self.assertIsNone(parse_chat_request({"messages": USER}, cfg).tools)
        for extra in ({"tool_choice": "required"}, {"tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
                      {"functions": [{"name": "f"}]}, {"function_call": "auto"}, {"thinking_budget": -1}):
            with self.assertRaises(RequestError):
                parse_chat_request(dict({"messages": USER, "tools": TOOLS}, **extra), cfg)
        p = parse_chat_request({"messages": USER, "thinking": {"type": "disabled"}, "thinking_budget": 500}, cfg)
        self.assertEqual((p.template_kwargs.get("enable_thinking"), p.thinking_budget), (False, 500))
        self.assertEqual(parse_chat_request({"messages": USER}, ServerConfig(port=0, default_thinking_budget=4096)).thinking_budget, 4096)


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class ToolRunTests(unittest.TestCase):
    """The model's own chat template and tokenizer, the stub library in place of the engine."""

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.engine = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000)
        self.runner = ChatRunner(self.engine, self.fmt, seed0=1)
        self.state = ApiState(ServerConfig(port=0, default_thinking_budget=0))
        self.state.runner = self.runner
        self.httpd = bind(self.state)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.close()
        self.tmp.cleanup()

    def ids(self, text):
        return self.fmt.encode(text).tolist()

    def post(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        self.addCleanup(c.close)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json", "Host": "127.0.0.1:%d" % self.port})
        r = c.getresponse()
        raw = r.read().decode("utf-8")
        c.close()
        return r.status, raw

    def test_the_template_shows_the_tools_and_reads_a_call_back(self):
        msgs = clean_messages(USER + [
            {"role": "assistant", "content": "I will check.", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": json.dumps({"city": "Osaka", "days": 3})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny, 22 C"}])
        text = self.fmt.render(msgs, tools=clean_tools(TOOLS))
        self.assertIn('"name": "get_weather"', text)
        self.assertIn(CALL, text)                                        # the template writes a call in the form the filter reads
        self.assertIn("<tool_response>\nsunny, 22 C\n</tool_response>", text)
        f = ToolCallFilter(TOOLS)
        self.assertEqual(calls_of(feed(f, CALL)), [{"name": "get_weather", "arguments": {"city": "Osaka", "days": 3}}])

    def test_runner_returns_the_call(self):
        self.stub.push(self.ids("The user wants the weather.") + [self.fmt.think_end] + self.ids("\n\nI will check.\n\n" + CALL) + [EOS])
        ev = []
        u = self.runner.run(ChatParams(messages=USER, tools=clean_tools(TOOLS)), lambda kind, item: ev.append((kind, item)))
        self.assertEqual(content_of(ev), "I will check.")
        self.assertEqual(calls_of(ev), [{"name": "get_weather", "arguments": {"city": "Osaka", "days": 3}}])
        self.assertEqual((u.finish, u.tool_calls), ("tool_calls", 1))

    def test_server_answers_with_tool_calls(self):
        self.stub.push([self.fmt.think_end] + self.ids("\n\n" + CALL + "\n" + CALL2) + [EOS])
        status, raw = self.post({"messages": USER, "tools": TOOLS})
        self.assertEqual(status, 200, raw)
        choice = json.loads(raw)["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        calls = choice["message"]["tool_calls"]
        self.assertEqual([(c["type"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [("function", "get_weather", {"city": "Osaka", "days": 3}), ("function", "get_weather", {"city": "Tokyo"})])
        self.assertEqual(len({c["id"] for c in calls}), 2)

    def test_server_stream_sends_one_chunk_per_call(self):
        self.stub.push([self.fmt.think_end] + self.ids("\n\nI will check.\n\n" + CALL + "\n" + CALL2) + [EOS])
        status, raw = self.post({"messages": USER, "tools": TOOLS, "stream": True})
        self.assertEqual(status, 200, raw)
        objs = [json.loads(e[len("data: "):]) for e in raw.split("\n\n") if e.startswith("data: ") and not e.endswith("[DONE]")]
        deltas = [o["choices"][0]["delta"] for o in objs if o.get("choices")]
        calls = [c for d in deltas for c in d.get("tool_calls") or []]
        self.assertEqual([(c["index"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [(0, "get_weather", {"city": "Osaka", "days": 3}), (1, "get_weather", {"city": "Tokyo"})])
        self.assertEqual("".join(d.get("content") or "" for d in deltas), "I will check.")
        self.assertEqual([o["choices"][0]["finish_reason"] for o in objs if o.get("choices")][-1], "tool_calls")

    def test_server_takes_the_tool_result_and_answers(self):
        self.stub.push([self.fmt.think_end] + self.ids("\n\nIt is sunny in Osaka.") + [EOS])
        msgs = USER + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"Osaka\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 22 C"}]
        status, raw = self.post({"messages": msgs, "tools": TOOLS})
        self.assertEqual(status, 200, raw)
        choice = json.loads(raw)["choices"][0]
        self.assertEqual((choice["finish_reason"], choice["message"]["content"]), ("stop", "It is sunny in Osaka."))
        self.assertNotIn("tool_calls", choice["message"])

    def test_server_without_tools_leaves_the_text_alone(self):
        self.stub.push([self.fmt.think_end] + self.ids("\n\n" + CALL) + [EOS])
        status, raw = self.post({"messages": USER})                        # no tools in the request: nothing is read as a call
        self.assertEqual(status, 200, raw)
        choice = json.loads(raw)["choices"][0]
        self.assertEqual((choice["finish_reason"], choice["message"]["content"]), ("stop", CALL))

    def test_server_refuses_a_forced_tool_choice(self):
        for extra, word in (({"tool_choice": "required"}, "tool_choice"), ({"functions": [{"name": "f"}]}, "functions")):
            status, raw = self.post(dict({"messages": USER, "tools": TOOLS}, **extra))
            self.assertEqual(status, 400, raw)
            self.assertIn(word, json.loads(raw)["error"]["message"])

    def test_an_input_that_does_not_fit_is_a_400_before_the_stream_starts(self):
        long = [{"role": "user", "content": "word " * 30000}]
        status, raw = self.post({"messages": long, "tools": TOOLS, "stream": True})
        self.assertEqual(status, 400, raw[:200])
        err = json.loads(raw)["error"]
        self.assertEqual(err["code"], "context_length_exceeded")
        self.assertIn("maximum context length", err["message"])


if __name__ == "__main__":
    unittest.main()
