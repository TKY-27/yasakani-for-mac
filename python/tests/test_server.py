# SPDX-License-Identifier: Apache-2.0
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests import support
from yasakani.chat import ChatRunner
from yasakani.runtime import EOS_IDS
from yasakani.server import ApiState, ServerConfig, bind
from yasakani.textio import ChatFormatter

EOS = EOS_IDS[0]
HAVE_TOK = support.tokenizer_dir() is not None


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def start(self, **cfg_kw):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.engine = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000)
        cfg = ServerConfig(port=0, **cfg_kw)
        self.state = ApiState(cfg)
        self.state.runner = ChatRunner(self.engine, self.fmt, seed0=1)
        self.httpd = bind(self.state)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
            self.engine.close()
            self.tmp.cleanup()
            self.httpd = None

    def ids(self, text):
        return self.fmt.encode(text).tolist()

    def request(self, method, path, body=None, headers=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        self.addCleanup(c.close)
        h = {"Host": host or "127.0.0.1:%d" % self.port}
        if body is not None:
            h["Content-Type"] = "application/json"
            body = json.dumps(body)
        h.update(headers or {})
        c.request(method, path, body=body, headers=h)
        return c, c.getresponse()

    def chat(self, **body):
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        c, r = self.request("POST", "/v1/chat/completions", body)
        data = r.read()
        c.close()
        return r.status, json.loads(data)

    def test_health_and_models(self):
        self.start()
        c, r = self.request("GET", "/health")
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["status"], "ok")
        c, r = self.request("GET", "/v1/models")
        self.assertEqual(json.loads(r.read())["data"][0]["id"], "qwen3.8-27b")
        self.assertEqual(r.getheader("X-Frame-Options"), "DENY")
        c, r = self.request("GET", "/nope")
        self.assertEqual(r.status, 404)

    def test_loading_state(self):
        self.start()
        self.state.runner = None
        c, r = self.request("GET", "/health")
        self.assertEqual((r.status, json.loads(r.read())["status"]), (503, "loading"))
        status, body = self.chat()
        self.assertEqual(status, 503)

    def test_completion(self):
        self.start()
        self.stub.push(self.ids("Thinking it over.") + [self.fmt.think_end] + self.ids("\n\nThe answer is 4.") + [EOS])
        status, d = self.chat(max_tokens=200, seed=5)
        self.assertEqual(status, 200)
        m = d["choices"][0]["message"]
        self.assertEqual((m["content"], m["reasoning_content"]), ("The answer is 4.", "Thinking it over."))
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        u = d["usage"]
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(self.stub.opt(self.engine, 13), 5)

    def test_streaming(self):
        self.start()
        self.stub.push(self.ids("Let me see.") + [self.fmt.think_end] + self.ids("\n\nHello, world.") + [EOS])
        self.stub.delay_us(2000)
        c, r = self.request("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}], "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(r.status, 200)
        self.assertIn("text/event-stream", r.getheader("Content-Type"))
        raw = r.read().decode("utf-8")
        events = [e[len("data: "):] for e in raw.split("\n\n") if e.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        objs = [json.loads(e) for e in events[:-1]]
        self.assertEqual(objs[0]["choices"][0]["delta"], {"role": "assistant"})
        reasoning = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs if o["choices"])
        content = "".join(o["choices"][0]["delta"].get("content", "") for o in objs if o["choices"])
        self.assertEqual((reasoning, content), ("Let me see.", "Hello, world."))
        finals = [o for o in objs if o["choices"] and o["choices"][0]["finish_reason"]]
        self.assertEqual(finals[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(objs[-1]["choices"], [])
        self.assertEqual(objs[-1]["usage"]["completion_tokens"] > 0, True)

    def test_stream_has_one_chunk_per_token(self):
        self.start()
        text = "The quick brown fox jumps over the lazy dog, again and again."
        answer = self.ids(text)
        self.stub.push([self.fmt.think_end] + answer + [EOS])        # the stub writes one to four ids per pass
        self.stub.delay_us(3000)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        raw = r.read().decode("utf-8")
        objs = [json.loads(e[len("data: "):]) for e in raw.split("\n\n") if e.startswith("data: ") and not e.endswith("[DONE]")]
        pieces = [o["choices"][0]["delta"]["content"] for o in objs if o["choices"] and o["choices"][0]["delta"].get("content")]
        self.assertEqual("".join(pieces), text)
        self.assertEqual(len(pieces), len(answer))                   # a client that counts chunks counts tokens

    def test_rejections(self):
        self.start()
        for body, text in (({"messages": []}, "messages"), ({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}, "tools"),
                           ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n must"),
                           ({"messages": [{"role": "user", "content": "x"}], "temperature": 5}, "temperature"),
                           ({"messages": [{"role": "user", "content": "x"}], "stop": ["a"] * 5}, "stop"),
                           ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, "text content")):
            status, d = self.chat(**body)
            self.assertEqual(status, 400, body)
            self.assertIn(text, d["error"]["message"])
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(c.close)
        c.request("POST", "/v1/chat/completions", body=b"", headers={"Content-Length": str(64 * 1024 * 1024), "Host": "127.0.0.1:%d" % self.port})
        self.assertEqual(c.getresponse().status, 413)                # refused from the header alone (the body is never read)
        c.close()

    def test_bad_json(self):
        self.start()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(c.close)
        c.request("POST", "/v1/chat/completions", body=b"{not json", headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        c.close()

    def test_host_and_origin_checks(self):
        self.start()
        c, r = self.request("GET", "/health", host="evil.example:%d" % self.port)
        self.assertEqual(r.status, 403)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]},
                            {"Origin": "http://evil.example"})
        self.assertEqual(r.status, 403)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]},
                            {"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status, 403)

    def test_api_key(self):
        self.start(api_key="s3cret")
        self.stub.push([1000, EOS])
        status, d = self.chat()
        self.assertEqual(status, 401)
        c, r = self.request("GET", "/v1/models")
        self.assertEqual(r.status, 401)
        c, r = self.request("GET", "/health")                    # health needs no key
        self.assertEqual(r.status, 200)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]},
                            {"Authorization": "Bearer s3cret"})
        self.assertEqual(r.status, 200)
        r.read()

    def test_engine_failure_is_a_500(self):
        self.start()
        # a prompt whose first id is 7 makes the stub fail; render by raw messages is not possible, so fail via the runner
        orig = self.state.runner.engine.generate

        def failing(*a, **k):
            from yasakani._native import YasakaniError
            raise YasakaniError("CUDA error: something")
        self.state.runner.engine.generate = failing
        status, d = self.chat()
        self.assertEqual(status, 500)
        self.assertIn("CUDA error", d["error"]["message"])
        self.state.runner.engine.generate = orig

    def test_client_disconnect_stops_the_engine(self):
        self.start()
        self.stub.push(self.ids("endless text goes on ") * 3, loop_from=0)
        self.stub.delay_us(1000)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 1_000_000,
                           "chat_template_kwargs": {"enable_thinking": False}, "loop_guard": False}).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % (self.port, len(body)) + body)
        got = b""
        while got.count(b"data: ") < 3:                         # headers, the role chunk and some text
            got += s.recv(4096)
        s.close()                                              # the client goes away mid-stream
        t0 = time.time()
        self.stub.push([1000, EOS])
        status = None
        while time.time() - t0 < 10:
            c2, r2 = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "again"}], "max_tokens": 20})
            status = r2.status
            r2.read()
            c2.close()
            if status == 200:
                break
        self.assertEqual(status, 200)                           # the engine was cancelled and the lock released
        self.assertLess(time.time() - t0, 8)


if __name__ == "__main__":
    unittest.main()
