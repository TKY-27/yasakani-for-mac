# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""YASAKANI quickstart: compare generation with and without look-ahead on your own PC.

docs/TECHNOLOGY.md says that without look-ahead the engine confirms one token per pass of the model, that with the shipping
settings it confirms several, and that the text which is written is the same.  This script lets you check that.  Four tasks are
generated with greedy decoding, first with all look-ahead switched off and then with the shipping settings.  For every run it
prints the time of one pass, the tokens confirmed per pass and the speed, and it tells whether both runs produced the same tokens.

    python lookahead_check.py --model-dir <model folder> --dll <path of yskn.dll> [--tokens 768] [--lang ja|en] [--json]

Only the public Python package is used: yasakani.Engine and its per-request speculation settings.  Nothing is written to disk.
The four tasks and the settings are the ones behind the first table of docs/TECHNOLOGY.md (measured on November 12, 2026).
"""
import argparse
import json
import sys
import time
import unicodedata

# all three sources of drafts switched off: the draft model, the copy of earlier text and the lookup table
NO_LOOKAHEAD = dict(bdraft=0, use_d0=0, use_d1=0)

CSV = (
    "id,first_name,last_name,department,city,salary,start_date,remote\n"
    "1001,Alice,Morgan,Engineering,Osaka,41000,2012-01-01,true\n"
    "1002,Ben,Novak,Finance,Porto,53250,2017-08-12,false\n"
    "1003,Chloe,Lindqvist,Sales,Lyon,65500,2022-03-23,false\n"
    "1004,Daniel,Okafor,Design,Tampere,77750,2013-10-06,false\n"
    "1005,Emma,Tanaka,Support,Austin,49750,2018-05-17,true\n"
    "1006,Felix,Reyes,Engineering,Busan,62000,2023-12-28,false\n"
    "1007,Grace,Reyes,Finance,Leeds,74250,2014-07-11,false\n"
    "1008,Hugo,Morgan,Sales,Osaka,46250,2019-02-22,false\n"
    "1009,Iris,Novak,Design,Porto,58500,2024-09-05,true\n"
    "1010,Jonas,Lindqvist,Support,Lyon,70750,2015-04-16,false\n"
    "1011,Alice,Okafor,Engineering,Tampere,42750,2020-11-27,false\n"
    "1012,Ben,Tanaka,Finance,Austin,55000,2025-06-10,false\n"
    "1013,Chloe,Tanaka,Sales,Busan,67250,2016-01-21,true\n"
    "1014,Daniel,Reyes,Design,Leeds,79500,2021-08-04,false\n"
    "1015,Emma,Morgan,Support,Osaka,51500,2012-03-15,false\n"
    "1016,Felix,Novak,Engineering,Porto,63750,2017-10-26,false\n"
    "1017,Grace,Lindqvist,Finance,Lyon,76000,2022-05-09,true\n"
    "1018,Hugo,Okafor,Sales,Tampere,48000,2013-12-20,false\n"
    "1019,Iris,Okafor,Design,Austin,60250,2018-07-03,false\n"
    "1020,Jonas,Tanaka,Support,Busan,72500,2023-02-14,false\n"
    "1021,Alice,Reyes,Engineering,Leeds,44500,2014-09-25,true\n"
    "1022,Ben,Morgan,Finance,Osaka,56750,2019-04-08,false\n"
    "1023,Chloe,Novak,Sales,Porto,69000,2024-11-19,false\n"
    "1024,Daniel,Lindqvist,Design,Lyon,41000,2015-06-02,false\n"
    "1025,Emma,Lindqvist,Support,Tampere,53250,2020-01-13,true\n"
    "1026,Felix,Okafor,Engineering,Austin,65500,2025-08-24,false\n"
    "1027,Grace,Tanaka,Finance,Busan,77750,2016-03-07,false\n"
    "1028,Hugo,Reyes,Sales,Leeds,49750,2021-10-18,false\n"
    "1029,Iris,Morgan,Design,Osaka,62000,2012-05-01,true\n"
    "1030,Jonas,Novak,Support,Porto,74250,2017-12-12,false"
)
TASKS = [
    # name, thinking, prompt
    ("prose", False, "Write an essay of about 700 words on why cities should plant more trees along their streets."),
    ("thinking", True, "量子コンピュータの仕組みを、高校生にもわかるように800字くらいで説明してください。"),
    ("code", False, "build me a html three.js website for harry potter"),
    ("csv-to-json", False, "Convert the following CSV into a JSON array of objects, one object per row, using the header names as keys. Numbers must be JSON numbers and the remote column must be a JSON boolean. Return only the JSON.\n\n" + CSV),
]
# docs/TECHNOLOGY.md, table 1.3: one RTX 5090, the fastest model, 768 tokens, greedy
PUBLISHED = {"off": 126, "prose": 279, "thinking": 298, "code": 420, "csv-to-json": 1264}

T = {
    "title": {"ja": "先読みあり・なしの比較（貪欲デコード、1 課題 {n} トークンまで）",
              "en": "Look-ahead on and off (greedy decoding, up to {n} tokens per task)"},
    "load": {"ja": "モデルを読み込みます（{g} の重み）…", "en": "loading the model ({g} of weights) ..."},
    "ready": {"ja": "準備ができました（{s:.0f} 秒）。4 つの課題を、先読みなし → あり の順に書かせます。1 分ほどかかります。",
              "en": "ready in {s:.0f} s.  Four tasks follow, each without and then with look-ahead.  This takes about a minute."},
    "head": {"ja": ("課題", "先読み", "1 回の計算", "1 回で確定", "速さ", "書かれた文章"),
             "en": ("task", "look-ahead", "one pass", "tokens/pass", "speed", "text")},
    "off": {"ja": "なし", "en": "off"},
    "on": {"ja": "あり", "en": "on"},
    "same": {"ja": "先読みなしと同じ", "en": "same as without"},
    "diff": {"ja": "違う（{k} 番目のトークンから）", "en": "DIFFERENT (from token {k})"},
    "sum": {"ja": "まとめ", "en": "Summary"},
    "sum_off": {"ja": "  先読みなし: {a:.0f} tok/s（1 回の計算で {t:.2f} トークン、1 回 {m:.2f} ms）",
                "en": "  without look-ahead: {a:.0f} tok/s ({t:.2f} tokens per pass, {m:.2f} ms per pass)"},
    "sum_on": {"ja": "  先読みあり: {lo:,.0f}〜{hi:,.0f} tok/s（1 回の計算で {tl:.2f}〜{th:.2f} トークン）",
               "en": "  with look-ahead:    {lo:,.0f} to {hi:,.0f} tok/s ({tl:.2f} to {th:.2f} tokens per pass)"},
    "sum_same": {"ja": "  書かれた文章: {n} 課題とも、先読みなしと同じでした。",
                 "en": "  text: the same with and without look-ahead in all {n} tasks."},
    "sum_diff": {"ja": "  書かれた文章: {k} 課題で違いがありました。ほかのプログラムが GPU を使っていると起きることがあります\n"
                       "  （docs/MEASUREMENTS.ja.md）。ほかのプログラムを閉じて、もう一度試してください。",
                 "en": "  text: {k} task(s) differed.  This can happen while other programs use the GPU (docs/MEASUREMENTS.md).\n"
                       "  Close them and try again."},
    "pub": {"ja": "技術の解説（docs/TECHNOLOGY.ja.md の 1.3）に載せた値（RTX 5090・最速モデル）:\n"
                  "  なし {off} / 文章 {prose} / 思考つき {thinking} / コード 約 {code} / CSV→JSON {csv} tok/s\n"
                  "  （品質モデルでは、これより 1〜2 割ほど低くなります）",
            "en": "Published in docs/TECHNOLOGY.md, table 1.3 (one RTX 5090, the fastest model):\n"
                  "  off {off} / prose {prose} / with thinking {thinking} / code about {code} / CSV to JSON {csv} tok/s\n"
                  "  (with the quality model the numbers are 10 to 20% lower)"},
}


def ui_lang(arg):
    if arg in ("ja", "en"):
        return arg
    try:
        import ctypes
        return "ja" if (ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF) == 0x11 else "en"
    except Exception:  # noqa: BLE001 - not Windows, or no answer: English
        return "en"


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def table_line(cols):
    """task | look-ahead | one pass | tokens per pass | speed | text  (columns padded by display width: Japanese is double width)"""
    cells = [cols[0] + " " * max(0, 12 - width(cols[0])), cols[1] + " " * max(0, 10 - width(cols[1]))]
    cells += [" " * max(0, n - width(c)) + c for c, n in zip(cols[2:5], (10, 11, 13))]
    return ("  ".join(cells) + ("    " + cols[5] if len(cols) > 5 and cols[5] else "")).rstrip()


def human(n):
    return "%.1f GiB" % (n / (1 << 30))


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1
    return min(len(a), len(b)) + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare generation with and without look-ahead")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--tokens", type=int, default=768, help="tokens to generate per task (default 768, as in the published table)")
    ap.add_argument("--lang", choices=("ja", "en"))
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    lang = ui_lang(a.lang)
    t = lambda key, **kw: T[key][lang].format(**kw) if kw else T[key][lang]     # noqa: E731
    say = (lambda *x: print(*x, file=sys.stderr, flush=True)) if a.json else (lambda *x: print(*x, flush=True))

    from yasakani import Engine, ModelPack, __version__
    from yasakani.engine import Sampling
    from yasakani.textio import ChatFormatter

    pack = ModelPack.load(a.model_dir)
    fmt = ChatFormatter(pack.root)
    say(t("title", n=a.tokens))
    say(t("load", g=human(pack.weights.stat().st_size)))
    t0 = time.time()
    eng = Engine(pack, dll=a.dll)
    rows = []
    try:
        eng.warmup([fmt.encode("warm-up"), fmt.prompt([{"role": "user", "content": "warm-up"}])[0]], n_gen=32)
        say(t("ready", s=time.time() - t0))
        say("")
        say(table_line(t("head")))
        greedy = Sampling(0.0, 0.95, 20, 0.0, 0)

        def run(ids, spec):
            eng.reset()
            eng.set_sampling(greedy)
            r = eng.generate(ids, a.tokens, spec=spec)
            n = len(r.ids)
            passes = max(1, r.passes)
            return dict(ids=[int(x) for x in r.ids], tokens=n, passes=r.passes, ms_per_pass=r.gen_ms / passes,
                        tokens_per_pass=(n - 1) / passes, tokens_per_s=(n - 1) / (r.gen_ms / 1000.0) if r.gen_ms > 0 else 0.0)

        def cells(name, label, r):
            return [name, label, "%.2f ms" % r["ms_per_pass"], "%.2f" % r["tokens_per_pass"], "{:,.0f} tok/s".format(r["tokens_per_s"])]

        for name, think, text in TASKS:
            ids, _ = fmt.prompt([{"role": "user", "content": text}], enable_thinking=think)
            off = run(ids, NO_LOOKAHEAD)
            say(table_line(cells(name, t("off"), off)))
            on = min((run(ids, None) for _ in range(2)), key=lambda r: r["ms_per_pass"] * r["passes"])
            same = on["ids"] == off["ids"]
            verdict = t("same") if same else t("diff", k=first_difference(on["ids"], off["ids"]))
            say(table_line(cells("", t("on"), on) + [verdict]))
            for r in (off, on):
                del r["ids"]
            rows.append(dict(task=name, thinking=think, prompt_tokens=int(len(ids)), off=off, on=on, same_text=same))
    finally:
        eng.close()

    offs = [r["off"] for r in rows]
    ons = [r["on"] for r in rows]
    differ = sum(1 for r in rows if not r["same_text"])
    say("")
    say(t("sum"))
    say(t("sum_off", a=sum(o["tokens_per_s"] for o in offs) / len(offs), t=sum(o["tokens_per_pass"] for o in offs) / len(offs),
          m=sum(o["ms_per_pass"] for o in offs) / len(offs)))
    say(t("sum_on", lo=min(o["tokens_per_s"] for o in ons), hi=max(o["tokens_per_s"] for o in ons),
          tl=min(o["tokens_per_pass"] for o in ons), th=max(o["tokens_per_pass"] for o in ons)))
    say(t("sum_diff", k=differ) if differ else t("sum_same", n=len(rows)))
    say("")
    say(t("pub", off=PUBLISHED["off"], prose=PUBLISHED["prose"], thinking=PUBLISHED["thinking"], code=PUBLISHED["code"],
          csv="{:,}".format(PUBLISHED["csv-to-json"])))
    if a.json:
        print(json.dumps(dict(version=__version__, model=pack.model_id, tokens_per_task=a.tokens, no_lookahead=NO_LOOKAHEAD, tasks=rows,
                              published_tokens_per_s=PUBLISHED), indent=1))
    return 1 if differ else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
