import ast,json,random,re,textwrap as TW
from pathlib import Path

def strip_fences(s):
    t = s.strip("\n")
    m = re.match(r"^```[a-zA-Z]*\n(.*)\n```\s*$", t, re.S)
    return (m.group(1), True) if m else (s, False)

class Rename(ast.NodeTransformer):
    def __init__(self, a, b): self.a, self.b = a, b
    def visit_Name(self, node):
        if node.id == self.a: node.id = self.b
        return node
    def visit_arg(self, node):
        if node.arg == self.a: node.arg = self.b
        return node

def ast_check(original, edited):
    """every top-level node equal except dedent; dedent equal after renaming common_margin -> margin in the edited one"""
    o = ast.parse(original); e = ast.parse(edited); out = dict(ok=False, n_top=len(o.body), notes=[])
    if len(o.body) != len(e.body): out["notes"].append(f"top-level node count {len(o.body)} vs {len(e.body)}"); return out
    bad = []
    for a, b in zip(o.body, e.body):
        da, db = ast.dump(a), ast.dump(b)
        if da == db: continue
        if isinstance(b, ast.FunctionDef) and b.name == "dedent":
            names_e = sorted({n.id for n in ast.walk(b) if isinstance(n, ast.Name)}); names_o = sorted({n.id for n in ast.walk(a) if isinstance(n, ast.Name)})
            out["dedent_names_original"] = names_o; out["dedent_names_edited"] = names_e
            out["common_margin_in_dedent"] = "common_margin" in names_e; out["margin_left_in_dedent"] = "margin" in names_e
            b2 = Rename("common_margin", "margin").visit(ast.parse(ast.unparse(b)).body[0]); eq = ast.dump(b2) == da
            out["dedent_equal_after_rename_back"] = eq
            if not eq:
                # strings changed too?  compare with string constants masked
                def mask(t):
                    for n in ast.walk(t):
                        if isinstance(n, ast.Constant) and isinstance(n.value, str): n.value = "<str>"
                    return ast.dump(t)
                out["dedent_equal_ignoring_strings"] = mask(ast.parse(ast.unparse(b2)).body[0]) == mask(ast.parse(ast.unparse(a)).body[0])
            if not out["common_margin_in_dedent"] or out["margin_left_in_dedent"] or not (eq or out.get("dedent_equal_ignoring_strings")): bad.append("dedent")
        else:
            bad.append(getattr(b, "name", type(b).__name__))
    out["changed_outside_dedent"] = [x for x in bad if x != "dedent"]
    dedent_ok = out.get("common_margin_in_dedent") is True and out.get("margin_left_in_dedent") is False and (out.get("dedent_equal_after_rename_back") or out.get("dedent_equal_ignoring_strings"))
    out["ok"] = bool(not out["changed_outside_dedent"] and dedent_ok)
    return out

def differential(edited_code, original_code, n_dedent=10000, n_wrap=2000, seed=0):
    """the model's module against the ORIGINAL file of the prompt (same textwrap version, both executed here) on random inputs; the installed
    stdlib textwrap is compared too, for information only (the prompt's file is an older textwrap than Python 3.14's)"""
    ns = {}; exec(compile(edited_code, "model_textwrap.py", "exec"), ns); no = {}; exec(compile(original_code, "original_textwrap.py", "exec"), no)
    rng = random.Random(seed); alphabet = "abc  \t\n\n  xyz., -\u3042\u3044\u3046 "; bad = 0; bad_std = 0; cases = 0
    for _ in range(n_dedent):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        if rng.random() < 0.5: s = "\n".join((" " * rng.randint(0, 6) + "\t" * rng.randint(0, 2) + l) for l in s.split("\n"))
        cases += 1; m = ns["dedent"](s); bad += m != no["dedent"](s); bad_std += m != TW.dedent(s)
    words = "the quick brown fox jumps over the lazy dog while a very-long-hyphenated-word and some_identifier_names appear".split()
    for _ in range(n_wrap):
        s = " ".join(rng.choice(words) for _ in range(rng.randint(0, 40))); w = rng.randint(1, 60); kw = dict(width=w, break_long_words=rng.random() < 0.5, break_on_hyphens=rng.random() < 0.5)
        cases += 3
        for fn in ("wrap", "fill"):
            m = ns[fn](s, **kw); bad += m != no[fn](s, **kw); bad_std += m != getattr(TW, fn)(s, **kw)
        t = s.replace(" ", "\n", 3); m = ns["indent"](t, "> "); bad += m != no["indent"](t, "> "); bad_std += m != TW.indent(t, "> ")
        if w >= 10: cases += 1; m = ns["shorten"](s, width=w); bad += m != no["shorten"](s, width=w); bad_std += m != TW.shorten(s, width=w)
    return dict(cases=cases, mismatches_vs_original=bad, mismatches_vs_installed_stdlib=bad_std, ok=bad == 0)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('original');ap.add_argument('generated');a=ap.parse_args()
 original=Path(a.original).read_text(encoding='utf-8');code,_=strip_fences(Path(a.generated).read_text(encoding='utf-8'))
 check=ast_check(original,code);assert check['ok'],'AST rejected: will not execute generated code'
 result=differential(code,original)
 print(json.dumps(dict(ast=check,differential=result),ensure_ascii=False,indent=2))
 raise SystemExit(0 if result['ok'] else 1)
