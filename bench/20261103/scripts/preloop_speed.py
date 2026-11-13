"""preloop_speed.py -- speed before the tail loop (story_en_mystery captures), and MegaCapybara over the same number of tokens."""
import json, sys
sys.path.insert(0, 'scripts')   # run from the 20261103 folder of the published layout
from loop_scan import scan_text
def load(name):
    j = json.load(open('results/video_runs/%s_scenario_story_en_mystery_r0.json' % name, encoding='utf-8'))
    ch = j['chunks']; t = [c[0] for c in ch]; s = [c[1] for c in ch]
    return j, t, s
out = {}
for name in ['YASAKANI_fast', 'YASAKANI_std', 'MegaCapybara_Tiny']:
    j, t, s = load(name); full = ''.join(s); r = scan_text(full)
    toks = j['completion_tokens']; n = len(full)
    t0 = t[0]; t1 = t[-1]
    if r:
        p, span, i = r; acc = 0
        for k, c in enumerate(s):
            acc += len(c)
            if acc >= i: tl = t[k]; break
        ntok = round(toks * i / n)
        out[name] = (ntok, (ntok - 1) / ((tl - t0) / 1000), (toks - ntok) / ((t1 - tl) / 1000))
        print('%-18s tokens %d  loop from ~%d tok (%.1f%%)  pre-loop %.1f tok/s  in-loop %.1f tok/s  overall %.1f (reported %s)' % (name, toks, ntok, 100 * (n - i) / n, out[name][1], out[name][2], (toks - 1) / ((t1 - t0) / 1000), j['decode_tok_s']))
    else:
        print('%-18s tokens %d  no loop  overall %.1f (reported %s), chunks %d' % (name, toks, (toks - 1) / ((t1 - t0) / 1000), j['decode_tok_s'], len(t)))
j, t, s = load('MegaCapybara_Tiny')
for nm in ['YASAKANI_fast', 'YASAKANI_std']:
    N = out[nm][0]
    # MegaCapybara streams one token per chunk (3,982 chunks for 3,983 tokens): the N-th token arrives with chunk N-1
    tn = t[min(N, len(t)) - 1]
    mc = (N - 1) / ((tn - t[0]) / 1000)
    print('same first %d tokens: %s %.1f vs MegaCapybara %.1f -> %.2fx' % (N, nm, out[nm][1], mc, out[nm][1] / mc))
