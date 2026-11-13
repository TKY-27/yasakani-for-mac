"""loop_scan.py -- find exact repetition loops at the tail of generated texts (2026-11-03 post check).
A tail loop = the longest suffix that is periodic with some period p (20..4000 chars) and spans >= 2 periods.
Reports loop start (fraction of chars) per output. Read-only over existing outputs."""
import json, glob, os, sys
def tail_loop(s):
    n = len(s); best = None
    for p in range(20, min(4000, n // 2) + 1):
        if s[n - p:] != s[n - 2 * p:n - p]:
            # allow the final cycle to be cut: compare the last full cycle before the cut
            continue
        i = n - p
        while i > 0 and s[i - 1] == s[i - 1 + p]: i -= 1
        span = n - i
        if span >= 2 * p and (best is None or span > best[1]): best = (p, span, i)
        break
    if best is None:
        # retry allowing a partial last cycle: search for the smallest p where the region before the cut repeats
        for p in range(20, min(4000, n // 3) + 1):
            m = n - (n % p) if False else n
            k = 0
            while k < p and s[n - 1 - k] == s[n - 1 - k - p]: k += 1
            if k == p:  # covered above
                continue
        return None
    return best
def scan_text(s):
    r = tail_loop(s)
    if not r: 
        # partial-cycle tolerant: drop up to 4000 trailing chars and retry
        for cut in range(1, 4000, 7):
            if cut >= len(s) // 2: break
            r = tail_loop(s[:-cut])
            if r: 
                p, span, i = r
                # extend through the cut tail if it continues the cycle
                j = len(s) - cut
                while j < len(s) and s[j] == s[j - p]: j += 1
                if j == len(s): return (p, len(s) - i, i)
                return None
        return None
    return r
def main():
    global rows
    rows = []
    def add(label, s, extra=''):
        r = scan_text(s)
        if r: p, span, i = r; rows.append((label, len(s), round(100 * span / len(s), 1), p, extra))
        else: rows.append((label, len(s), 0.0, 0, extra))
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ['yasakani_fast_greedy', 'yasakani_std_greedy', 'yasakani_std_greedy_kv8', 'yasakani_fast_t06', 'megacapybara_kv64', 'megacapybara_kv88', 'yasakani_fast_code2000', 'yasakani_std_code2000']:  # folder names of the published layout
        for f in sorted(glob.glob(os.path.join(base, 'results', d, 'texts', '*.txt'))):
            add(d + '/' + os.path.basename(f), open(f, encoding='utf-8', errors='replace').read())
    for f in sorted(glob.glob(os.path.join(base, 'results', 'video_runs', '*.json'))):
        j = json.load(open(f, encoding='utf-8')); s = ''.join(c[1] if isinstance(c, list) else c.get('text', '') for c in j.get('chunks', []))
        add('video_runs/' + os.path.basename(f), s, '%s tok %s' % (j.get('completion_tokens'), j.get('finish')))
    for f in sorted(glob.glob(os.path.join(base, 'results', 'own_tests', '*.txt'))):
        add('own_tests/' + os.path.basename(f), open(f, encoding='utf-8', errors='replace').read())
    for r in rows:
        print('%-62s chars=%6d loop%%=%5.1f period=%4d %s' % r)
if __name__ == '__main__':
    main()
