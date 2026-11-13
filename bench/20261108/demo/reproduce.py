"""Re-run the saved request against an already-running LOCAL compatible server.
This script loads no model and modifies no engine configuration.
Usage: python reproduce.py --base http://127.0.0.1:8099 --out new_capture.json
"""
import argparse,json,time,urllib.request,urllib.parse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
assert urllib.parse.urlparse(a.base).hostname in ('127.0.0.1','localhost','::1'), 'local server only'
dest=Path(a.out);assert not dest.exists(), 'refusing to overwrite a previous run'
body=Path(__file__).with_name('request.json').read_bytes()
req=urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
events=[];chunks=[];usage=None;finish=None;t0=time.perf_counter()
with urllib.request.urlopen(req,timeout=180) as response:
 for line in response:
  if not line.startswith(b'data: '):continue
  raw=line[6:].strip()
  if raw==b'[DONE]':break
  t=time.perf_counter()-t0;j=json.loads(raw);events.append({'t':t,'payload':j})
  if j.get('usage'):usage=j['usage']
  for choice in j.get('choices') or []:
   delta=choice.get('delta') or {}
   if delta.get('content'):chunks.append([t,'content',delta['content']])
   if choice.get('finish_reason'):finish=choice['finish_reason']
tend=time.perf_counter()-t0
assert usage and chunks, 'no complete result'
n=usage['completion_tokens'];first=chunks[0][0];last=chunks[-1][0]
out=dict(ttft_s=first,t_last_s=last,t_end_s=tend,completion_tokens=n,decode_tok_s=(n-1)/(last-first),finish=finish,usage=usage,chunks=chunks,events=events)
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('events','chunks')},ensure_ascii=False))
