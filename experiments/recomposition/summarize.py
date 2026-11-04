#!/usr/bin/env python3
"""Summarize raw measured repetitions, excluding the warmups omitted by Rust."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
root=p.parse_args().results
summary={}
def stats(xs):
 return {'median':statistics.median(xs),'min':min(xs),'max':max(xs),'n':len(xs)}
for step in range(1,4):
 records=[json.loads(s) for s in (root/f'measure-{step}.jsonl').read_text().splitlines()]
 data={'plan':next(r for r in records if r['event']=='plan'),'oracle':next(r for r in records if r['event']=='oracle')}
 for event in ['fresh','carried']:
  rows=[r for r in records if r['event']==event]
  if rows:data[event]={key:stats([r[key] for r in rows]) for key in rows[0] if key.endswith('_seconds')}
 rows=[r['timings'] for r in records if r['event']=='openings_split']
 data['openings']={key:stats([sum(r[key]) if isinstance(r[key],list) else r[key] for r in rows]) for key in rows[0]}
 proofs=root/f'proofs-{step}.jsonl'
 if proofs.exists():
  rows=[json.loads(s) for s in proofs.read_text().splitlines()]
  data['proofs']={v:stats([r['seconds'] for r in rows if r['variant']==v]) for v in ['production','factor']}
  data['proof_plan_setup_seconds']=rows[0]['setup_seconds']
  data['proofs']['paired_savings']=stats([next(r['seconds'] for r in rows if r['trial']==t and r['variant']=='production')-next(r['seconds'] for r in rows if r['trial']==t and r['variant']=='factor') for t in range(1,4)])
 summary[step]=data
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
