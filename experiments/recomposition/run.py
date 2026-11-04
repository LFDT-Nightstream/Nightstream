#!/usr/bin/env python3
"""All test processes have an unconditional 300-second process-group cap."""
import argparse,json,os,signal,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('mode',choices=['capture','measure','proofs','checks'])
p.add_argument('--directory',default='/private/tmp/nightstream-recomposition-handoff')
p.add_argument('--target-dir')
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
p.add_argument('--step',type=int,default=1)
a=p.parse_args()
out=a.output.resolve()
out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]
os.chdir(root)
name='capture' if a.mode=='capture' else f'{a.mode}-{a.step}'
cmd=['cargo','test','--release','-p','nightstream' if a.mode=='capture' else 'neo-reductions','--lib','capture_recomposition_workload' if a.mode=='capture' else ('recomposition_measure' if a.mode=='measure' else 'recomposition_proofs'),'--','--ignored','--nocapture','--test-threads=1']
if a.mode=='checks':
 name='checks'
 cmd=['cargo','test','--release','-p','neo-reductions','--lib','--','--test-threads=4']
if a.target_dir: cmd[2:2]=['--target-dir',a.target_dir]
with (out/(name+'.log')).open('w') as log:
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,RAYON_NUM_THREADS='4'),start_new_session=True)
 try: proc.communicate(json.dumps({"directory":str(Path(a.directory).resolve()),"step":a.step}).encode(),timeout=300)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise SystemExit('FAIL: 300-second cap')
if proc.returncode: raise SystemExit(f'FAIL: see {out/(name+".log")}')
with (out/(name+'.jsonl')).open('w') as f:
 for line in (out/(name+'.log')).read_text().splitlines():
  if 'EXPERIMENT ' in line: f.write(line.split('EXPERIMENT ',1)[1]+'\n')
print(name,'passed')
