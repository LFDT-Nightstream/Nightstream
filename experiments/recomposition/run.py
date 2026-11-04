#!/usr/bin/env python3
"""Run one recomposition experiment mode.

Each cargo-test process has an unconditional 300-second process-group cap.
With --processes N, the mode runs N times in sequence, with --cooldown seconds
between runs, and all EXPERIMENT records go to one JSONL file. Each record gets
a "process" field.
"""
import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

CAP_SECONDS = 300
TESTS = {
    'capture': ('nightstream', 'capture_recomposition_workload'),
    'measure': ('neo-reductions', 'recomposition_measure'),
    'proofs': ('neo-reductions', 'recomposition_proofs'),
}

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['capture', 'measure', 'proofs', 'checks'])
parser.add_argument('--directory', type=Path,
                    default=Path(tempfile.gettempdir()) / 'nightstream-recomposition-handoff')
parser.add_argument('--target-dir')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--pairs', type=int, default=3, help='measured proof pairs per process')
parser.add_argument('--processes', type=int, default=1)
parser.add_argument('--cooldown', type=float, default=30, help='seconds between processes')
args = parser.parse_args()

output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.chdir(Path(__file__).resolve().parents[2])

if args.mode == 'checks':
    name = 'checks'
    command = ['cargo', 'test', '--release', '-p', 'neo-reductions', '--lib', '--', '--test-threads=4']
else:
    name = 'capture' if args.mode == 'capture' else f'{args.mode}-{args.step}'
    package, test = TESTS[args.mode]
    command = ['cargo', 'test', '--release', '-p', package, '--lib', test,
               '--', '--ignored', '--nocapture', '--test-threads=1']
if args.target_dir:
    command[2:2] = ['--target-dir', args.target_dir]
request = json.dumps({'directory': str(args.directory.resolve()), 'step': args.step, 'pairs': args.pairs})


def run(log_path):
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, RAYON_NUM_THREADS='4'), start_new_session=True)
        try:
            process.communicate(request.encode(), timeout=CAP_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit(f'FAIL: {CAP_SECONDS}-second cap, see {log_path}')
    if process.returncode:
        raise SystemExit(f'FAIL: see {log_path}')


records = []
for index in range(args.processes):
    if index:
        time.sleep(args.cooldown)
    log_path = output / (f'{name}.log' if args.processes == 1 else f'{name}.p{index}.log')
    run(log_path)
    for line in log_path.read_text().splitlines():
        if 'EXPERIMENT ' in line:
            record = json.loads(line.split('EXPERIMENT ', 1)[1])
            record['process'] = index
            records.append(record)
    print(f'{name} process {index} passed', flush=True)

if args.mode != 'checks':
    with (output / f'{name}.jsonl').open('w') as jsonl:
        for record in records:
            jsonl.write(json.dumps(record, sort_keys=True) + '\n')
