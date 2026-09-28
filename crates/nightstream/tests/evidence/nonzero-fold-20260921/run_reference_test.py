from pathlib import Path
import ctypes
import json
import resource
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
binary, test, label = sys.argv[1:]
command = [binary, test, '--ignored', '--exact', '--nocapture']
observer = ctypes.CDLL(str(root / 'libmemory-observer.dylib'), use_errno=True)
observer.memory_usage.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
observer.child_exited.argtypes = [ctypes.c_int, ctypes.c_int]
words = (ctypes.c_uint64 * 3)()
started = time.monotonic()
outcome = 'running'
samples = []
maximum_footprint = 0
log = root / (label + '.log')
with log.open('xb') as output:
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    try:
        while True:
            finished = observer.child_exited(process.pid, 0)
            if finished < 0 or observer.memory_usage(process.pid, words) != 0:
                outcome = 'memory-observation-failed'
                break
            rss, footprint, lifetime_peak = words
            maximum_footprint = max(maximum_footprint, lifetime_peak)
            samples.append([time.monotonic() - started, rss, footprint, lifetime_peak])
            if finished:
                break
            remaining = 300 - (time.monotonic() - started)
            if remaining <= 0:
                outcome = 'time-cap'
                break
            time.sleep(min(1, remaining))
    finally:
        if outcome != 'running':
            process.kill()
        code = process.wait()
if outcome == 'running':
    outcome = 'passed' if code == 0 else 'failed'
if outcome == 'passed' and 'test result: ok. 1 passed; 0 failed;' not in log.read_text():
    outcome = 'missing-test-result'
record = {
    'command': command,
    'elapsed_seconds': time.monotonic() - started,
    'exit': code, 'outcome': outcome,
    'maximum_resident_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    'maximum_physical_footprint_bytes': maximum_footprint,
    'time_cap_seconds': 300,
    'memory_cap_bytes': None,
    'memory_cap_metric': 'RSS recorded without cutoff for the one owner-approved CPU reference run.',
    'metric_authority': 'https://developer.apple.com/videos/play/wwdc2022/10106/',
    'metric_status': 'Owner explicitly approved this one CPU reference run above the approximate 16 GB RSS limit; 300-second timeout unchanged. This is not a production memory pass.',
    'sample_fields': ['elapsed_seconds', 'resident_bytes', 'physical_footprint_bytes', 'lifetime_peak_footprint_bytes'],
    'samples': samples,
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_changes': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
    'source_note': 'Current worktree full-fold test; no temporary pressure relief or memory snapshot code.',
}
(root / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key not in ('samples', 'source_changes')}), flush=True)
raise SystemExit(0 if outcome == 'passed' else 1)
