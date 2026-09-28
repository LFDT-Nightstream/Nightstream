from pathlib import Path
import ctypes
import json
import resource
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
binary, engine, label = sys.argv[1:]
command = [binary, '--engine', engine, '--steps', '3']
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
            if rss > 16 * 1024**3:
                outcome = 'memory-cap'
                break
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
if outcome == 'passed':
    events = []
    for line in log.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    if not any(event.get('event') == 'benchmark_finished' and event.get('verified') is True and event.get('steps') == 3 for event in events):
        outcome = 'missing-benchmark-result'
record = {
    'command': command,
    'elapsed_seconds': time.monotonic() - started,
    'exit': code, 'outcome': outcome,
    'maximum_resident_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    'maximum_physical_footprint_bytes': maximum_footprint,
    'time_cap_seconds': 300,
    'memory_cap_bytes': 16 * 1024**3,
    'memory_cap_metric': 'RSS; owner chose process residency and accepted approximately 16 GB for this pass. Working guard: 16 GiB (17.18 decimal GB).',
    'metric_authority': 'https://developer.apple.com/videos/play/wwdc2022/10106/',
    'metric_status': 'Owner confirmed RSS; footprint is diagnostic only. Further memory tuning is deferred by the owner.',
    'sample_fields': ['elapsed_seconds', 'resident_bytes', 'physical_footprint_bytes', 'lifetime_peak_footprint_bytes'],
    'samples': samples,
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_changes': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
    'source_note': 'Current worktree direct-mask three-step lifecycle benchmark; same binary and fixed inputs for both engines.',
}
(root / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key not in ('samples', 'source_changes')}), flush=True)
raise SystemExit(0 if outcome == 'passed' else 1)
