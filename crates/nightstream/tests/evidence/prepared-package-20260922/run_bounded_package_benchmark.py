from pathlib import Path
import ctypes
import json
import resource
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
BINARY = root / 'bounded-package-benchmark-binary'
PACKAGE = root / 'bounded-poseidon2.nsc'
SCOPE = 'prepared_package_load_prove_verify'
binary, engine, label = sys.argv[1:]
assert Path(binary).resolve() == BINARY and engine in ('optimized', 'metal')
assert Path(label).name == label and PACKAGE.is_file(), 'Expected the bounded-poseidon2.nsc package.'
command = [str(BINARY), 'run', '--package', str(PACKAGE), '--engine', engine, '--steps', '3']


def package_snapshot():
    info = PACKAGE.stat()
    return {'path': str(PACKAGE), 'device': info.st_dev, 'inode': info.st_ino,
            'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


package_before = package_snapshot()
observer = ctypes.CDLL(str(root / 'libmemory-observer.dylib'), use_errno=True)
observer.memory_usage.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
observer.child_exited.argtypes = [ctypes.c_int, ctypes.c_int]
words = (ctypes.c_uint64 * 3)()
started = time.monotonic()
outcome = 'running'
samples = []
maximum_footprint = 0
log = root / (label + '.log')
with log.open('xb') as output, (root / (label + '-memory-samples.jsonl')).open('x', buffering=1) as sample_log:
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
            sample_log.write(json.dumps(samples[-1]) + '\n')
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
    starts = [event for event in events if event.get('event') == 'benchmark_started']
    finishes = [event for event in events if event.get('event') == 'benchmark_finished']
    if not (len(starts) == len(finishes) == 1
            and starts[0].get('schema') == finishes[0].get('schema') == 2
            and starts[0].get('timing_scope') == finishes[0].get('timing_scope') == SCOPE
            and starts[0].get('package') == str(PACKAGE) and starts[0].get('engine') == engine
            and starts[0].get('steps') == finishes[0].get('steps') == 3
            and finishes[0].get('verified') is True):
        outcome = 'missing-benchmark-result'
maximum_resident = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if outcome == 'passed' and maximum_resident > 16 * 1024**3:
    outcome = 'memory-cap'
try:
    package_after = package_snapshot()
except OSError:
    package_after = None
if outcome == 'passed' and package_before != package_after:
    outcome = 'package-changed'
record = {
    'command': command,
    'timing_scope': SCOPE,
    'one_time_compile_included': False,
    'prepared_package_load_included': True,
    'terminal_verification_included': True,
    'preparation_time_subtracted': False,
    'directly_comparable_to_historical_compile_inclusive_results': False,
    'package_before': package_before, 'package_after': package_after,
    'package_unchanged': package_before == package_after,
    'package_scope': 'Same fixed path and unchanged file metadata before/after; this is not content authentication.',
    'elapsed_seconds': time.monotonic() - started,
    'exit': code, 'outcome': outcome,
    'maximum_resident_bytes': maximum_resident,
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
    'source_note': 'Prepared-package load, three proof steps and terminal verification; compilation and writing occur in a separate process. Historical schema1 results include preparation.',
}
(root / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key not in ('samples', 'source_changes')}), flush=True)
raise SystemExit(0 if outcome == 'passed' else 1)
