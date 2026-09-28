from pathlib import Path
import json
import resource
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
binary, test, label = sys.argv[1:]
command = [binary, test, '--ignored', '--exact', '--nocapture']
started = time.monotonic()
outcome = 'running'
log = root / (label + '.log')
with log.open('xb') as output:
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    while process.poll() is None:
        remaining = 300 - (time.monotonic() - started)
        if remaining <= 0:
            outcome = 'time-cap'
            process.kill()
            break
        try:
            process.wait(timeout=min(1, remaining))
        except subprocess.TimeoutExpired:
            rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True).stdout.strip()
            if rss and int(rss) * 1024 > 16_391_487_488:
                outcome = 'memory-cap'
                process.send_signal(signal.SIGSTOP)
                try:
                    with (root / (label + '-vmmap.log')).open('xb') as snapshot:
                        subprocess.run(['vmmap', '-summary', str(process.pid)], stdout=snapshot, stderr=subprocess.STDOUT, timeout=max(0, 300 - (time.monotonic() - started)))
                finally:
                    process.kill()
                break
    code = process.wait()
if outcome == 'running':
    outcome = 'passed' if code == 0 else 'failed'
if outcome == 'passed' and 'test result: ok. 1 passed; 0 failed;' not in log.read_text():
    outcome = 'missing-test-result'
record = {
    'command': command, 'elapsed_seconds': time.monotonic() - started,
    'exit': code, 'outcome': outcome,
    'maximum_resident_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    'memory_cap_authority': 'Owner accepted the observed 16,391,487,488-byte peak for now; 16 GB remains the target.',
    'time_cap_seconds': 300, 'memory_cap_bytes': 16_391_487_488,
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_changes': subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines(),
}
(root / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
raise SystemExit(0 if outcome == 'passed' else 1)
