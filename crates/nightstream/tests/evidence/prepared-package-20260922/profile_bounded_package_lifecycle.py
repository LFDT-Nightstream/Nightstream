from pathlib import Path
import ctypes
import datetime
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CAP = 1800  # AGENTS.md: includes the target and trace finalization.
RSS_CAP = 16 * 1024**3  # Existing process RSS guard for this pass.
BINARY = ROOT / 'bounded-package-benchmark-binary'
PACKAGE = ROOT / 'bounded-poseidon2.nsc'
SCOPE = 'prepared_package_load_prove_verify'


def package_snapshot():
    info = PACKAGE.stat()
    return {'path': str(PACKAGE), 'device': info.st_dev, 'inode': info.st_ino,
            'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def ps(pid, field):
    result = subprocess.run(['ps', '-p', str(pid), '-o', field + '='], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def target_matches(state):
    expected = state['target_command']
    matches = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1] == expected:
            matches.append(int(fields[0]))
    return matches


def stop_owned(state, recorder_signal):
    target = state.get('target_pid')
    if target is None:
        matches = target_matches(state)
        target = matches[0] if len(matches) == 1 else None
    stopped = []
    if target is not None and ps(target, 'command') == state['target_command']:
        expected_start = state.get('target_start')
        if expected_start is None or ps(target, 'lstart') == expected_start:
            try:
                os.kill(target, signal.SIGKILL)
                stopped.append(target)
            except ProcessLookupError:
                pass
    recorder = state.get('recorder_pid')
    command = ps(recorder, 'command') if recorder else None
    if command and state['trace_path'] in command and 'xctrace' in command:
        try:
            os.kill(recorder, recorder_signal)
            stopped.append(recorder)
        except ProcessLookupError:
            pass
    return stopped


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def watch(path):
    outcome = 'processes-ended'
    stopped = []
    while True:
        state = json.loads(path.read_text())
        target = state.get('target_pid')
        if target is None:
            matches = target_matches(state)
            target = matches[0] if len(matches) == 1 else None
        target_live = target is not None and ps(target, 'command') == state['target_command']
        if target_live and state.get('target_start') is not None:
            target_live = ps(target, 'lstart') == state['target_start']
        if not target_live and (state.get('recorder_finished') or not alive(state.get('recorder_pid'))):
            break
        if time.time() >= state['deadline_unix_seconds']:
            outcome = 'instrument-time-cap'
            stopped = stop_owned(state, signal.SIGKILL)
            break
        if not alive(state['controller_pid']):
            outcome = 'controller-ended-with-live-profile'
            stopped = stop_owned(state, signal.SIGKILL)
            break
        time.sleep(min(1, max(0, state['deadline_unix_seconds'] - time.time())))
    save(ROOT / (state['label'] + '-deadline.json'), {
        'outcome': outcome, 'stopped_pids': stopped,
        'controller_pid': state['controller_pid'], 'target_pid': target,
        'recorder_pid': state['recorder_pid'],
        'deadline_unix_seconds': state['deadline_unix_seconds'],
        'cap_seconds': CAP, 'authority': state['time_authority'],
    })


class TaskInfo(ctypes.Structure):
    _fields_ = [('virtual_size', ctypes.c_uint64), ('resident_size', ctypes.c_uint64),
                ('resident_size_max', ctypes.c_uint64), ('user_seconds', ctypes.c_int),
                ('user_micros', ctypes.c_int), ('system_seconds', ctypes.c_int),
                ('system_micros', ctypes.c_int), ('policy', ctypes.c_int),
                ('suspend_count', ctypes.c_int)]


def run(engine, label):
    assert engine in ('optimized', 'metal') and Path(label).name == label
    assert PACKAGE.is_file(), 'Expected the bounded-poseidon2.nsc package.'
    package_before = package_snapshot()
    command = ['xcrun', 'xctrace', 'record', '--template', 'Time Profiler',
               '--time-limit', '1800s', '--output', str(ROOT / (label + '.trace')),
               '--target-stdout', str(ROOT / (label + '-benchmark.log')),
               '--launch', '--', str(BINARY), 'run', '--package', str(PACKAGE), '--engine', engine, '--steps', '3']
    controller = os.getppid()
    controller_command = ps(controller, 'command')
    assert controller_command and 'timeout --signal=KILL 1800' in controller_command, controller_command
    controller_start = ps(controller, 'lstart')
    deadline = datetime.datetime.strptime(controller_start, '%a %b %d %H:%M:%S %Y').timestamp() + CAP
    state = {'label': label, 'engine': engine, 'controller_pid': controller,
             'controller_start': controller_start, 'controller_command': controller_command,
             'driver_pid': os.getpid(), 'deadline_unix_seconds': deadline,
             'time_authority': 'AGENTS.md30-minute Instruments cap; ps start is rounded down, never extending the cap.',
             'target_command': str(BINARY) + ' run --package ' + str(PACKAGE) + ' --engine ' + engine + ' --steps 3',
             'package': str(PACKAGE),
             'trace_path': str(ROOT / (label + '.trace')), 'target_pid': None,
             'target_start': None, 'recorder_finished': False}
    assert not target_matches(state), 'An identical target is already live.'
    assert not (ROOT / (label + '.trace')).exists(), 'Do not overwrite a trace.'
    assert not (ROOT / (label + '.json')).exists(), 'Do not overwrite a run.'
    state_path = ROOT / (label + '-state.json')
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    lib.mach_task_self.restype = ctypes.c_uint
    lib.task_name_for_pid.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
    lib.task_info.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    owner, port = lib.mach_task_self(), ctypes.c_uint()
    observer = ctypes.CDLL(str(ROOT / 'libmemory-observer.dylib'), use_errno=True)
    observer.memory_usage.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    words = (ctypes.c_uint64 * 3)()
    started_wall, started_mono = time.time(), time.monotonic()
    samples, ended, outcome, error = [], False, 'running', None
    final_peak_after_exit = None
    with (ROOT / (label + '-xctrace.log')).open('xb') as output:
        recorder = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        state['recorder_pid'] = recorder.pid
        save(state_path, state)
        with (ROOT / (label + '-watchdog.log')).open('xb') as watch_log:
            watchdog = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watch', str(state_path)],
                                        stdout=watch_log, stderr=subprocess.STDOUT, start_new_session=True)
        state['watchdog_pid'] = watchdog.pid
        save(state_path, state)
        print(json.dumps(state), flush=True)
        try:
            while recorder.poll() is None:
                if time.time() >= deadline:
                    outcome = 'instrument-time-cap'
                    stop_owned(state, signal.SIGKILL)
                    break
                if state['target_pid'] is None:
                    matches = target_matches(state)
                    if len(matches) > 1:
                        raise RuntimeError('Native target PID is ambiguous')
                    if matches:
                        state['target_pid'] = matches[0]
                        state['target_start'] = ps(matches[0], 'lstart')
                        save(state_path, state)
                        status = lib.task_name_for_pid(owner, matches[0], ctypes.byref(port))
                        if status:
                            raise RuntimeError(f'task_name_for_pid failed: {status}')
                        print(json.dumps({'target_pid': matches[0], 'target_start': state['target_start']}), flush=True)
                if state['target_pid'] is not None and not ended:
                    info = TaskInfo()
                    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
                    status = lib.task_info(port.value, 20, ctypes.byref(info), ctypes.byref(count))
                    footprint_status = observer.memory_usage(state['target_pid'], words)
                    process_status = None if status == 0 and footprint_status == 0 else ps(state['target_pid'], 'stat')
                    if status == 0:
                        samples.append([time.monotonic() - started_mono, info.resident_size,
                                        info.resident_size_max, words[1] if footprint_status == 0 else None,
                                        words[2] if footprint_status == 0 else None])
                        if info.resident_size_max > RSS_CAP:
                            outcome, ended = 'memory-cap', True
                            stop_owned(state, signal.SIGINT)
                    if status or footprint_status:
                        if process_status is None or process_status.startswith('Z'):
                            ended = True
                            if status == 0:
                                final_peak_after_exit = info.resident_size_max
                        else:
                            raise RuntimeError(f'Live memory observer failed: task={status}, footprint={footprint_status}')
                time.sleep(min(1, max(0, deadline - time.time())))
        except BaseException as caught:
            outcome, error = 'observer-error', repr(caught)
            stop_owned(state, signal.SIGKILL)
        finally:
            if recorder.poll() is None:
                stop_owned(state, signal.SIGKILL)
            code = recorder.wait()
            # A failed recorder may leave its separately grouped launch target alive.
            remaining_targets = target_matches(state)
            if remaining_targets:
                if outcome == 'running':
                    outcome = 'recorder-ended-with-live-target'
                if state['target_pid'] is None and len(remaining_targets) == 1:
                    state['target_pid'] = remaining_targets[0]
                    state['target_start'] = ps(remaining_targets[0], 'lstart')
                stop_owned(state, signal.SIGKILL)
            if port.value:
                lib.mach_port_deallocate(owner, port.value)
            state['recorder_finished'] = True
            save(state_path, state)
            watchdog.wait()
    events = []
    path = ROOT / (label + '-benchmark.log')
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    starts = [event for event in events if event.get('event') == 'benchmark_started']
    finishes = [event for event in events if event.get('event') == 'benchmark_finished']
    verified = (len(starts) == len(finishes) == 1
                and starts[0].get('schema') == finishes[0].get('schema') == 2
                and starts[0].get('timing_scope') == finishes[0].get('timing_scope') == SCOPE
                and starts[0].get('package') == str(PACKAGE) and starts[0].get('engine') == engine
                and starts[0].get('steps') == finishes[0].get('steps') == 3
                and finishes[0].get('verified') is True)
    if outcome == 'running':
        outcome = 'passed' if code == 0 and state['target_pid'] is not None and verified else 'failed'
    try:
        package_after = package_snapshot()
    except OSError:
        package_after = None
    if outcome == 'passed' and package_before != package_after:
        outcome = 'package-changed'
    record = {'command': command, 'state': state, 'outcome': outcome, 'error': error, 'exit': code,
              'one_time_compile_included': False, 'prepared_package_load_included': True,
              'terminal_verification_included': True, 'preparation_time_subtracted': False,
              'directly_comparable_to_historical_compile_inclusive_results': False,
              'package_before': package_before, 'package_after': package_after,
              'package_unchanged': package_before == package_after,
              'package_scope': 'Same fixed path and unchanged file metadata before/after; this is not content authentication.',
              'verified': verified, 'elapsed_monotonic_seconds': time.monotonic() - started_mono,
              'elapsed_wall_seconds': time.time() - started_wall, 'time_cap_seconds': CAP,
              'memory_guard_bytes': RSS_CAP,
              'observed_kernel_peak_resident_bytes': max((s[2] for s in samples), default=0),
              'peak_physical_footprint_bytes': max((s[4] or 0 for s in samples), default=0),
              'final_kernel_peak_after_exit': final_peak_after_exit,
              'rss_scope': 'Kernel lifetime peak observed while live; if final peak is null, the final unsampled interval is not proved.',
              'timing_scope': SCOPE,
              'instrumentation_scope': 'Prepared-package load, proof lifecycle and terminal verification under Time Profiler; no controlled ratio against raw runs or historical compile-inclusive runs.',
              'power_control': 'Launched with command-scoped caffeinate -is on AC; no permanent setting change.',
              'sample_fields': ['elapsed_seconds','resident_bytes','kernel_peak_resident_bytes','physical_footprint_bytes','lifetime_peak_footprint_bytes'],
              'samples': samples, 'events': events}
    save(ROOT / (label + '.json'), record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('samples','events')}), flush=True)
    raise SystemExit(0 if outcome == 'passed' else 1)


if __name__ == '__main__':
    if sys.argv[1] == '--watch':
        watch(Path(sys.argv[2]))
    else:
        run(*sys.argv[1:])
