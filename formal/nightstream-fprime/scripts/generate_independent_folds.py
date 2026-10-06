#!/usr/bin/env python3
"""Independent Lean C/R/D generation from checked original witness openings.

Every producer reads original sources or earlier Lean outputs. Native results
enter only comparison commands. Checkpoints retain exact bytes and capped command
receipts; an incomplete run is never reported as complete. Run outside graph locks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
FORMAL = REPO / "formal/nightstream-fprime"
TESTS = REPO / "crates/nightstream/tests"
sys.path[:0] = [str(TESTS), str(FORMAL / "tests")]
from check_lean_fold import Check, compare_caller, compare_source, package_pin
from project_replay_sources import BLOCKS, LOGICAL, read, require
from lean_graph.policy import CAPS

ROWS, CARRIER = 1992940, BLOCKS * 54
TOOLCHAIN = "nightstream-lean-4.32.2-3019a32c"
ARTIFACT = FORMAL / "artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


# SHA-256 and file reads release the GIL, so threads hash different files in parallel.
HASHING = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def pending_identity(path):
    """Start hashing the files under `path`; the returned function waits for `identity(path)`."""
    path = Path(path)
    require(not path.is_symlink(), f"unexpected evidence symlink: {path}")
    if path.is_dir():
        children = [(child.name, pending_identity(child)) for child in sorted(path.iterdir())]
        return lambda: {name: value() for name, value in children}
    return HASHING.submit(file_identity, path).result


def identity(path):
    return pending_identity(path)()


def start_identities(paths):
    """Start hashing a mapping `{key: path}`; the returned function waits for `{key: identity(path)}`."""
    pending = {key: pending_identity(path) for key, path in paths.items()}
    return lambda: {key: value() for key, value in pending.items()}


def identities(paths):
    return start_identities(paths)()


def matrix_geometry(package):
    """Schedule only. Lean producers/mergers check their own complete row bounds."""
    require(package[0] == 6 and package[1][4][:2] == [ROWS, LOGICAL],
            "package differs from the selected replay geometry")
    result = []
    first = 0
    for index, (tag, block) in enumerate(package[2]):
        alignment = 1
        if tag == 0:
            kind, schedule = block[0]
            require(kind in (0, 1), "unknown ordinary row schedule")
            count = sum(length for start, length in schedule) if kind == 0 else len(schedule)
        elif tag == 1:
            count = len(block[1])
        elif tag == 2:
            alignment = 150
            count = block[0] * alignment
        elif tag == 3:
            alignment = 108
            count = sum(sources * blocks * cells for sources, blocks, cells in block[0]) * alignment
        else:
            raise ValueError("unexpected selected matrix block")
        require(count > 0, "empty matrix block")
        result.append({"block": index, "tag": tag, "first": first, "count": count,
                       "alignment": alignment})
        first += count
    require(first == ROWS and len(result) == 19, "incomplete selected matrix geometry")
    return result


def cuts(size, chunk):
    return [(lo, min(size, lo + chunk)) for lo in range(0, size, chunk)]


def timed_cuts(costs, overhead, target):
    """Keep every range in order; a costly range can occupy a batch alone."""
    require(math.isfinite(overhead) and 0 <= overhead < target
            and math.isfinite(target), "invalid matrix batching budget")
    require(all(math.isfinite(cost) and cost >= 0 for cost in costs),
            "invalid matrix range timing")
    result, first, elapsed = [], 0, overhead
    for index, cost in enumerate(costs):
        if index > first and elapsed + cost > target:
            result.append((first, index))
            first, elapsed = index, overhead
        elapsed += cost
    if first < len(costs):
        result.append((first, len(costs)))
    return result


def measured_matrix_batches(ranges, logs):
    """First-fold timings guide scheduling only; they cannot establish values."""
    indices = {tuple(bounds): index for index, bounds in enumerate(ranges)}
    costs, overheads = {}, []
    for path in sorted(logs.glob("d-matrix-*.command.json")):
        record = read(path)
        require(record.get("exit") == 0 and record.get("outcome") == "passed",
                "matrix timing requires completed first-fold batches")
        subtotal = 0
        for line in path.with_name(record["name"] + ".log").read_text().splitlines():
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            if event.get("event") != "range_complete":
                continue
            bounds = (event["block"], event["first_local_row"], event["last_local_row_exclusive"])
            require(bounds in indices and indices[bounds] not in costs,
                    "unknown or duplicate matrix timing range")
            cost = event["total_ns"] / 1e9
            costs[indices[bounds]] = cost
            subtotal += cost
        overheads.append(record["elapsed_seconds"] - subtotal)
    require(len(costs) == len(ranges) and overheads, "incomplete first-fold matrix timings")
    require(all(math.isfinite(value) and value >= 0 for value in overheads),
            "invalid matrix batch overhead")
    # Half the existing hard cap leaves room for different second-fold values.
    return timed_cuts([costs[index] for index in range(len(ranges))],
                      max(overheads), CAPS["lean"] / 2)


def native_test_completed(command, log):
    return "--exact" in command and "test result: ok. 1 passed; 0 failed; 0 ignored;" in log and any(
        f"test {argument} ... ok" in log.splitlines() for argument in command[1:])


def source_snapshot(root):
    names = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
        "crates", "formal/nightstream-fprime", "scripts", "Cargo.toml", "Cargo.lock",
        "rust-toolchain.toml", ".cargo/config.toml"], cwd=REPO, text=True).splitlines()
    sources = {}
    for name in sorted(set(names)):
        path = REPO / name
        if path.is_file() and (path.suffix in (".lean", ".rs", ".py", ".sh", ".toml", ".lock")
                               or path.name in ("lean-toolchain", "lake-manifest.json")):
            sources[name] = path
    record = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
              cwd=REPO, text=True).strip(), "files": identities(sources), "package": identity(ARTIFACT),
              "lean_toolchain": TOOLCHAIN,
              "compiler_commit": "3019a32cb6f44782ff1e1210676099d683b8d3a8"}
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    directory = root / "source-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(encoded).hexdigest() + ".json")
    if path.exists():
        require(path.read_bytes() == encoded, "changed source snapshot")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    return path


def command_input_paths(argv, request, outputs, cwd):
    """Actual input files and executables; output paths are checked separately."""
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
    excluded = {path.resolve() for path in outputs}
    result = {}
    for word in strings([argv, request]):
        if "/" not in word:
            continue
        path = Path(word)
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if path not in excluded and path.exists():
            result[str(path)] = path
    return result


def command_inputs(argv, request, outputs, cwd):
    """Bind actual input files and executables; output paths are checked separately."""
    return identities(command_input_paths(argv, request, outputs, cwd))


class Replay:
    def __init__(self, root, native, iteration, binary):
        self.root, self.native, self.iteration, self.binary = root, native, iteration, binary
        self.directory = root / f"step-{iteration}-to-{iteration + 1}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logs = self.directory / "logs"
        self.logs.mkdir(exist_ok=True)
        self.check = Check(native, iteration, self.logs, binary)
        self.public = self.directory / "sources/public.json"
        self.sources = self.directory / "sources/sources.jsonl"
        self.rounds = self.directory / "rounds"
        self.rounds.mkdir(exist_ok=True)
        self.geometry = matrix_geometry(read(ARTIFACT))
        setup = read(FORMAL / "artifacts/nightstream-fprime-stage1-base-step-fixture-v1.json")
        # The same context/identity are checked again by the current native verifier.
        self.package = package_pin(REPO / "crates/nightstream-fprime/src/identity.rs")
        self.context = setup[1]

    def out(self, name):
        return self.directory / name

    def run(self, name, kind, argv, outputs=(), cwd=REPO, request=None, input_paths=None):
        argv = list(map(str, argv))
        outputs = list(map(Path, outputs))
        receipt = self.logs / f"{name}.outputs.json"
        input_receipt = self.logs / f"{name}.inputs.json"
        command_receipt = self.logs / f"{name}.command.json"
        def current_inputs():
            if input_paths is not None:
                return start_identities({str(path): path for path in input_paths})
            return start_identities(command_input_paths(argv, request, outputs, cwd))
        def current_outputs():
            return start_identities({str(path): path for path in outputs})
        def check_native_completion():
            if kind == "rust":
                require(native_test_completed(argv, (self.logs / f"{name}.log").read_text()),
                        f"native test did not complete: {name}")
        inputs = current_inputs()()
        if command_receipt.exists():
            record = read(command_receipt)
            require(record.get("exit") == 0 and record.get("outcome") == "passed",
                    f"failed checkpoint must be retained, not reused: {name}")
            require(record["command"] == argv and record["cwd"] == str(cwd)
                    and record.get("input") == request, f"changed checkpoint command: {name}")
            require(read(input_receipt) == inputs, f"changed checkpoint inputs: {name}")
            check_native_completion()
            require(read(receipt) == current_outputs()(), f"changed checkpoint outputs: {name}")
            return
        print(json.dumps({"event": "stage_started", "step": self.iteration, "stage": name}), flush=True)
        write_new(input_receipt, inputs)
        self.check.phase(name, kind, argv, cwd=cwd, input=request)
        check_native_completion()
        # The inputs are hashed again while the outputs are hashed.
        after, produced = current_inputs(), current_outputs()
        require(after() == inputs, f"inputs changed during execution: {name}")
        write_new(receipt, produced())
        if hasattr(self, "source_snapshot"):
            record = read(command_receipt)
            record["source_snapshot"] = str(self.source_snapshot)
            command_receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"event": "stage_passed", "step": self.iteration, "stage": name}), flush=True)

    def lean(self, name, executable, *args, outputs=()):
        self.run(name, "lean", ["bash", "scripts/validate.sh",
                 "lean-executable", f".lake/build/bin/{executable}", *args], outputs, cwd=FORMAL)

    def python(self, name, script, *args, outputs=()):
        self.run(name, "python", [sys.executable, "-B", FORMAL / script, *args], outputs)

    def native_check(self, name, operation, **request):
        self.run(name, "rust", [self.binary,
                 "lifecycle::tests::staged::fold::lean::golden::native_checker",
                 "--ignored", "--exact", "--nocapture"], request={"operation": operation, **request})

    def prepare(self):
        previous = self.root / "step-2-to-3"
        if self.iteration == 3:
            require((previous / "handoff.json").is_file(), "first Lean handoff is incomplete")
            handoff = read(previous / "handoff.json")
            feedback = identities({name: previous / name for name in
                                   ("fresh-witness.json", "fresh-claim.json", "children.json",
                                    "digits-0.jsonl", "digits-1.jsonl")})
            require(handoff["from_iteration"] == 2 and handoff["to_iteration"] == 3
                    and handoff["lean_feedback"] == feedback,
                    "first Lean handoff inputs changed")
        self.run("source-openings", "rust", [self.binary,
                 "lifecycle::tests::staged::run_phase", "--ignored", "--exact", "--nocapture"],
                 request={"phase": "sources", "directory": str(self.native),
                          "step": self.iteration, "engine": "optimized"},
                 input_paths=[self.binary, ARTIFACT, self.native / f"step-{self.iteration}",
                              TESTS / "fixtures/stage1_recursive_states/nonzero-running.json"])
        if self.iteration == 2:
            self.python("source-projection", "scripts/project_replay_sources.py", "original",
                        self.native / "step-2", self.out("sources"), outputs=[self.out("sources")])
        else:
            self.python("source-projection", "scripts/project_replay_sources.py", "feedback",
                        previous / "fresh-witness.json", previous / "fresh-claim.json",
                        previous / "children.json", self.out("sources"),
                        *[previous / f"digits-{i}.jsonl" for i in range(2)], outputs=[self.out("sources")])

    def matrix_ranges(self):
        ranges = []
        for block in self.geometry:
            chunk = max(block["alignment"], 131072 // block["alignment"] * block["alignment"])
            for lo, hi in cuts(block["count"], chunk):
                ranges.append((block["block"], lo, hi))
        return ranges

    def ccs(self):
        first, prefix = "replayPiCCSFirstRound", "replayPiCCSPrefix"
        u, s = self.public, self.sources
        q = [self.rounds / f"round-{index}.json" for index in range(28)]
        def component(index, kind):
            return self.out(f"q{index}-{kind}.json")
        def initial(name, mode, *args):
            output_index = {
                "norm": 1, "fresh": 1, "carried-matrix": 1, "carried-pad": 1,
                "compose": 2, "fresh-prefix": 2, "fresh-after-first": 2,
                "norm-after-first": 2, "carried-matrix-prefix": 2,
                "carried-pad-prefix": 2, "compose-second": 3,
            }[mode]
            self.lean(name, first, mode, u, *args, outputs=[args[output_index]])
        initial("q0-norm", "norm", s, component(0, "norm"), 0, BLOCKS)
        initial("q0-fresh", "fresh", s, component(0, "fresh"), 0, (ROWS + 1) // 2)
        # Fresh polynomials combine row pairs; carried moments sum individual rows.
        matrix_ranges = cuts(ROWS, 524288)
        matrix0 = []
        for index, (lo, hi) in enumerate(matrix_ranges):
            output = self.out(f"q0-matrix-{index}.json")
            initial(f"q0-matrix-{index}", "carried-matrix", s, output, lo, hi)
            matrix0.append(output)
        pad_cuts = [0, 8192, BLOCKS // 2, BLOCKS]
        pad0 = []
        for index, (lo, hi) in enumerate(zip(pad_cuts, pad_cuts[1:])):
            output = self.out(f"q0-pad-{index}.json")
            initial(f"q0-pad-{index}", "carried-pad", s, output, lo, hi)
            pad0.append(output)
        initial("q0-compose", "compose", component(0, "fresh"), component(0, "norm"),
                q[0], *matrix0, "pad", *pad0)

        fresh1 = self.out("fresh-after1")
        initial("fresh-after1", "fresh-prefix", s, q[0], fresh1, 0, (ROWS + 1) // 2)
        initial("q1-fresh", "fresh-after-first", q[0], fresh1, component(1, "fresh"))
        initial("q1-norm", "norm-after-first", s, q[0], component(1, "norm"), 0, (CARRIER + 3) // 4)
        matrix_cuts = [lo for lo, _ in cuts((ROWS + 1) // 2, 524288)] + [(ROWS + 1) // 2]
        prefixes = {"fresh": [fresh1], "matrix": [], "pad": []}
        for kind, boundaries in [("matrix", matrix_cuts), ("pad", pad_cuts)]:
            for index, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
                output = self.out(f"{kind}-after1-{index}")
                initial(f"{kind}-after1-{index}", f"carried-{kind}-prefix", s, q[0], output, lo, hi)
                prefixes[kind].append(output)
        initial("q1-compose", "compose-second", q[0], component(1, "fresh"),
                component(1, "norm"), q[1],
                *(directory / "moments.json" for directory in prefixes["matrix"]), "pad",
                *(directory / "moments.json" for directory in prefixes["pad"]))

        for kind, directories in prefixes.items():
            self.lean(f"{kind}-after2", prefix, "advance-first", kind, u, q[0], q[1],
                      self.out(f"{kind}-after2"), *directories, outputs=[self.out(f"{kind}-after2")])
            self.python(f"{kind}-after2-bytes", "tests/check_piccs_prefix_fold.py",
                        kind, q[0], q[1], self.out(f"{kind}-after2"), *directories)
        # 131072 is the completed original norm-prefix chunk extent.
        self.lean("norm-after2", prefix, "norm-codes", u, s, q[0], q[1],
                  self.out("norm-after2"), 0, (CARRIER + 3) // 4, 131072, outputs=[self.out("norm-after2")])
        self.lean("norm-after2-check", prefix, "check-norm-codes", u, s, q[0], q[1],
                  self.out("norm-after2"), 0, (CARRIER + 3) // 4, 131072)
        self.lean("q2-norm", prefix, "norm-after-two", u, q[0], q[1],
                  self.out("norm-after2"), component(2, "norm"), 0, (CARRIER + 7) // 8, 131072,
                  outputs=[component(2, "norm")])
        for index in range(2, 28):
            previous = q[:index]
            if index >= 3:
                self.lean(f"q{index}-norm", prefix, "norm-prefix", u, self.out(f"norm-after{index}"),
                          component(index, "norm"), 0, (CARRIER + (1 << (index + 1)) - 1) >> (index + 1),
                          *previous, outputs=[component(index, "norm")])
            self.lean(f"q{index}-fresh", prefix, "fresh-prefix", u, self.out(f"fresh-after{index}"),
                      component(index, "fresh"), 0, (ROWS + (1 << (index + 1)) - 1) >> (index + 1), *previous,
                      outputs=[component(index, "fresh")])
            for kind in ["matrix", "pad"]:
                self.lean(f"q{index}-{kind}", prefix, "carried-prefix", kind, u,
                          self.out(f"{kind}-after{index}"), component(index, kind), *previous,
                          outputs=[component(index, kind)])
            self.lean(f"q{index}-compose", prefix, "compose", u, component(index, "fresh"),
                      component(index, "norm"), component(index, "matrix"), component(index, "pad"),
                      q[index], *previous, outputs=[q[index]])
            extended = q[:index + 1]
            for kind in ["fresh", "matrix", "pad"]:
                self.lean(f"{kind}-after{index + 1}", prefix, "advance-prefix", kind, u,
                          self.out(f"{kind}-after{index}"), self.out(f"{kind}-after{index + 1}"), *extended,
                          outputs=[self.out(f"{kind}-after{index + 1}")])
                self.python(f"{kind}-after{index + 1}-bytes", "tests/check_piccs_binary_fold.py",
                            self.out(f"{kind}-after{index}"), self.out(f"{kind}-after{index + 1}"), q[index])
            if index == 2:
                self.lean("norm-after3", prefix, "norm-fields-after-two", u, *extended,
                          self.out("norm-after2"), self.out("norm-after3"), 131072, outputs=[self.out("norm-after3")])
                self.python("norm-after3-bytes", "tests/check_piccs_norm_field_fold.py",
                            self.out("norm-after2"), self.out("norm-after3"), *extended)
            else:
                self.lean(f"norm-after{index + 1}", prefix, "advance-norm", u,
                          self.out(f"norm-after{index}"), self.out(f"norm-after{index + 1}"), *extended,
                          outputs=[self.out(f"norm-after{index + 1}")])
                for source in range(17):
                    self.python(f"norm-after{index + 1}-source{source}-bytes", "tests/check_piccs_binary_fold.py",
                                self.out(f"norm-after{index}/source-{source}"),
                                self.out(f"norm-after{index + 1}/source-{source}"), q[index])
        matrix_ranges = [{"block": block, "local": [lo, hi]}
                         for block, lo, hi in self.matrix_ranges()]
        matrix_outputs = [self.out(f"c-matrix-{index}.json") for index in range(len(matrix_ranges))]
        batches = [range(lo, hi) for lo, hi in cuts(len(matrix_ranges), 4)]
        for batch_index, indices in enumerate(batches):
            requests, outputs = [], []
            for index in indices:
                record, output = matrix_ranges[index], matrix_outputs[index]
                requests.extend([output, record["block"], *record["local"]])
                outputs.append(output)
            self.lean(f"c-matrix-batch-{batch_index}", prefix, "original-matrix",
                      u, s, *requests, "--", *q, outputs=outputs)
        pad_outputs, pad_requests = [], []
        for lo in range(0, BLOCKS, 74272):
            hi = min(lo + 74272, BLOCKS)
            output = self.out(f"c-pad-{lo}.json")
            pad_requests.extend([output, lo, hi])
            pad_outputs.append(output)
        # Keep full 17-source evaluation in capped batches; merge checks complete coverage.
        for batch, (first, finish) in enumerate(cuts(len(pad_outputs), 16)):
            self.lean(f"c-pad-{batch}", prefix, "original-pad", u, s,
                      *pad_requests[3 * first:3 * finish], "--", *q,
                      outputs=pad_outputs[first:finish])
        self.lean("c-evaluations", prefix, "merge-original", u, self.out("c-evaluations.json"),
                  *pad_outputs, "--", *matrix_outputs, "--", *q, outputs=[self.out("c-evaluations.json")])
        self.lean("c-finish", prefix, "finish-original", u, self.out("c-evaluations.json"),
                  self.out("ccs-input.json"), self.out("ccs-phase.json"), self.out("ccs-words.json"), *q,
                  outputs=[self.out("ccs-input.json"), self.out("ccs-phase.json"), self.out("ccs-words.json")])
        observed = read(self.native / f"fold-{self.iteration}/actual_result.json")
        for name, value in [("native-ccs.input.json", observed["pi_ccs_input"]),
                            ("native-ccs.phase.json", observed["pi_ccs_phase"])]:
            if self.out(name).exists():
                require(read(self.out(name)) == value, "changed native comparison target")
            else:
                write_new(self.out(name), value)
        self.python("c-round-comparison", "tests/check_piccs_all_rounds.py", u, self.rounds,
                    self.out("native-ccs.input.json"), self.out("native-ccs.phase.json"))

        self.python("c-terminal-prefix-comparison", "tests/check_piccs_terminal_prefix.py",
                    u, q[27], self.out("fresh-after28"), self.out("norm-after28"),
                    self.out("native-ccs.input.json"), self.out("native-ccs.phase.json"))
        self.python("c-evaluation-comparison", "tests/check_piccs_original_complete.py",
                    u, q[27], self.out("c-evaluations.json"),
                    self.out("native-ccs.input.json"), self.out("native-ccs.phase.json"))
        self.python("c-complete-bytes", "tests/check_piccs_complete_bytes.py",
                    self.out("ccs-input.json"), self.out("ccs-phase.json"), self.out("ccs-words.json"),
                    self.out("native-ccs.input.json"), self.out("native-ccs.phase.json"))
        self.python("c-finish-rejections", "tests/check_piccs_finish_rejection.py",
                    FORMAL / ".lake/build/bin/replayPiCCSPrefix", u, self.out("c-evaluations.json"),
                    self.out("c-finish-rejections"), *q, outputs=[self.out("c-finish-rejections")])

    def reductions(self):
        ccs = self.out("ccs-input.json")
        parents, digits = [], []
        for index, (lo, hi) in enumerate([(0, 74272), (74272, BLOCKS)]):
            parent, digit = self.out(f"parent-{index}.jsonl"), self.out(f"digits-{index}.jsonl")
            self.lean(f"rlc-{index}", "replayPiRLCWitness", ccs, self.sources, parent, lo, hi, outputs=[parent])
            self.lean(f"digits-{index}", "replayPiDECWitness", parent, digit, lo, hi, outputs=[digit])
            parents.append(parent)
            digits.append(digit)
        native = self.native / f"fold-{self.iteration}"
        self.native_check("rlc-comparison", "parent-witness", native=str(native / "parent-witness.json"),
                          ranges=list(map(str, parents)))
        self.native_check("digits-comparison", "child-witnesses", native=str(native),
                          ranges=list(map(str, digits)))
        parts = self.out("parent-parts")
        self.python("parent-partition", "scripts/split_pidec_parent.py", parents[1], parts, 74272,
                    outputs=[parts])
        part_records = [(parents[0], 0, 74272)]
        part_records.extend((Path(record["path"]), record["start"], record["end"])
                            for record in read(parts / "manifest.json")["outputs"])
        commitments, pads = [], []
        for index, (parent, lo, hi) in enumerate(part_records):
            commitment, pad = self.out(f"d-commitment-{index}.json"), self.out(f"d-pad-{index}.json")
            self.lean(f"d-commitment-{index}", "replayPiDECCommitment", parent, commitment, lo, hi,
                      outputs=[commitment])
            self.lean(f"d-pad-{index}", "replayPiDECEvaluation", "pad", ccs, parent, pad, lo, hi, outputs=[pad])
            commitments.append(commitment)
            pads.append(pad)
        self.lean("d-commitment-merge", "replayPiDECCommitment", "merge", self.out("d-commitments.json"),
                  *commitments, outputs=[self.out("d-commitments.json")])
        self.lean("d-pad-merge", "replayPiDECEvaluation", "merge-pad", self.out("d-pad.json"),
                  *pads, outputs=[self.out("d-pad.json")])
        ranges = self.matrix_ranges()
        matrices = [self.out(f"d-matrix-{i}.json") for i in range(len(ranges))]
        batches = (cuts(len(ranges), 4) if self.iteration == 2 else
                   measured_matrix_batches(ranges, self.root / "step-2-to-3/logs"))
        for batch, (first, finish) in enumerate(batches):
            requests = [word for index in range(first, finish) for word in (matrices[index], *ranges[index])]
            self.lean(f"d-matrix-{batch}", "replayPiDECMatrix", "ranges", ccs, *requests, "--", *parents,
                      outputs=matrices[first:finish])
        self.lean("d-evaluations", "mergePiDECMatrix", ccs, self.out("d-pad.json"),
                  self.out("d-evaluations.json"), *matrices, outputs=[self.out("d-evaluations.json")])
        self.lean("d-finish", "checkPiDECInput", "from-replay", *self.package, ccs,
                  self.out("d-commitments.json"), self.out("d-evaluations.json"),
                  self.out("children.json"), self.out("nifs-result.json"),
                  outputs=[self.out("children.json"), self.out("nifs-result.json")])
        self.native_check("nifs-comparison", "compare", package=str(ARTIFACT), directory=str(native),
                          lean=str(self.out("nifs-result.json")))
        self.python("d-finish-rejections", "tests/check_pidec_finish_rejection.py",
                    FORMAL / ".lake/build/bin/checkPiDECInput", *self.package, ccs,
                    self.out("d-commitments.json"), self.out("d-evaluations.json"),
                    self.out("d-finish-rejections"), outputs=[self.out("d-finish-rejections")])

    def state_request(self):
        if self.iteration == 2:
            envelope = read(self.native / "step-2/envelope.json")
            message = read(TESTS / "fixtures/stage1_recursive_states/nonzero-running.json")[3]
            return [envelope["iteration"], envelope["z0"], envelope["current"], message]
        return read(self.root / "step-2-to-3/next-message-input.json")

    def successor(self):
        caller, request = self.out("caller.json"), self.out("message-input.json")
        if request.exists():
            require(read(request) == self.state_request(), "changed external prior state request")
        else:
            write_new(request, self.state_request())
        self.lean("caller", "emitRecursiveStepFixture", *self.context, self.out("ccs-input.json"),
                  self.out("children.json"), request, caller, outputs=[caller])
        native = self.native / f"fold-{self.iteration}"
        successor = self.native / f"step-{self.iteration + 1}"
        # These comparisons cannot provide any value to the Lean producer above.
        compare_source(read(self.out("ccs-input.json")), read(self.native / f"step-{self.iteration}/envelope.json"),
                       read(self.native / f"step-{self.iteration}/fresh-claim.json"), read(request), self.package)
        counts = compare_caller(read(native / "caller-inputs.json"), read(caller),
                                read(native / "actual_result.json"), self.context)
        physical = self.out("physical.bin")
        self.lean("physical", "replayPhysicalWitness", caller, physical, outputs=[physical])
        self.run("physical-witness-bytes", "static", ["cmp", physical, native / "physical.bin"])
        assignment = self.out("fresh-assignment")
        self.lean("assignment", "replayFreshAssignment", physical, assignment, 0, 26, outputs=[assignment])
        self.python("assignment-comparison", "tests/check_fresh_assignment_bytes.py",
                    successor / "fresh-witness.json", assignment, "--complete")
        carrier, witness = self.out("fresh-carrier.bin"), self.out("fresh-witness.json")
        self.python("assignment-assemble", "scripts/assemble_fresh_assignment.py",
                    carrier, witness, assignment, outputs=[carrier, witness])
        self.lean("canonical-rows", "checkFreshRows", carrier, caller,
                  self.out("canonical-rows.json"), outputs=[self.out("canonical-rows.json")])
        self.python("canonical-row-rejections", "tests/check_fresh_rows_rejection.py",
                    carrier, caller, self.out("canonical-row-rejections"), outputs=[self.out("canonical-row-rejections")])
        self.run("fresh-witness-bytes", "static", ["cmp", witness, successor / "fresh-witness.json"])
        commitments = []
        for index, (lo, hi) in enumerate([(0, 74272), (74272, 148544), (148544, BLOCKS)]):
            output = self.out(f"fresh-commitment-{index}.json")
            self.lean(f"fresh-commitment-{index}", "replayFreshCommitment", carrier, output, lo, hi, outputs=[output])
            commitments.append(output)
        self.lean("fresh-commitment-merge", "replayFreshCommitment", "merge",
                  self.out("fresh-commitment.json"), *commitments, outputs=[self.out("fresh-commitment.json")])
        self.python("fresh-claim-comparison", "tests/check_fresh_commitment_bytes.py",
                    self.out("fresh-commitment.json"), caller, successor / "fresh-claim.json",
                    self.out("fresh-claim.json"), outputs=[self.out("fresh-claim.json")])
        self.python("physical-rejections", "tests/check_stored_witness_rejection.py",
                    caller, physical, self.out("physical-rejections"), outputs=[self.out("physical-rejections")])
        prior, generated = read(request), read(caller)
        private, result = generated[2], generated[4]
        offset = 49393  # StateMessage's fixed current-schema field width, checked against both producers.
        require(private[28] == prior[0] and private[30:34] == prior[1] and private[35:39] == prior[2]
                and private[-4:] == prior[3] and private[offset + 28] == prior[0] + 1
                and private[offset + 30:offset + 34] == prior[1]
                and private[offset + 35:offset + 39] == result[0], "caller state handoff differs")
        next_request = [private[offset + 28], private[offset + 30:offset + 34], result[0], prior[3]]
        envelope = read(successor / "envelope.json")
        require(next_request[:3] == [envelope["iteration"], envelope["z0"], envelope["current"]],
                "independently produced successor states differ")
        destination = self.out("next-message-input.json")
        if destination.exists():
            require(read(destination) == next_request, "changed feedback state")
        else:
            write_new(destination, next_request)
        completed = self.out("handoff.json")
        record = {"from_iteration": prior[0], "to_iteration": next_request[0], "caller": counts,
                  "next_request": next_request, "lean_feedback": identities({
                      name: self.out(name) for name in
                      ("fresh-witness.json", "fresh-claim.json", "children.json", "digits-0.jsonl", "digits-1.jsonl")})}
        if completed.exists():
            require(read(completed) == record, "changed completed handoff")
        else:
            write_new(completed, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(2, 3), required=True)
    parser.add_argument("--phase", choices=("prepare", "ccs", "reductions", "successor", "all"), required=True)
    args = parser.parse_args()
    replay = Replay(args.directory.absolute(), args.native.absolute(), args.step, args.binary.resolve(strict=True))
    replay.source_snapshot = source_snapshot(replay.root)
    phases = ("prepare", "ccs", "reductions", "successor") if args.phase == "all" else (args.phase,)
    for phase in phases:
        getattr(replay, phase)()
    require(source_snapshot(replay.root) == replay.source_snapshot, "producer source changed during this invocation")


if __name__ == "__main__":
    main()
