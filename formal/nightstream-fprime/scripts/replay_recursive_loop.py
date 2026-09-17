#!/usr/bin/env python3
"""Run one fresh recursive replay checkpoint through the shared command queue.

Each stage uses the guard; owner authorization can disable its deadline.
This coordinator is not a verifier:
the existing Lean/Rust checks remain the authorities for the values they check.
Large outputs and command records belong in the supplied external run directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
FORMAL = REPO / "formal/nightstream-fprime"
REVIEW = REPO / "docs/reviews/nightstream-fprime-requirements"
BLOCKS, ROWS, CARRIER = 4685394, 6377559, 253011276
PACKAGE = ["9705822157724451396", "520958727644325895",
           "9285622073986934000", "874020794279380938"]
CONTEXT = ["18363630987318625048", "9406776669274472459",
           "1104198490699942438", "1757792822492309855"]


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


import hashlib

def identity(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"unexpected output symlink: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"kind": "file", "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if path.is_dir():
        return {"kind": "directory", "entries": {
            child.name: identity(child) for child in sorted(path.iterdir())}}
    raise ValueError(f"missing checkpoint output: {path}")


def check_outputs(record):
    outputs = record["outputs"]
    if not all(Path(path).exists() for path in outputs):
        raise ValueError("missing checkpoint output")
    actual = {path: identity(path) for path in outputs}
    if record.get("output_identities") != actual:
        raise ValueError("changed checkpoint output bytes or directory members")


def check_saved_outputs(root):
    for path in sorted(root.glob("step-*-to-*/logs/*.json")):
        record = read(path)
        if "outputs" in record and record.get("exit") == 0:
            check_outputs(record)


def producer_sources():
    paths = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
        "crates", "formal/nightstream-fprime", "scripts/lean_graph",
        "Cargo.toml", "Cargo.lock"], cwd=REPO, text=True).splitlines()
    result = {}
    for name in sorted(set(paths)):
        path = REPO / name
        if path.suffix in (".rs", ".lean", ".py", ".sh", ".toml", ".lock") or path.name in ("lean-toolchain", "lake-manifest.json"):
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # The explicit source loader and the legacy terminal helper use the same
    # package bytes. Pin both paths because the latter chooses its path itself.
    artifact = "formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
    result[artifact] = hashlib.sha256((REPO / artifact).read_bytes()).hexdigest()
    for name in ("PICCS_FIRST_ROUND_REPLAY.json", "PIDEC_MATRIX_RANGES.json"):
        result[str((REVIEW / name).relative_to(REPO))] = hashlib.sha256((REVIEW / name).read_bytes()).hexdigest()
    return result


def pin_sources(root):
    path = root / "producer-sources.json"
    sources = producer_sources()
    sources["original-sources"] = identity(root / "original-sources")
    sources["original-package"] = identity(root / "original-package.json")
    if selected := os.environ.get("LEAN_SYSROOT"):
        sysroot = Path(selected).resolve(strict=True)
        sources["selected-lean-runtime"] = {
            "path": str(sysroot),
            "files": {name: identity(sysroot / name) for name in
                      ("bin/lean", "lib/lean/libleanshared.so", "lib/lean/libleanrt.a")}}
    if path.exists():
        if read(path)["files"] != sources:
            raise ValueError("producer sources changed since this fresh run began")
    else:
        write_new(path, {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                         "files": sources, "scope": "Pinned executable source/input bytes; digests are custody records, not protocol authority."})


class Replay:
    def __init__(self, root, iteration, no_timeout=False):
        self.no_timeout = no_timeout
        self.root = root.resolve(strict=True)
        pin_sources(self.root)
        check_saved_outputs(self.root)
        self.directory = self.root / f"step-{iteration}-to-{iteration + 1}"
        self.directory.mkdir(exist_ok=True)
        self.logs = self.directory / "logs"
        self.logs.mkdir(exist_ok=True)
        self.iteration = iteration
        self.projection = self.directory / "sources"
        self.public = self.projection / "public.json"
        self.sources = self.projection / "sources.jsonl"
        self.rounds = self.directory / "rounds"
        self.rounds.mkdir(exist_ok=True)
        self.package = self.root / "original-package.json"
        self.native_sources = (self.root / "original-sources" if iteration == 2 else
                               self.root / "step-2-to-3/native-successor")
        self.request = ((self.root / "original-sources") if iteration == 2 else (self.root / "step-2-to-3")) / "next-message-input.json"

    def out(self, name):
        return self.directory / name

    def run(self, name, kind, cwd, argv, outputs=()):
        argv = [str(word) for word in argv]
        record = self.logs / f"{name}.json"
        if record.exists():
            saved = read(record)
            if (saved["argv"] != argv or saved["kind"] != kind or saved["cwd"] != str(cwd) or saved["exit"] != 0
                    or saved["outputs"] != [str(path) for path in outputs]):
                raise ValueError(f"changed or failed checkpoint: {record}")
            check_outputs(saved)
            return
        cap = None if self.no_timeout else (1500 if kind == "lean" else 300)
        command = ["/usr/bin/time", "-v"]
        if cap is not None:
            command += ["timeout", "--signal=TERM", f"{cap}s"]
        command += ["python3", "-B", str(REPO / "scripts/lean_graph/guard.py"),
                    "--kind", kind, "--cwd", str(cwd)]
        if self.no_timeout:
            command.append("--no-timeout")
        command += ["--", *argv]
        started = time.time()
        print(json.dumps({"event": "stage_started", "stage": name, "cap_seconds": cap}), flush=True)
        with (self.logs / f"{name}.log").open("x") as log:
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        write_new(record, {"argv": argv, "kind": kind, "cwd": str(cwd),
                           "started_unix": started, "elapsed_seconds": time.time() - started,
                           "exit": result.returncode, "cap_seconds": cap,
                           "outputs": [str(path) for path in outputs],
                           "output_identities": ({str(path): identity(path) for path in outputs}
                               if result.returncode == 0 and all(Path(path).exists() for path in outputs) else None)})
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed: inspect {self.logs / (name + '.log')}")
        if not all(Path(path).exists() for path in outputs):
            raise RuntimeError(f"{name} did not produce its required outputs")
        print(json.dumps({"event": "stage_passed", "stage": name}), flush=True)

    def lean(self, name, executable, *args, outputs=()):
        self.run(name, "lean", FORMAL,
                 ["bash", "scripts/validate.sh", "lean-executable",
                  f".lake/build/bin/{executable}", *args], outputs)

    def python(self, name, script, *args, outputs=()):
        self.run(name, "python", REPO, ["python3", "-B", FORMAL / script, *args], outputs)

    def rust(self, name, *args, outputs=()):
        self.run(name, "rust", REPO,
                 ["cargo", "run", "--locked", "-p", "neo-fold-clean", "--release",
                  "--features", "perf-timers", "--bin", "generate_pi_ccs_fixture", "--", *args],
                 outputs)

    def build(self):
        # Keep compilation outside the timed source and arithmetic commands.
        for executable in (
                "replayPiCCSFirstRound", "replayPiCCSPrefix", "replayPiRLCWitness",
                "replayPiDECWitness", "replayPiDECCommitment", "replayPiDECEvaluation",
                "replayPiDECMatrix", "mergePiDECMatrix", "checkPiDECInput",
                "emitRecursiveStepFixture", "replayPhysicalWitness", "replayFreshAssignment",
                "checkFreshRows", "replayFreshCommitment"):
            self.run("build-" + executable, "lean", FORMAL,
                     ["bash", "scripts/validate.sh", "build", executable],
                     outputs=[FORMAL / ".lake/build/bin" / executable])

    def prepare(self):
        if self.iteration == 2:
            self.python("source-projection", "scripts/project_replay_sources.py",
                        "original", self.native_sources, self.projection,
                        outputs=[self.public, self.sources])
        else:
            previous = self.root / "step-2-to-3"
            self.python("source-projection", "scripts/project_replay_sources.py",
                        "feedback", previous / "fresh-witness.json", previous / "fresh-claim.json",
                        previous / "children.json", self.projection,
                        previous / "digits-0.jsonl", previous / "digits-1.jsonl",
                        outputs=[self.public, self.sources])

    def native(self):
        ccs, parent, material = self.out("native-ccs.json"), self.out("native-parent"), self.out("native-material")
        self.rust("native-sources", "check-owned-sources", self.package, self.native_sources)
        self.rust("native-ccs", "prove-owned-ccs", self.package, self.native_sources, ccs, outputs=[ccs, ccs.with_suffix(".input.json"), ccs.with_suffix(".phase.json")])
        self.rust("native-rlc", "prove-owned-rlc", self.package, self.native_sources, ccs, parent, outputs=[parent])
        self.rust("native-digits", "check-owned-parent", self.package, self.native_sources, parent, material, outputs=[material])
        openings = self.out("native-openings")
        openings.mkdir(exist_ok=True)
        active = read(material / "split.json")["nonzero"]
        if len(active) != 16 or any(type(flag) is not bool for flag in active):
            raise ValueError("invalid newly computed child activity")
        for child, live in enumerate(active):
            if live:
                output = openings / f"child-{child}.json"
                self.rust(f"native-opening-{child}", "open-owned-child", self.package,
                          self.native_sources, parent, material, child, output, outputs=[output])
        self.rust("native-nifs", "assemble-owned-nifs", self.package, self.native_sources,
                  parent, material, openings, self.out("native-nifs"), outputs=[self.out("native-nifs")])

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
        # Retain the measured contiguous geometry, not any saved contribution.
        matrix_ranges = [record["rows"] for record in
                         read(REVIEW / "PICCS_FIRST_ROUND_REPLAY.json")["matrix_ranges"]]
        matrix0 = []
        for index, (lo, hi) in enumerate(matrix_ranges):
            output = self.out(f"q0-matrix-{index}.json")
            initial(f"q0-matrix-{index}", "carried-matrix", s, output, lo, hi)
            matrix0.append(output)
        # The prior full Pad command took 1368s. The next source has more active
        # witnesses; reuse the measured prefix cut to keep each command capped.
        pad_cuts = [0, 8192, 2346793, BLOCKS]
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
        matrix_cuts = [0, 16384, 580450, 1518288, 1931726, 2579729, (ROWS + 1) // 2]
        legacy = {"fresh": [fresh1], "matrix": [], "pad": []}
        for kind, cuts in [("matrix", matrix_cuts), ("pad", pad_cuts)]:
            for index, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
                output = self.out(f"{kind}-after1-{index}")
                initial(f"{kind}-after1-{index}", f"carried-{kind}-prefix", s, q[0], output, lo, hi)
                legacy[kind].append(output)
        initial("q1-compose", "compose-second", q[0], component(1, "fresh"),
                component(1, "norm"), q[1],
                *(directory / "moments.json" for directory in legacy["matrix"]), "pad",
                *(directory / "moments.json" for directory in legacy["pad"]))

        for kind, directories in legacy.items():
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
        matrix_ranges = read(REVIEW / "PIDEC_MATRIX_RANGES.json")["ranges"]
        matrix_outputs = [self.out(f"c-matrix-{index}.json") for index in range(len(matrix_ranges))]
        # The complete original-matrix record measured these same request groups
        # below the 1500s cap. Only grouping is reused; every value is new.
        batches = [(0, 21, 27), (1,), range(2, 6), range(6, 9), range(9, 11),
                   (11, 12), range(13, 21), range(22, 27), (28,), range(29, 34),
                   range(34, 37), range(37, 53)]
        if sorted(index for batch in batches for index in batch) != list(range(len(matrix_ranges))):
            raise ValueError("original matrix batch geometry changed")
        # With no deadline, reuse one source load and retain every request.
        if self.no_timeout:
            batches = [tuple(index for batch in batches for index in batch)]
        for batch_index, indices in enumerate(batches):
            requests, outputs = [], []
            for index in indices:
                record, output = matrix_ranges[index], matrix_outputs[index]
                requests.extend([output, record["block"], *record["local"]])
                outputs.append(output)
            self.lean(f"c-matrix-batch-{batch_index}", prefix, "original-matrix",
                      u, s, *requests, "--", *q, outputs=outputs)
        pad_outputs, pad_requests = [], []
        # Full original Pad took 847.30s with these 64 requests and one source load.
        for lo in range(0, BLOCKS, 74272):
            hi = min(lo + 74272, BLOCKS)
            output = self.out(f"c-pad-{lo}.json")
            pad_requests.extend([output, lo, hi])
            pad_outputs.append(output)
        self.lean("c-pad-all", prefix, "original-pad", u, s, *pad_requests, "--", *q, outputs=pad_outputs)
        self.lean("c-evaluations", prefix, "merge-original", u, self.out("c-evaluations.json"),
                  *pad_outputs, "--", *matrix_outputs, "--", *q, outputs=[self.out("c-evaluations.json")])
        self.lean("c-finish", prefix, "finish-original", u, self.out("c-evaluations.json"),
                  self.out("ccs-input.json"), self.out("ccs-phase.json"), self.out("ccs-words.json"), *q,
                  outputs=[self.out("ccs-input.json"), self.out("ccs-phase.json"), self.out("ccs-words.json")])
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
        self.rust("rlc-comparison", "compare-pirlc-replay",
                  self.out("native-parent/parent-witness.json"), *parents)
        self.rust("digits-comparison", "compare-pidec-replay", self.out("native-material"), *digits)
        parts = self.out("parent-parts")
        self.python("parent-partition", "scripts/split_pidec_parent.py",
                    parents[1], parts, 74272, outputs=[parts / "manifest.json"])
        part_records = [(parents[0], 0, 74272)]
        for record in read(parts / "manifest.json")["outputs"]:
            part_records.append((Path(record["path"]), record["start"], record["end"]))
        commitments, pads = [], []
        for index, (parent, lo, hi) in enumerate(part_records):
            commitment, pad = self.out(f"d-commitment-{index}.json"), self.out(f"d-pad-{index}.json")
            self.lean(f"d-commitment-{index}", "replayPiDECCommitment", parent, commitment, lo, hi, outputs=[commitment])
            self.lean(f"d-pad-{index}", "replayPiDECEvaluation", "pad", ccs, parent, pad, lo, hi, outputs=[pad])
            commitments.append(commitment)
            pads.append(pad)
        self.lean("d-commitment-merge", "replayPiDECCommitment", "merge",
                  self.out("d-commitments.json"), *commitments, outputs=[self.out("d-commitments.json")])
        self.lean("d-pad-merge", "replayPiDECEvaluation", "merge-pad", self.out("d-pad.json"),
                  *pads, outputs=[self.out("d-pad.json")])
        matrix_ranges = read(REVIEW / "PIDEC_MATRIX_RANGES.json")["ranges"]
        matrices = [self.out(f"d-matrix-{index}.json") for index in range(len(matrix_ranges))]
        batches = {}
        # Reuse the previously capped D request groups, including both tail
        # batches, so each original parent is loaded once per measured group.
        for index, record in enumerate(matrix_ranges):
            batches.setdefault(record["log"]["path"], []).append(index)
        groups = list(batches.values())
        if self.no_timeout:
            groups = [[index for group in groups for index in group]]
        for batch_index, indices in enumerate(groups):
            requests, outputs = [], []
            for index in indices:
                record, output = matrix_ranges[index], matrices[index]
                requests.extend([output, record["block"], *record["local"]])
                outputs.append(output)
            self.lean(f"d-matrix-batch-{batch_index}", "replayPiDECMatrix", "ranges", ccs,
                      *requests, "--", *parents, outputs=outputs)
        self.lean("d-evaluations", "mergePiDECMatrix", ccs, self.out("d-pad.json"),
                  self.out("d-evaluations.json"), *matrices, outputs=[self.out("d-evaluations.json")])
        self.lean("d-finish", "checkPiDECInput", "from-replay", *PACKAGE, ccs,
                  self.out("d-commitments.json"), self.out("d-evaluations.json"),
                  self.out("children.json"), self.out("nifs-result.json"),
                  outputs=[self.out("children.json"), self.out("nifs-result.json")])
        self.rust("nifs-comparison", "check-owned-nifs", self.package,
                  self.out("native-nifs"), self.out("nifs-result.json"))
        self.python("d-finish-rejections", "tests/check_pidec_finish_rejection.py",
                    FORMAL / ".lake/build/bin/checkPiDECInput", *PACKAGE, ccs,
                    self.out("d-commitments.json"), self.out("d-evaluations.json"),
                    self.out("d-finish-rejections"), outputs=[self.out("d-finish-rejections")])

    def successor(self):
        caller = self.out("caller.json")
        self.lean("caller", "emitRecursiveStepFixture", *CONTEXT, self.out("ccs-input.json"),
                  self.out("children.json"), self.request, caller, outputs=[caller])
        successor = self.out("native-successor")
        self.rust("native-successor", "complete-later-envelope", self.native_sources,
                  self.out("native-nifs"), caller, self.out("native-material"), successor,
                  outputs=[successor / "envelope.json", successor / "fresh-claim.json",
                           successor / "fresh-witness.json", successor / "physical.bin"])
        physical = self.out("physical.bin")
        self.lean("physical", "replayPhysicalWitness", caller, physical, outputs=[physical])
        self.run("physical-witness-bytes", "static", REPO,
                 ["cmp", physical, successor / "physical.bin"])
        assignment = self.out("fresh-assignment")
        self.lean("assignment", "replayFreshAssignment", physical, assignment, 0, 30, outputs=[assignment])
        self.python("assignment-comparison", "tests/check_fresh_assignment_bytes.py",
                    successor / "fresh-witness.json", assignment, "--complete")
        carrier, witness = self.out("fresh-carrier.bin"), self.out("fresh-witness.json")
        self.python("assignment-assemble", "scripts/assemble_fresh_assignment.py",
                    carrier, witness, assignment, outputs=[carrier, witness])
        self.lean("canonical-rows", "checkFreshRows", carrier, caller,
                  self.out("canonical-rows.json"), outputs=[self.out("canonical-rows.json")])
        self.python("canonical-row-rejections", "tests/check_fresh_rows_rejection.py",
                    carrier, caller, self.out("canonical-row-rejections"),
                    outputs=[self.out("canonical-row-rejections")])
        self.run("fresh-witness-bytes", "static", REPO,
                 ["cmp", witness, successor / "fresh-witness.json"])
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
        self.handoff()

    def handoff(self):
        prior = read(self.request)
        caller = read(self.out("caller.json"))
        native = self.out("native-successor")
        envelope = read(native / "envelope.json")
        # The existing selected caller schema has two 49,393-word state preimages.
        private, result = caller[2], caller[4]
        offset = 49393
        if (len(caller) != 5 or caller[0] != 1 or caller[1] != list(map(int, CONTEXT))
                or len(result) != 7 or private[28] != prior[0]
                or private[30:34] != prior[1] or private[35:39] != prior[2]
                or private[-4:] != prior[3] or private[offset + 28] != prior[0] + 1
                or private[offset + 30:offset + 34] != prior[1]
                or private[offset + 35:offset + 39] != result[0]):
            raise ValueError("caller does not contain the exact requested prior and returned successor")
        lean_request = [private[offset + 28], private[offset + 30:offset + 34], result[0], prior[3]]
        native_request = [envelope["iteration"], envelope["z0"], envelope["current"], prior[3]]
        if lean_request != native_request:
            raise ValueError("independently returned successor states differ")
        for path, request in [(self.out("next-message-input.json"), lean_request),
                              (native / "next-message-input.json", native_request)]:
            if path.exists():
                if read(path) != request:
                    raise ValueError("changed saved next request")
            else:
                with path.open("x") as stream:
                    json.dump(request, stream, separators=(",", ":"))
                    stream.write("\n")
        for child in range(16):
            source = self.out(f"native-material/digit-{child}.json")
            target = native / source.name
            if target.is_symlink():
                raise ValueError("unexpected native child handoff symlink")
            if target.exists():
                if not target.samefile(source):
                    raise ValueError("changed native child handoff")
            else:
                target.hardlink_to(source)
        record = self.out("handoff.json")
        value = {"from_iteration": prior[0], "to_iteration": lean_request[0],
                 "lean_request": lean_request, "native_request": native_request,
                 "lean_private_inputs": ["fresh-witness.json", "fresh-claim.json",
                                         "children.json", "digits-0.jsonl", "digits-1.jsonl"],
                 "authority": "Both states are calculated by their own producer; equality is checked here. "
                              "The next Lean source projection reads only Lean outputs."}
        if record.exists():
            if read(record) != value:
                raise ValueError("changed handoff record")
        else:
            write_new(record, value)
        receipt = self.logs / "handoff.json"
        outputs = [self.out("next-message-input.json"), native / "next-message-input.json", record,
                   *(native / f"digit-{child}.json" for child in range(16))]
        identities = {str(path): identity(path) for path in outputs}
        if receipt.exists():
            check_outputs(read(receipt))
        else:
            write_new(receipt, {"operation": "checked exact successor handoff", "exit": 0,
                                "outputs": [str(path) for path in outputs],
                                "output_identities": identities})

    def terminal(self):
        native, material, caller = self.out("native-successor"), self.out("native-material"), self.out("caller.json")
        self.rust("terminal-accepted", "check-later-terminal", "accepted", native, material,
                  caller, self.out("terminal-accepted.json"), outputs=[self.out("terminal-accepted.json")])
        for case in ["ce-evaluation", "ce-matrix-evaluation", "fresh-private"]:
            changed, result = self.out(f"terminal-{case}-input"), self.out(f"terminal-{case}.json")
            self.rust(f"terminal-prepare-{case}", "prepare-terminal-mutation", case,
                      native, material, caller, changed, outputs=[changed])
            self.rust(f"terminal-check-{case}", "check-prepared-terminal-mutation", case,
                      changed, material, caller, result, outputs=[result])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-timeout", action="store_true",
                        help="disable command deadlines when explicitly authorized by the owner")
    parser.add_argument("root", type=Path)
    parser.add_argument("iteration", type=int, choices=(2, 3))
    parser.add_argument("checkpoint", choices=("build", "prepare", "native", "ccs", "reductions", "successor", "terminal", "all"))
    args = parser.parse_args()
    if args.checkpoint == "all":
        if args.iteration != 2:
            parser.error("the selected complete loop starts at iteration 2")
        for iteration in (2, 3):
            replay = Replay(args.root, iteration, no_timeout=args.no_timeout)
            if iteration == 2:
                replay.build()
            for checkpoint in ("prepare", "native", "ccs", "reductions", "successor"):
                getattr(replay, checkpoint)()
        replay.terminal()
    else:
        replay = Replay(args.root, args.iteration, no_timeout=args.no_timeout)
        getattr(replay, args.checkpoint)()


if __name__ == "__main__":
    main()
