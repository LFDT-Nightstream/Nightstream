#!/usr/bin/env python3
"""Check one current native fold, caller and physical witness with Lean.

This coordinates verifier checks. Rust supplies C proof messages; this does
not claim independent Lean proof generation. Each child command has the
project cap and shared queue; the coordinator holds neither across commands.
"""

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent
FORMAL = ROOT / "formal/nightstream-fprime"
sys.path[:0] = [str(TESTS), str(ROOT / "scripts"), str(FORMAL / "scripts")]
from compare_recursive_outputs import equal, load  # noqa: E402
from check_identity import rust_code  # noqa: E402
from generate_lean_mutations import CHILDREN, MATRICES, numeric_json  # noqa: E402
from lean_graph.guard import build_lock, check_build_processes  # noqa: E402
from lean_graph.policy import CAPS  # noqa: E402
from project_replay_sources import commitment, field, original_running, vector, wrapped_field  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def package_pin(path):
    matches = re.findall(
        r"pub const POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY:\s*\[u64;\s*4\]\s*=\s*\[([\d_,\s]+)\];",
        rust_code(path.read_text()),
    )
    require(len(matches) == 1, "expected one selected package identity pin")
    words = [int(word.strip().replace("_", "")) for word in matches[0].split(",") if word.strip()]
    vector(words, 4, field)
    return words


def compare_files(actual, expected, label):
    # filecmp's cache can hide a changed file with identical stat fields.
    filecmp.clear_cache()
    require(filecmp.cmp(actual, expected, shallow=False), f"{label}: complete bytes differ")


def compare_source(input_value, envelope, fresh, request, identity):
    equal(envelope["package_identity"], identity, "source package identity")
    equal([envelope["iteration"], envelope["z0"], envelope["current"]], request[:3], "external prior state")
    equal(input_value[1], commitment(fresh["c"]), "original fresh commitment")
    equal(input_value[2], vector(fresh["x"], 270, wrapped_field), "original fresh public input")
    equal(input_value[6], original_running(envelope), "original running claims")


def compare_caller(native, lean, observed, context):
    equal(set(native), {"schema", "verifier_context", "private_values", "public_values",
                        "output", "output_digest", "next_public_input"}, "native caller fields")
    equal(len(lean), 5, "Lean caller schema width")
    equal(len(lean[4]), 7, "Lean caller result width")
    equal(native["schema"], 1, "native caller schema")
    equal(lean[0], 1, "Lean caller schema")
    for label, actual, expected in (
        ("native verifier context", native["verifier_context"], context),
        ("Lean verifier context", lean[1], context),
        ("every private caller word", native["private_values"], lean[2]),
        ("every public caller word", native["public_values"], lean[3]),
        ("application output", native["output"], lean[4][0]),
        ("output digest", native["output_digest"], lean[4][1]),
        ("next public input", native["next_public_input"], lean[4][2]),
        ("caller C point", observed["pi_ccs_phase"][6], lean[4][3]),
        ("caller C state", observed["pi_ccs_phase"][14], lean[4][4]),
        ("caller R state", observed["outgoing_state"], lean[4][5]),
        ("caller R public input", observed["pi_rlc_parent"][1], lean[4][6]),
    ):
        equal(actual, expected, label)
    return {"private_words": len(lean[2]), "public_words": len(lean[3]),
            "complete_caller_word_equality_checked": True, "all_seven_result_fields_checked": True}


def mutation_rejections(manifest, log):
    owners = {owner: [] for owner in ("decoder", "public_check", "pi_ccs")}
    for case in manifest["cases"]:
        owner = "pi_ccs" if case["expected_owner"] == "upstream_pi_ccs" else case["expected_owner"]
        require(owner in owners, f"unknown Lean mutation owner: {owner}")
        if owner == "pi_ccs":
            present = f"lean_pi_ccs_mutation={case['case']} rejected_by=pi_ccs" in log.splitlines()
        else:
            present = f"lean_pi_dec_mutation={Path(case['file']).name} rejected_by={owner}" in log
        require(present, f"missing Lean rejection: {case['case']}")
        owners[owner].append(case["case"])
    equal(owners["pi_ccs"], ["invalid_first_round_constant"], "selected PiCCS mutation class")
    require(f"lean_pi_dec_mutations=passed public={len(owners['public_check'])} "
            f"encoding={len(owners['decoder'])} unbounded=1 rejected_C_stops_D=1" in log,
            "incomplete Lean mutation result")
    return {**owners, "internal": ["unbounded_parent", "rejected_C_stops_D"]}


class Check:
    def __init__(self, directory, step, output, native_checker):
        self.directory, self.step, self.output = directory, step, output
        self.native_checker = native_checker
        self.originals = {}
        self.records = []

    def snapshot(self, source, relative):
        source = source.resolve(strict=True)
        if source in self.originals:
            return self.originals[source]
        target = self.output / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), f"snapshot already exists: {target}")
        shutil.copyfile(source, target)
        self.originals[source] = target
        return target

    def phase(self, name, kind, command, cwd=ROOT):
        cap = CAPS["lean" if kind == "lean" else "rust"]
        record = {"name": name, "kind": kind, "command": list(map(str, command)),
                  "cwd": str(cwd), "cap_seconds": cap, "outcome": "failed-to-start"}
        self.records.append(record)
        log = self.output / f"{name}.log"
        started = time.monotonic()
        process = None
        handlers = {}

        def interrupted(signum, _frame):
            raise InterruptedError(f"signal {signum}")

        try:
            with build_lock(), log.open("xb") as stream:
                check_build_processes()
                environment = dict(os.environ)
                if kind == "lean":
                    environment["LEAN_TIMEOUT_SECONDS"] = str(cap)
                process = subprocess.Popen(record["command"], cwd=cwd, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupted)
                try:
                    code = process.wait(timeout=max(0, cap - (time.monotonic() - started)))
                    record.update(exit=code, outcome="passed" if code == 0 else "failed")
                except subprocess.TimeoutExpired:
                    record.update(exit=124, outcome="timed-out")
                except InterruptedError:
                    record.update(exit=130, outcome="interrupted")
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        except Exception as error:
            record.update(exit=1, error=str(error))
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            record["elapsed_seconds"] = time.monotonic() - started
            with (self.output / f"{name}.json").open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
        require(record.get("exit") == 0, f"{name} failed; see {log}")
        return log.read_text()

    def lean(self, name, phase, *arguments):
        return self.phase(name, "lean", ["bash", "scripts/validate.sh", phase, *map(str, arguments)], FORMAL)

    def fold_inputs(self, step):
        relative = Path(f"fold-{step}")
        for name in ("pi_ccs_input.json", "children.json", "actual_result.json", "proof.native", "caller-inputs.json"):
            self.snapshot(self.directory / relative / name, relative / name)
        for name in ("envelope.json", "fresh-claim.json"):
            self.snapshot(self.directory / f"step-{step}" / name, Path(f"step-{step}") / name)
        folder = self.output / "inputs" / relative
        shutil.copyfile(folder / "proof.native", folder / "proof.bin")
        return folder

    def verify_fold(self, step, folder, identity, package):
        observed = load(folder / "actual_result.json")
        equal(observed["package_identity"], identity, "selected package identity")
        equal(observed["pi_ccs_input"], load(folder / "pi_ccs_input.json"), "exported C input")
        equal(observed["children"], load(folder / "children.json"), "exported D input")
        result = self.output / f"step-{step}-nifs.json"
        self.lean(f"step-{step}-verifier", "pi-dec-input-check", *identity,
                  folder / "pi_ccs_input.json", folder / "children.json", result)
        lean = load(result)
        equal(len(lean), 10, "complete Lean C/R/D result")
        for accepted in (lean[5][0], lean[7][0], lean[9][0], lean[9][16][0]):
            equal(accepted, 1, "Lean verifier acceptance")
        log = self.phase(f"step-{step}-native-comparison", "native",
                         [self.native_checker, "check-owned-nifs", package, folder, result])
        for marker in ("complete_nifs_wire=passed", "saved_actual_pi_dec=passed complete_fields=17 normal_wrapper=true",
                       "actual_selected_nifs_Lean_comparison=passed"):
            require(marker in log, f"native comparison omitted {marker}")
        wire = self.output / f"step-{step}-lean-proof.native"
        self.phase(f"step-{step}-wire", "native", [self.native_checker, "encode-lean-nifs", result, wire])
        compare_files(wire, folder / "proof.native", "Lean/native proof wire")
        rejected = re.findall(r"^pi_dec_mutation=(\S+) rejected=", log, re.MULTILINE)
        # Exact families in neo-fold-clean/tests/nifs/pi_dec_actual_mutations.rs.
        expected = {f"child_{child}_commitment" for child in range(CHILDREN)} | {
            f"child_eval_A{matrix}" for matrix in range(MATRICES)
        } | {f"nonzero_eval_A{matrix}_padding" for matrix in range(MATRICES)} | {
            "child_public", "child_point", "child_eval_K", "missing_child", "short_commitment",
            "short_point", "short_eval_K", "missing_matrix", "nonzero_eval_K_padding",
            "child_fold_digest", "child_digit_range",
        }
        equal(set(rejected), expected, "native D rejection classes")
        equal(len(rejected), len(expected), "unique native D rejections")
        return observed, rejected

    def caller(self, step, folder, request, context, observed):
        request_path, caller_path = (self.output / f"step-{step}-{name}.json" for name in ("request", "caller"))
        request_path.write_text(numeric_json(request))
        self.lean(f"step-{step}-caller", "recursive-step-fixture", *context,
                  folder / "pi_ccs_input.json", folder / "children.json", request_path, caller_path)
        lean = load(caller_path)
        counts = compare_caller(load(folder / "caller-inputs.json"), lean, observed, context)
        return lean, caller_path, counts

    def run(self):
        base = load(self.snapshot(TESTS / "fixtures/lean/nightstream-fprime-stage1-base-step-fixture-v1.json", "base.json"))
        request2 = load(self.snapshot(TESTS / "fixtures/stage1_recursive_states/nonzero-running.json", "request-2.json"))
        context = base[1]
        equal(len(context), 4, "original verifier context width")
        identity = package_pin(self.snapshot(ROOT / "crates/nightstream-fprime/src/identity.rs", "identity.rs"))
        package = self.snapshot(ROOT / "crates/nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json",
                                "package.json")
        folder = self.fold_inputs(self.step)
        if self.step == 1:
            request = [1, base[2][30:34], base[4][0], base[2][-4:]]
        elif self.step == 2:
            request = request2
        else:
            previous = self.fold_inputs(2)
            compare_source(load(previous / "pi_ccs_input.json"), load(self.output / "inputs/step-2/envelope.json"),
                           load(self.output / "inputs/step-2/fresh-claim.json"), request2, identity)
            observed, _ = self.verify_fold(2, previous, identity, package)
            prior_caller, _, _ = self.caller(2, previous, request2, context, observed)
            request = [3, request2[1], prior_caller[4][0], request2[3]]
            equal(load(folder / "pi_ccs_input.json")[6], load(previous / "children.json"), "own returned child handoff")
            equal(load(folder / "pi_ccs_input.json")[2], prior_caller[4][2], "prior returned fresh public input")
        compare_source(load(folder / "pi_ccs_input.json"), load(self.output / f"inputs/step-{self.step}/envelope.json"),
                       load(self.output / f"inputs/step-{self.step}/fresh-claim.json"), request, identity)
        observed, native_mutations = self.verify_fold(self.step, folder, identity, package)
        lean_caller, caller_path, counts = self.caller(self.step, folder, request, context, observed)
        successor = load(self.snapshot(self.directory / f"step-{self.step + 1}/envelope.json", "successor-envelope.json"))
        equal([successor["iteration"], successor["z0"], successor["current"]],
              [self.step + 1, request[1], lean_caller[4][0]], "returned successor state")
        native_physical = self.snapshot(self.directory / f"fold-{self.step}/physical.bin", "native-physical.bin")
        physical = self.output / "lean-physical.bin"
        self.lean("physical-build", "build", "replayPhysicalWitness")
        self.lean("physical", "lean-executable", FORMAL / ".lake/build/bin/replayPhysicalWitness", caller_path, physical)
        compare_files(physical, native_physical, "complete physical witness")

        generated = self.output / "mutation-inputs"
        self.phase("mutation-inputs", "python", [sys.executable, "-B", TESTS / "generate_lean_mutations.py",
                   folder / "pi_ccs_input.json", folder / "children.json", generated])
        manifest = load(generated / "manifest.json")
        log = self.lean("mutations", "pi-dec-mutations", *identity, folder / "pi_ccs_input.json",
                        folder / "children.json", generated / manifest["changed_ccs_input"],
                        generated / manifest["mutation_directory"])
        return {"caller": counts, "physical_bytes": physical.stat().st_size,
                "Lean_rejections": mutation_rejections(manifest, log),
                "native_D_rejections": native_mutations,
                "independent_proof_generation": False,
                "scope": "Fresh Lean verifier acceptance/rejection, complete native proof-byte, caller and physical-witness equality. "
                         "C proof messages are native inputs. Independent proof generation and terminal opening checks are separate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-checker", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checker = Check(args.directory.resolve(), args.step, output, args.native_checker.resolve(strict=True))
    result = {"schema": 1, "step": args.step, "directory": str(checker.directory),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "outcome": "failed", "commands": checker.records}
    try:
        result.update(checker.run())
        for source, snapshot in checker.originals.items():
            compare_files(source, snapshot, "original source changed during checking")
        result["outcome"] = "passed"
    except Exception as error:
        result["error"] = str(error)
    result["input_snapshots"] = []
    for source, snapshot in checker.originals.items():
        with snapshot.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result["input_snapshots"].append({"original": str(source), "snapshot": str(snapshot), "sha256": digest})
    result["digest_scope"] = "file custody only; equality checks use actual content"
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    return 0 if result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
