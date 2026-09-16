"""Reject malformed stored-witness inputs without writing result files.

Run from the repository root under the native graph guard.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

P = 18446744069414584321
CARRIER_BYTES = 54 * 4685394

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caller")
    parser.add_argument("physical")
    parser.add_argument("directory")
    args = parser.parse_args()
    formal = Path.cwd() / "formal/nightstream-fprime"
    directory = Path(args.directory).resolve()
    directory.mkdir()
    caller_path, physical_path = Path(args.caller).resolve(), Path(args.physical).resolve()
    caller = json.loads(caller_path.read_text())
    rejected = []

    def run(name, executable, arguments, output, reason, existing=False):
        if existing:
            output.write_bytes(b"retained output")
        before = output.read_bytes() if output.is_file() else None
        result = subprocess.run(
            ["bash", "scripts/validate.sh", "lean-executable",
             str(formal / ".lake/build/bin" / executable), *map(str, arguments)],
            cwd=formal, capture_output=True, timeout=300)
        log = result.stdout + result.stderr
        (directory / (name + ".log")).write_bytes(log)
        require(result.returncode != 0, name + ": malformed input accepted")
        require(reason.encode() in log, name + ": missing rejection reason")
        if existing:
            require(output.read_bytes() == before, name + ": existing output changed")
        else:
            require(not output.exists(), name + ": rejected input wrote output")
        rejected.append(name)

    def caller_case(name, change, reason):
        value = copy.deepcopy(caller)
        change(value)
        source, output = directory / (name + ".json"), directory / (name + ".bin")
        source.write_text(json.dumps(value, separators=(",", ":")))
        run(name, "replayPhysicalWitness", [source, output], output, reason)

    output = directory / "physical-existing.bin"
    run("physical-existing", "replayPhysicalWitness", [caller_path, output], output,
        "physical output already exists", True)
    caller_case("caller-schema", lambda value: value.__setitem__(0, 2), "expected a complete schema-one caller")
    caller_case("caller-fields", lambda value: value.pop(), "expected a complete schema-one caller")
    caller_case("private-width", lambda value: value[2].pop(), "caller width differs")
    caller_case("public-width", lambda value: value[3].pop(), "caller width differs")
    caller_case("private-noncanonical", lambda value: value[2].__setitem__(-1, P), "noncanonical caller field")
    caller_case("public-noncanonical", lambda value: value[3].__setitem__(-1, P), "noncanonical caller field")
    caller_case("changed-public", lambda value: value[3].__setitem__(0, (value[3][0] + 1) % P),
                "physical assertion failed")

    empty = directory / "empty.bin"
    empty.write_bytes(b"")
    noncanonical = directory / "noncanonical.bin"
    with noncanonical.open("wb") as output_file:
        output_file.truncate(physical_path.stat().st_size)
        output_file.seek(0)
        output_file.write(P.to_bytes(8, "little"))
    for name, source, bounds, reason in [
        ("physical-truncated", empty, [0, 1], "physical file size differs"),
        ("physical-noncanonical", noncanonical, [0, 1], "noncanonical physical word"),
        ("assignment-reversed", physical_path, [2, 1], "assignment block range"),
        ("assignment-outside", physical_path, [30, 31], "assignment block range"),
        ("assignment-negative", physical_path, [-1, 1], "indices must be natural"),
    ]:
        output = directory / (name + "-output")
        run(name, "replayFreshAssignment", [source, output, *bounds], output, reason)
    output = directory / "assignment-existing"
    run("assignment-existing", "replayFreshAssignment", [physical_path, output, 0, 1],
        output, "assignment output directory already exists", True)

    invalid_code, invalid_tail = directory / "invalid-code.bin", directory / "invalid-tail.bin"
    for path, index, byte in [(invalid_code, 0, 2), (invalid_tail, CARRIER_BYTES - 1, 1)]:
        with path.open("wb") as output_file:
            output_file.truncate(CARRIER_BYTES)
            output_file.seek(index)
            output_file.write(bytes([byte]))
    for name, source, bounds, reason in [
        ("carrier-truncated", empty, [0, 1], "wrong complete byte count"),
        ("carrier-nonunit", invalid_code, [0, 1], "non-unit fresh coefficient"),
        ("carrier-tail", invalid_tail, [0, 1], "nonzero tail padding"),
        ("commitment-reversed", empty, [2, 1], "range is outside"),
        ("commitment-outside", empty, [0, 4685395], "range is outside"),
        ("commitment-negative", empty, [-1, 1], "bounds must be natural"),
    ]:
        output = directory / (name + ".json")
        run(name, "replayFreshCommitment", [source, output, *bounds], output, reason)
    output = directory / "commitment-existing.json"
    run("commitment-existing", "replayFreshCommitment", [empty, output, 0, 1],
        output, "output already exists", True)
    rows = [[0] * 54 for _ in range(22)]
    for name, data, reason in [
        ("merge-gap", [1, 4685394, 1, 4685394, rows], "gap, overlap or invalid endpoint"),
        ("merge-incomplete", [1, 4685394, 0, 1, rows], "do not cover the complete carrier"),
        ("merge-row-count", [1, 4685394, 0, 4685394, rows[:-1]], "wrong key-row count"),
        ("merge-lane-count", [1, 4685394, 0, 4685394, [row[:-1] for row in rows]], "wrong size"),
        ("merge-noncanonical", [1, 4685394, 0, 4685394, [[P] + row[1:] for row in rows]], "noncanonical"),
    ]:
        source, output = directory / (name + "-input.json"), directory / (name + ".json")
        source.write_text(json.dumps(data, separators=(",", ":")))
        run(name, "replayFreshCommitment", ["merge", output, source], output, reason)
    print(json.dumps({"status": "passed", "rejected": rejected, "count": len(rejected)}))

if __name__ == "__main__":
    main()
