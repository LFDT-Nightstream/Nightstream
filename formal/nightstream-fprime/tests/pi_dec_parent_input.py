"""Check parent range decoding and zero-input arithmetic, not a producer trace."""

import json
from pathlib import Path
import subprocess
import sys
from threading import Thread


# The selected setup and field; these are existing production dimensions.
BLOCKS = 4_685_394
MODULUS = 2**64 - 2**32 + 1
LANES = 54


def stream(first=0, last=BLOCKS, records=(), terminator=True, extra=""):
    lines = [json.dumps([1, BLOCKS, first, last])]
    lines.extend(json.dumps(record) for record in records)
    if terminator:
        lines.append("[]")
    return "\n".join(lines) + "\n" + extra


def check_case(binary, ccs, directory, name, ranges, error=None, first=0, last=94):
    paths = []
    for index, text in enumerate(ranges):
        path = directory / f"{name}.{index}.jsonl"
        with path.open("x") as output:
            output.write(text)
        paths.append(str(path))
    output = directory / f"{name}.output.json"
    result = subprocess.run(
        [str(binary), str(ccs), str(output), "0", str(first), str(last), *paths],
        capture_output=True, text=True, timeout=300,
    )
    text = result.stdout + result.stderr
    with (directory / f"{name}.log").open("x") as log:
        log.write(text)
    if error is not None:
        assert result.returncode != 0, f"{name}: invalid parent accepted"
        assert error in text, f"{name}: wrong rejection\n{text}"
        assert not output.exists(), f"{name}: result written before rejection"
        print(f"pidec_parent_boundary={name} rejected", flush=True)
        return
    assert result.returncode == 0, f"{name}: valid zero parent rejected\n{text}"
    check_zero_output(output, first, last)
    print(f"pidec_parent_boundary={name} accepted", flush=True)


def check_zero_output(output, first, last, point=None):
    values = json.loads(output.read_text())
    assert isinstance(values, list) and len(values) == 6, "complete matrix range"
    assert all(type(word) is int for word in values[:4]), "numeric range identity"
    assert values[:4] == [1, 6_377_559, first, last], "selected range identity"
    assert isinstance(values[4], list) and len(values[4]) == 28, "complete C-derived point"
    for pair in values[4]:
        assert isinstance(pair, list) and len(pair) == 2, "complete point coefficient"
        assert all(type(word) is int and 0 <= word < MODULUS for word in pair), \
            "canonical point coefficient"
    if point is not None:
        assert values[4] == point, "batch changed the C-derived point"
    assert isinstance(values[5], list) and len(values[5]) == 16, "all children"
    for matrices in values[5]:
        assert isinstance(matrices, list) and len(matrices) == 14, "all matrices"
        for lanes in matrices:
            assert isinstance(lanes, list) and len(lanes) == LANES, "all Phi81 lanes"
            for pair in lanes:
                assert isinstance(pair, list) and len(pair) == 2, "complete K coefficient"
                assert all(type(word) is int and word == 0 for word in pair), \
                    "zero matrix action"
    return values


def reject_batch(binary, ccs, directory, name, arguments, outputs, error):
    result = subprocess.run(
        [str(binary), "ranges", str(ccs), *map(str, arguments)],
        capture_output=True, text=True, timeout=300,
    )
    text = result.stdout + result.stderr
    with (directory / f"{name}.log").open("x") as log:
        log.write(text)
    assert result.returncode != 0, f"{name}: invalid batch accepted"
    assert error in text, f"{name}: wrong rejection\n{text}"
    assert all(not output.exists() for output in outputs), \
        f"{name}: result written before batch rejection"
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert not any(event.get("event") in
                   ("range_begin", "range_complete", "batch_complete") for event in events), \
        f"{name}: started or completed a range before rejecting the batch"
    print(f"pidec_parent_boundary={name} rejected", flush=True)


def accept_batch(binary, ccs, directory, parent, first_output, second_output):
    # Two complete Poseidon ranges overlap but have different identities/paths.
    requested = {str(first_output): (0, 94), str(second_output): (0, 188)}
    reference_path = directory / "complete_zero_ranges.output.json"
    reference = check_zero_output(reference_path, 0, 94)
    command = [str(binary), "ranges", str(ccs),
               str(first_output), "0", "0", "94",
               str(second_output), "0", "0", "188", "--", str(parent)]
    events, lines, failures, completed = [], [], [], []
    with (directory / "batch_overlapping_zero.log").open("x") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)

        def read_events():
            for line in process.stdout:
                lines.append(line)
                log.write(line)
                log.flush()
                try:
                    if not line.startswith("{"):
                        continue
                    event = json.loads(line)
                    events.append(event)
                    if event.get("event") == "range_complete":
                        output = event["output"]
                        assert output in requested, "completed an unrequested output"
                        assert output not in completed, "duplicate range completion"
                        first, last = requested[output]
                        # Read all coefficients when completion is observed, not just at exit.
                        values = check_zero_output(Path(output), first, last, reference[4])
                        expected = reference.copy()
                        expected[2:4] = [first, last]
                        assert values == expected, "batch output differs from the legacy zero action"
                        if output == str(first_output):
                            assert Path(output).read_bytes() == reference_path.read_bytes(), \
                                "same request changed legacy output bytes"
                        completed.append(output)
                except Exception as error:
                    failures.append(str(error))
                    # Keep draining stdout so a failed assertion cannot block the child.

        reader = Thread(target=read_events)
        reader.start()
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            reader.join()
            process.stdout.close()
    text = "".join(lines)
    assert process.returncode == 0, f"valid overlapping batch rejected\n{text}"
    assert not failures, f"completion output checks failed: {failures}\n{text}"
    assert completed == list(requested), "missing, reordered or incomplete output completion"
    names = [event.get("event") for event in events]
    for name in ("point_ready", "parent_ready", "parent_magnitude_ready", "basis_ready",
                 "shared_load_complete", "batch_complete"):
        assert names.count(name) == 1, f"expected one {name} event"
    shared = next(event for event in events if event.get("event") == "shared_load_complete")
    total = events[-1]
    assert total.get("event") == "batch_complete", "missing final batch completion"
    assert shared["ranges"] == total["ranges"] == len(requested), "wrong batch request count"
    assert total["shared_load_ns"] == shared["shared_load_ns"], "shared timing changed"
    assert type(shared["shared_load_ns"]) is int and shared["shared_load_ns"] >= 0
    assert type(total["total_ns"]) is int and total["total_ns"] >= 0
    parent_event = next(event for event in events if event.get("event") == "parent_ready")
    assert parent_event["blocks"] == BLOCKS and parent_event["records"] == 0, \
        "wrong shared zero-parent load"
    magnitude = next(event for event in events if event.get("event") == "parent_magnitude_ready")
    assert magnitude["maximum"] == 0, "wrong zero-parent magnitude"
    position = names.index("shared_load_complete")
    for name in ("point_ready", "parent_ready", "parent_magnitude_ready", "basis_ready"):
        assert names.index(name) < position, f"{name} was not part of shared loading"
    completions = []
    for output, (first, last) in requested.items():
        position += 1
        begin = events[position]
        assert begin.get("event") == "range_begin" and begin["output"] == output, \
            "range did not start after the shared load or prior completion"
        end_position = next(index for index in range(position + 1, len(events))
                            if events[index].get("event") == "range_complete")
        end = events[end_position]
        assert end["output"] == output, "completion belongs to a different output"
        for event in (begin, end):
            assert event["block"] == 0, "wrong completed block"
            assert (event["first_local_row"], event["last_local_row_exclusive"]) == (first, last)
            assert (event["start"], event["end"]) == (first, last), "wrong global row identity"
        children = [event for event in events[position + 1:end_position]
                    if event.get("event") == "child_complete"]
        assert [event["child"] for event in children] == list(range(16)), "incomplete children"
        assert all(event["zero_from_parent_bound"] is True for event in children)
        assert end["children"] == 16 and end["matrices"] == 14
        assert end["field_words"] == 16 * 14 * LANES * 2
        assert end["timing_scope"] == "range_after_shared_load", "wrong range timing scope"
        assert type(end["total_ns"]) is int and end["total_ns"] >= 0
        completions.append(end)
        position = end_position
    assert position + 1 == len(events) - 1, "unexpected events after the final range"
    assert total["total_ns"] >= shared["shared_load_ns"] + sum(
        event["total_ns"] for event in completions), "batch total omits shared or range work"
    print("pidec_parent_boundary=batch_overlapping_zero accepted "
          "scope=decoder_coverage_and_zero_action", flush=True)


def check_batches(binary, ccs, directory):
    parent = directory / "batch.zero.jsonl"
    with parent.open("x") as output:
        output.write(stream())
    first = directory / "batch.first.output.json"
    second = directory / "batch.second.output.json"
    request = [first, "0", "0", "94"]
    parents = ["--", parent]
    cases = [
        ("batch_missing_separator", request + [parent],
         "ranges requires -- before the Lean parent paths"),
        ("batch_empty_requests", parents, "expected at least one matrix range"),
        ("batch_incomplete_request", request[:-1] + parents,
         "expected complete output/block/first/last groups"),
        ("batch_no_parents", request + ["--"], "ranges requires Lean parent paths"),
        ("batch_duplicate_output", request + request + parents, "duplicate matrix output path"),
    ]
    for index, label in ((1, "block"), (2, "first"), (3, "last")):
        changed = request.copy()
        changed[index] = "not-a-natural-number"
        cases.append((f"batch_nonnumeric_{label}", changed + parents,
                      "block and row bounds must be natural numbers"))
    for name, block, lo, hi, error in [
        ("invalid_block", 23, 0, 1, "invalid selected matrix block"),
        ("empty_range", 0, 0, 0, "invalid selected matrix row range"),
        ("reversed_range", 0, 1, 0, "invalid selected matrix row range"),
        ("past_range", 0, 0, 6_377_559, "invalid selected matrix row range"),
        ("poseidon_start", 0, 1, 94, "Poseidon range must contain complete 94-row invocations"),
        ("poseidon_end", 0, 0, 1, "Poseidon range must contain complete 94-row invocations"),
        ("phi81_start", 10, 1, 108, "Phi81 range must contain complete 108-row invocations"),
        ("phi81_end", 10, 0, 1, "Phi81 range must contain complete 108-row invocations"),
    ]:
        cases.append((f"batch_later_{name}", request + [second, block, lo, hi] + parents, error))
    for name, arguments, error in cases:
        reject_batch(binary, ccs, directory, name, arguments, [first, second], error)
    existing = directory / "batch.existing.output.json"
    sentinel = b"existing output must remain unchanged\n"
    with existing.open("xb") as output:
        output.write(sentinel)
    reject_batch(binary, ccs, directory, "batch_existing_later_output",
                 request + [existing, 0, 0, 94] + parents, [first, second], "output already exists")
    assert existing.read_bytes() == sentinel, "batch overwrote an existing output"
    malformed_ccs = directory / "batch.malformed.ccs.json"
    with malformed_ccs.open("x") as output:
        output.write("[]\n")
    reject_batch(binary, malformed_ccs, directory, "batch_malformed_ccs",
                 request + parents, [first, second], "expected seven PiCCS input fields")
    malformed_parent = directory / "batch.malformed.parent.jsonl"
    with malformed_parent.open("x") as output:
        output.write(stream(terminator=False))
    reject_batch(binary, ccs, directory, "batch_malformed_parent",
                 request + ["--", malformed_parent], [first, second], "missing parent terminator")
    accept_batch(binary, ccs, directory, parent, first, second)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: pi_dec_parent_input.py <replayPiDECMatrix> <valid-C-input> <new-results-directory>")
    binary = Path(sys.argv[1]).resolve(strict=True)
    ccs = Path(sys.argv[2]).resolve(strict=True)
    directory = Path(sys.argv[3]).resolve()
    directory.mkdir()
    zero = [0] * LANES
    check_case(binary, ccs, directory, "complete_zero_ranges",
               [stream(0, 1), stream(1, BLOCKS)])
    check_case(binary, ccs, directory, "missing_tail", [stream(0, BLOCKS - 1)],
               "parent ranges do not cover the complete carrier")
    check_case(binary, ccs, directory, "gap", [stream(1, BLOCKS)],
               "parent ranges have a gap, overlap or wrong length")
    check_case(binary, ccs, directory, "overlap", [stream(0, 1), stream(0, BLOCKS)],
               "parent ranges have a gap, overlap or wrong length")
    check_case(binary, ccs, directory, "duplicate", [stream(records=[(0, zero), (0, zero)])],
               "duplicate or out-of-range parent block")
    check_case(binary, ccs, directory, "past_tail", [stream(records=[(BLOCKS, zero)])],
               "duplicate or out-of-range parent block")
    check_case(binary, ccs, directory, "noncanonical", [stream(records=[(0, [MODULUS] + zero[1:])])],
               "noncanonical parent coefficient")
    check_case(binary, ccs, directory, "strict_bound", [stream(records=[(0, [2**16] + zero[1:])])],
               "parent exceeds the strict B bound")
    check_case(binary, ccs, directory, "short_block", [stream(records=[(0, zero[:-1])])],
               "expected 54 parent coefficients")
    check_case(binary, ccs, directory, "missing_terminator", [stream(terminator=False)],
               "missing parent terminator")
    check_case(binary, ccs, directory, "after_terminator", [stream(extra="[]\n")],
               "extra data after parent terminator")
    check_case(binary, ccs, directory, "empty_invocation_range", [stream()],
               "invalid selected matrix row range", last=0)
    check_case(binary, ccs, directory, "reversed_invocation_range", [stream()],
               "invalid selected matrix row range", first=1, last=0)
    check_case(binary, ccs, directory, "past_invocation_range", [stream()],
               "invalid selected matrix row range", last=6_377_559)
    check_case(binary, ccs, directory, "incomplete_poseidon_invocation", [stream()],
               "Poseidon range must contain complete 94-row invocations", last=1)
    check_batches(binary, ccs, directory)
    print("pidec_parent_boundaries=passed scope=decoder_coverage_and_zero_action", flush=True)


if __name__ == "__main__":
    main()
