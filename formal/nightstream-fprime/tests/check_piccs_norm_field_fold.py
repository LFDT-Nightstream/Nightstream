#!/usr/bin/env python3
"""Check complete norm-code conversion with independent Python field arithmetic.

Existing Python helpers own format and byte checks. This file owns the signed
code interpretation and three interpolation steps. Causal trace derivation
and original source provenance remain separate evidence. The coordinator runs
this test under the project's 300-second native cap.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

from check_piccs_binary_fold import (
    FIELD, NONRESIDUE, P, compare_bytes, extension, natural, read_prefix, require, sequence,
)
from check_piccs_prefix_fold import read_round


# Exact complete input profile required by this replay milestone.
SOURCES = 17
CARRIER = 253011276
INPUT_CODES = 63252819
OUTPUT_ROWS = (INPUT_CODES + 1) // 2
TABLE_SIZE = 3 ** 4
ZERO_CODE = 40


def interpolate(low, high, challenge):
    a, b = low
    c, d = high
    r, s = challenge
    return ((a + r * (c - a) + NONRESIDUE * s * (d - b)) % P,
            (b + r * (d - b) + s * (c - a)) % P)


def pair_table(challenges):
    r0, r1, r2 = map(tuple, challenges)
    values = []
    for code in range(TABLE_SIZE):
        # code = 27*a + 9*b + 3*c + d, with signed digit value digit - 1.
        digits = [((code // divisor) % 3 - 1, 0) for divisor in (27, 9, 3, 1)]
        low = interpolate(digits[0], digits[1], r0)
        high = interpolate(digits[2], digits[3], r0)
        values.append(interpolate(low, high, r1))
    require(values[ZERO_CODE] == (0, 0), "the derived zero code is not zero")
    return tuple(FIELD.pack(*interpolate(low, high, r2)) for low in values for high in values)


def check_sources(directory, challenges):
    require(directory.is_dir(), f"{directory}: not a norm-code directory")
    path = directory / "manifest.json"
    manifest = sequence(json.loads(path.read_bytes()), 7, str(path))
    for value in manifest[:6]:
        natural(value, str(path))
    for coin in sequence(manifest[6], 2, str(path)):
        extension(coin, str(path))
    require(manifest == [1, 2, SOURCES, CARRIER, 0, INPUT_CODES, challenges[:2]],
            "norm-code manifest differs from the complete selected input")
    names = {"manifest.json"} | {f"source-{source}.bin" for source in range(SOURCES)}
    require({entry.name for entry in directory.iterdir()} == names,
            "norm-code directory has missing or extra source files")
    for source in range(SOURCES):
        path = directory / f"source-{source}.bin"
        require(path.is_file() and path.stat().st_size == INPUT_CODES,
                f"{path}: wrong source file type or byte count")


def check_source(job):
    source, code_directory, output_parent, challenges, table = job
    directory = output_parent / f"source-{source}"
    manifest, chunks = read_prefix(directory)
    require(manifest[:6] == [1, 3, 3 + source, 1, OUTPUT_ROWS, challenges],
            f"{directory}: wrong source identity, shape or challenges")
    input_path = code_directory / f"source-{source}.bin"
    input_bytes = output_bytes = output_rows = zero_tails = 0
    final_tail = None
    with input_path.open("rb") as stream:
        for chunk in chunks:
            first_code = 2 * chunk.first
            finish_code = min(2 * chunk.finish, INPUT_CODES)
            require(stream.tell() == first_code, f"{input_path}: noncontiguous code read")
            codes = stream.read(finish_code - first_code)
            require(len(codes) == finish_code - first_code, f"{input_path}: truncated source")
            for offset, code in enumerate(codes):
                if code >= TABLE_SIZE:
                    raise ValueError(f"{input_path}: invalid code {code} at {first_code + offset}")
            input_bytes += len(codes)
            expected = bytearray()
            for index in range(chunk.finish - chunk.first):
                low_index, high_index = 2 * index, 2 * index + 1
                low = codes[low_index]
                if high_index < len(codes):
                    high = codes[high_index]
                else:
                    require(first_code + high_index == INPUT_CODES and chunk.finish == OUTPUT_ROWS,
                            f"{input_path}: zero padding before the true source end")
                    high = ZERO_CODE
                    zero_tails += 1
                expected.extend(table[TABLE_SIZE * low + high])
            require(len(expected) == (chunk.finish - chunk.first) * FIELD.size,
                    f"{chunk.path}: wrong reference field count")
            actual = chunk.path.read_bytes()
            compare_bytes(expected, actual, str(chunk.path))
            output_rows += chunk.finish - chunk.first
            output_bytes += len(actual)
            if source == SOURCES - 1 and chunk.finish == OUTPUT_ROWS:
                final_tail = (bytes(expected[-FIELD.size:]), actual[-FIELD.size:])
        require(stream.read(1) == b"", f"{input_path}: extra source byte")
    require(input_bytes == INPUT_CODES and output_rows == OUTPUT_ROWS and
            output_bytes == OUTPUT_ROWS * FIELD.size and zero_tails == INPUT_CODES % 2,
            f"source {source}: incomplete field coverage or incorrect tail padding")
    return ({"source": source, "input_codes": input_bytes, "output_chunks": len(chunks),
             "compared_K_values": output_rows, "compared_field_words": 2 * output_rows,
             "compared_bytes": output_bytes, "zero_tail_pairs": zero_tails}, final_tail)


def main():
    require(len(sys.argv) == 6,
            "usage: check_piccs_norm_field_fold.py <complete-norm-code-dir> "
            "<norm-field-parent> <Q0> <Q1> <Q2>")
    code_directory, output_parent = map(Path, sys.argv[1:3])
    rounds = [read_round(Path(path)) for path in sys.argv[3:6]]
    for previous, current in zip(rounds, rounds[1:]):
        require(previous[1:3] == current[1:3] and previous[6] == current[3] and
                previous[9] == current[7], "saved Lean rounds do not form a consistent chain")
    challenges = [value[5] for value in rounds]
    check_sources(code_directory, challenges)
    require(output_parent.is_dir(), f"{output_parent}: not a norm-field directory")
    require({entry.name for entry in output_parent.iterdir()} ==
            {f"source-{source}" for source in range(SOURCES)},
            "norm-field parent has missing or extra source directories")
    table = pair_table(challenges)
    jobs = [(source, code_directory, output_parent, challenges, table) for source in range(SOURCES)]
    summaries = []
    final_tail = None
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for summary, tail in executor.map(check_source, jobs):
            summaries.append(summary)
            if tail is not None:
                require(final_tail is None, "duplicate final source tail")
                final_tail = tail
    require([summary["source"] for summary in summaries] == list(range(SOURCES)),
            "not every source identity was checked")
    input_bytes = sum(summary["input_codes"] for summary in summaries)
    output_bytes = sum(summary["compared_bytes"] for summary in summaries)
    require(input_bytes == SOURCES * INPUT_CODES and output_bytes == SOURCES * OUTPUT_ROWS * FIELD.size,
            "not every source code and output byte was checked")
    require(final_tail is not None, "missing final source output field")
    expected_tail, actual_tail = final_tail
    changed = bytearray(actual_tail)
    a, b = FIELD.unpack(changed)
    FIELD.pack_into(changed, 0, a, (b + 1) % P)
    try:
        compare_bytes(expected_tail, changed, "changed final source output word")
    except ValueError as error:
        require("field bytes differ" in str(error), "mutation failed for an unrelated reason")
    else:
        raise ValueError("changed final source output word was accepted")
    print(json.dumps({
        "event": "piccs_norm_field_fold_byte_match", "sources": summaries,
        "input_codes_checked": input_bytes, "compared_K_values": SOURCES * OUTPUT_ROWS,
        "compared_field_words": 2 * SOURCES * OUTPUT_ROWS, "compared_bytes": output_bytes,
        "final_tail_mutation_rejected": True,
    }))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
