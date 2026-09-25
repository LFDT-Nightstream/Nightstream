#!/usr/bin/env python3
"""Compare one complete binary prefix fold with indexed Python interpolation.

The coordinator runs this command under the 300-second native test cap.
Causal trace and source provenance are supplied by separate evidence.
"""

from __future__ import annotations

from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import sys

# Spec/Algebra.lean: canonical Goldilocks words, K = F[u]/(u² - 7).
P = 18446744069414584321
NONRESIDUE = 7
FIELD = struct.Struct("<QQ")


@dataclass(frozen=True)
class Chunk:
    path: Path
    first: int
    finish: int
    first_words: tuple[int, ...]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sequence(value, length, label):
    require(type(value) is list, f"{label}: expected an array")
    if length is not None:
        require(len(value) == length, f"{label}: wrong array length")
    return value


def natural(value, label):
    require(type(value) is int and value >= 0, f"{label}: expected a natural number")
    return value


def field(value, label):
    require(type(value) is int and 0 <= value < P, f"{label}: noncanonical field word")
    return value


def extension(value, label):
    a, b = sequence(value, 2, label)
    return field(a, label), field(b, label)


def round_challenge(path):
    label = str(path)
    value = sequence(json.loads(path.read_bytes()), 10, label)
    require(natural(value[0], label) == 1, f"{label}: wrong Lean round schema")
    for point in sequence(value[1], 28, f"{label}: alpha"):
        extension(point, label)
    extension(value[2], label)
    for index in (3, 6):
        for word in sequence(value[index], None, f"{label}: state"):
            field(word, label)
    for coefficient in sequence(value[4], 10, f"{label}: polynomial"):
        extension(coefficient, label)
    for index in (5, 7, 8, 9):
        extension(value[index], label)
    return value[5]


def decode_words(data, label):
    require(len(data) % FIELD.size == 0, f"{label}: partial extension-field value")
    words = array("Q")
    require(words.itemsize == 8, "binary format requires 8-byte unsigned words")
    words.frombytes(data)
    if sys.byteorder != "little":
        words.byteswap()
    for index, word in enumerate(words):
        if word >= P:
            raise ValueError(f"{label}: noncanonical field word {index}")
    return words


def read_prefix(directory):
    require(directory.is_dir(), f"{directory}: not a prefix directory")
    path = directory / "manifest.json"
    require(path.is_file(), f"{path}: not a manifest file")
    manifest = sequence(json.loads(path.read_bytes()), 7, str(path))
    schema, depth, kind, width, count, coins, ranges = manifest
    for value in manifest[:5]:
        natural(value, str(path))
    require(schema == 1 and width > 0, f"{path}: invalid schema or row width")
    for coin in sequence(coins, depth, f"{path}: challenges"):
        extension(coin, str(path))
    sequence(ranges, None, f"{path}: ranges")
    chunks = []
    names = {"manifest.json"}
    next_row = 0
    for extent in ranges:
        first, finish = sequence(extent, 2, str(path))
        natural(first, str(path))
        natural(finish, str(path))
        require(first == next_row and first < finish <= count,
                f"{path}: gap, overlap or invalid extent at row {next_row}")
        payload = directory / f"{first}-{finish}.bin"
        names.add(payload.name)
        require(payload.is_file(), f"{payload}: not a payload file")
        require(payload.stat().st_size == (finish - first) * width * FIELD.size,
                f"{payload}: wrong byte count")
        with payload.open("rb") as stream:
            first_row = stream.read(width * FIELD.size)
        require(len(first_row) == width * FIELD.size, f"{payload}: truncated first row")
        first_words = tuple(decode_words(first_row, str(payload)))
        chunks.append(Chunk(payload, first, finish, first_words))
        next_row = finish
    require(next_row == count, f"{path}: incomplete range coverage")
    require({entry.name for entry in directory.iterdir()} == names,
            f"{directory}: missing or extra files")
    return manifest, chunks


def compare_bytes(expected, actual, label):
    require(len(actual) == len(expected), f"{label}: wrong output byte count")
    require(len(actual) % FIELD.size == 0, f"{label}: partial output field value")
    for index, (a, b) in enumerate(FIELD.iter_unpack(actual)):
        if a >= P or b >= P:
            raise ValueError(f"{label}: noncanonical output field value {index}")
    if actual != expected:
        offset = next(index for index, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise ValueError(f"{label}: field bytes differ at byte {offset}")


def check_chunk(job):
    chunk, next_words, width, count, challenge, output = job
    data = chunk.path.read_bytes()
    input_bytes = (chunk.finish - chunk.first) * width * FIELD.size
    require(len(data) == input_bytes, f"{chunk.path}: input byte count changed")
    words = decode_words(data, str(chunk.path))
    del data
    require(tuple(words[:2 * width]) == chunk.first_words, f"{chunk.path}: first row changed")
    # Direct global row indices own the pairs. No pending-row state machine is used.
    if chunk.finish < count:
        require(next_words is not None, f"{chunk.path}: missing next first row")
        words.extend(next_words)
    else:
        require(chunk.finish == count and next_words is None, f"{chunk.path}: invalid source tail")
        words.extend([0] * (2 * width))
    first, finish = (chunk.first + 1) // 2, (chunk.finish + 1) // 2
    expected = bytearray((finish - first) * width * FIELD.size)
    r0, r1 = challenge
    r1_nr = NONRESIDUE * r1
    offset = 0
    for index in range(first, finish):
        low = (2 * index - chunk.first) * width * 2
        high = (2 * index + 1 - chunk.first) * width * 2
        for lane in range(width):
            left, right = low + 2 * lane, high + 2 * lane
            a, b = words[left], words[left + 1]
            delta0, delta1 = words[right] - a, words[right + 1] - b
            c0 = (a + r0 * delta0 + r1_nr * delta1) % P
            c1 = (b + r0 * delta1 + r1 * delta0) % P
            FIELD.pack_into(expected, offset, c0, c1)
            offset += FIELD.size
    del words
    require(offset == len(expected), f"{chunk.path}: reference field count differs")
    if first == finish:
        return input_bytes, 0, 0, 0, None
    target = output / f"{first}-{finish}.bin"
    actual = target.read_bytes()
    compare_bytes(expected, actual, str(target))
    tail = (bytes(expected[-FIELD.size:]), actual[-FIELD.size:]) if finish == (count + 1) // 2 else None
    crossing = int(chunk.finish % 2 == 1 and chunk.finish < count)
    return input_bytes, finish - first, len(actual), crossing, tail


def main():
    require(len(sys.argv) == 4,
            "usage: check_piccs_binary_fold.py <input-prefix-dir> <output-prefix-dir> <next-Lean-round-json>")
    source, output, round_path = map(Path, sys.argv[1:])
    manifest, chunks = read_prefix(source)
    _, depth, kind, width, count, coins, _ = manifest
    # A final-word mutation needs an actual final output word.
    require(count > 0, "final-word mutation requires a nonempty input prefix")
    challenge = round_challenge(round_path)
    expected_ranges = [[(chunk.first + 1) // 2, (chunk.finish + 1) // 2] for chunk in chunks
                       if (chunk.first + 1) // 2 < (chunk.finish + 1) // 2]
    expected_manifest = [1, depth + 1, kind, width, (count + 1) // 2,
                         coins + [challenge], expected_ranges]
    actual_manifest, _ = read_prefix(output)
    require(actual_manifest == expected_manifest,
            "output manifest differs from the complete input fold or next challenge")
    jobs = [(chunk, chunks[index + 1].first_words if index + 1 < len(chunks) else None,
             width, count, tuple(challenge), output) for index, chunk in enumerate(chunks)]
    input_bytes = output_rows = output_bytes = crossing_pairs = 0
    final_tail = None
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for checked_input, rows, checked_output, crossings, tail in executor.map(check_chunk, jobs):
            input_bytes += checked_input
            output_rows += rows
            output_bytes += checked_output
            crossing_pairs += crossings
            if tail is not None:
                require(final_tail is None, "duplicate final output range")
                final_tail = tail
    output_count = (count + 1) // 2
    require(input_bytes == count * width * FIELD.size, "not every input field was checked")
    require(output_rows == output_count and output_bytes == output_count * width * FIELD.size,
            "not every output field was compared")
    require(final_tail is not None, "missing final output field")
    expected_tail, actual_tail = final_tail
    changed = bytearray(actual_tail)
    a, b = FIELD.unpack(changed)
    FIELD.pack_into(changed, 0, a, (b + 1) % P)
    try:
        compare_bytes(expected_tail, changed, "changed final output word")
    except ValueError as error:
        require("field bytes differ" in str(error), "mutation failed for an unrelated reason")
    else:
        raise ValueError("changed final output word was accepted")
    print(json.dumps({
        "event": "piccs_binary_fold_byte_match", "kind": kind, "width": width,
        "input_depth": depth, "output_depth": depth + 1,
        "input_chunks": len(chunks), "output_chunks": len(expected_ranges),
        "input_rows": count, "input_K_values": count * width,
        "input_field_words": 2 * count * width, "input_bytes_checked": input_bytes,
        "output_rows": output_rows, "compared_K_values": output_count * width,
        "compared_field_words": 2 * output_count * width, "compared_bytes": output_bytes,
        "cross_file_pairs": crossing_pairs, "zero_tail_pairs": count % 2,
        "final_tail_mutation_rejected": True,
    }))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
