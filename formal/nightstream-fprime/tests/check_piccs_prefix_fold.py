#!/usr/bin/env python3
"""Compare every second-fold byte with indexed Python field interpolation.

The coordinator runs this command under the 300-second native test cap.
Original public-input and trace provenance remain separate evidence.
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

# Spec/Algebra.lean defines p and K.mul on the basis (1,u), with u² = 7.
P = 18446744069414584321
NONRESIDUE = 7
# Selected geometry: PICCS_CARRIED_PREFIX_REPLAY.json, pad and matrix records.
# Profile.lean fixes ring degree 54 and the 14 separate matrix ports.
ORIGINAL_ROWS = 6377559
PAD_RECORDS = 4685394
MATRIX_RECORDS = (ORIGINAL_ROWS + 1) // 2
FIELD = struct.Struct("<QQ")


@dataclass(frozen=True)
class Profile:
    kind: int
    records: int
    width: int

    @property
    def items_per_record(self):
        return 1 if self.kind == 2 else self.width

    @property
    def output_width(self):
        return self.width if self.kind == 2 else 1

    @property
    def output_count(self):
        return (self.items_per_record * self.records + 1) // 2


PROFILES = {
    "pad": Profile(0, PAD_RECORDS, 27),
    "matrix": Profile(1, MATRIX_RECORDS, 1),
    "fresh": Profile(2, MATRIX_RECORDS, 14),
}


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


def read_round(path):
    value = sequence(json.loads(path.read_bytes()), 10, str(path))
    require(natural(value[0], str(path)) == 1, f"{path}: wrong round schema")
    for point in sequence(value[1], 28, f"{path}: alpha"):
        extension(point, f"{path}: alpha")
    extension(value[2], f"{path}: gamma")
    for index in (3, 6):
        for word in sequence(value[index], None, f"{path}: state"):
            field(word, f"{path}: state")
    for coefficient in sequence(value[4], 10, f"{path}: polynomial"):
        extension(coefficient, f"{path}: polynomial")
    for index in (5, 7, 8, 9):
        extension(value[index], f"{path}: challenge or claim")
    return value


def header(value, profile, challenge, label):
    if profile.kind == 2:
        fields = sequence(value, 7, label)
        expected = [1, 1, profile.width, ORIGINAL_ROWS]
    else:
        fields = sequence(value, 8, label)
        expected = [1, 1, profile.kind, profile.width, profile.records]
    for word in fields[:-1]:
        natural(word, label)
    require(fields[:-3] == expected, f"{label}: wrong legacy profile")
    require(extension(fields[-1], label) == challenge, f"{label}: wrong first challenge")
    first, finish = fields[-3:-1]
    require(first < finish <= profile.records, f"{label}: invalid legacy extent")
    return first, finish


def record(line, width, index, label):
    position, values = sequence(json.loads(line), 2, label)
    require(natural(position, label) == index, f"{label}: nonconsecutive record {index}")
    sequence(values, width, label)
    words = []
    for pair in values:
        a, b = sequence(pair, 2, label)
        # Validate external integers before any modular arithmetic.
        if type(a) is not int or type(b) is not int or not (0 <= a < P and 0 <= b < P):
            raise ValueError(f"{label}: noncanonical field word at record {index}")
        words.extend((a, b))
    return words


def legacy_chunks(directories, profile, challenge):
    chunks = []
    for directory in directories:
        require(directory.is_dir(), f"{directory}: not a directory")
        for path in directory.iterdir():
            require(path.is_file(), f"{path}: not a regular input file")
            if profile.kind != 2 and path.name == "moments.json":
                continue  # Existing carried metadata; not an interpolation input.
            require(path.suffix == ".jsonl", f"{path}: unexpected legacy file")
            with path.open("rb") as stream:
                first, finish = header(json.loads(stream.readline()), profile, challenge, str(path))
                first_words = tuple(record(stream.readline(), profile.width, first, str(path)))
            chunks.append(Chunk(path, first, finish, first_words))
    chunks.sort(key=lambda chunk: chunk.first)
    next_record = 0
    for chunk in chunks:
        require(chunk.first == next_record, f"{chunk.path}: source gap or overlap at {next_record}")
        next_record = chunk.finish
    require(next_record == profile.records, "legacy source coverage is incomplete")
    return chunks


def output_range(profile, chunk):
    return ((profile.items_per_record * chunk.first + 1) // 2,
            (profile.items_per_record * chunk.finish + 1) // 2)


def check_manifest(directory, profile, chunks, challenges):
    manifest = sequence(json.loads((directory / "manifest.json").read_bytes()), 7, "manifest")
    for value in manifest[:5]:
        natural(value, "manifest")
    for coin in sequence(manifest[5], 2, "manifest challenges"):
        extension(coin, "manifest challenge")
    for extent in sequence(manifest[6], None, "manifest ranges"):
        for value in sequence(extent, 2, "manifest extent"):
            natural(value, "manifest extent")
    ranges = [list(output_range(profile, chunk)) for chunk in chunks
              if output_range(profile, chunk)[0] < output_range(profile, chunk)[1]]
    expected = [1, 2, profile.kind, profile.output_width, profile.output_count, challenges, ranges]
    require(manifest == expected, "output manifest differs from the complete source ranges or challenges")
    names = {"manifest.json"} | {f"{first}-{finish}.bin" for first, finish in ranges}
    require({path.name for path in directory.iterdir()} == names, "missing or extra output files")
    for first, finish in ranges:
        path = directory / f"{first}-{finish}.bin"
        require(path.is_file(), f"{path}: not a regular output file")
        require(path.stat().st_size == (finish - first) * profile.output_width * FIELD.size,
                f"{path}: wrong output byte count")
    return ranges


def compare_bytes(expected, actual, label):
    require(len(actual) == len(expected), f"{label}: wrong output byte count")
    require(len(actual) % FIELD.size == 0, f"{label}: partial field value")
    for a, b in FIELD.iter_unpack(actual):
        require(a < P and b < P, f"{label}: noncanonical output field word")
    if actual != expected:
        offset = next(index for index, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise ValueError(f"{label}: field bytes differ at byte {offset}")


def check_chunk(job):
    profile, chunk, next_words, r0, r1, directory = job
    label = str(chunk.path)
    words = array("Q")
    with chunk.path.open("rb") as stream:
        require(header(json.loads(stream.readline()), profile, r0, label) == (chunk.first, chunk.finish),
                f"{label}: input extent changed")
        for index in range(chunk.first, chunk.finish):
            row = record(stream.readline(), profile.width, index, label)
            if index == chunk.first:
                require(tuple(row) == chunk.first_words, f"{label}: first record changed")
            words.extend(row)
        require(stream.readline().strip() == b"[]", f"{label}: missing terminator")
        require(stream.read(1) == b"", f"{label}: extra data after terminator")

    item_start = profile.items_per_record * chunk.first
    item_finish = profile.items_per_record * chunk.finish
    width = profile.output_width
    require(len(words) == (item_finish - item_start) * width * 2, f"{label}: wrong input field count")
    # One indexed look-ahead row closes a pair crossing a file boundary.
    # Zero is available only after the true end of the complete source.
    if chunk.finish < profile.records:
        require(next_words is not None, f"{label}: missing next source record")
        words.extend(next_words if profile.kind == 2 else next_words[:2])
    else:
        require(next_words is None, f"{label}: unexpected next source record")
        words.extend([0] * (width * 2))

    first, finish = output_range(profile, chunk)
    expected = bytearray((finish - first) * width * FIELD.size)
    r_real, r_imag = r1
    r_imag_nr = NONRESIDUE * r_imag
    offset = 0
    for output_index in range(first, finish):
        low = (2 * output_index - item_start) * width * 2
        high = (2 * output_index + 1 - item_start) * width * 2
        for lane in range(width):
            at_low, at_high = low + 2 * lane, high + 2 * lane
            a, b = words[at_low], words[at_low + 1]
            delta_real = words[at_high] - a
            delta_imag = words[at_high + 1] - b
            c0 = (a + r_real * delta_real + r_imag_nr * delta_imag) % P
            c1 = (b + r_real * delta_imag + r_imag * delta_real) % P
            FIELD.pack_into(expected, offset, c0, c1)
            offset += FIELD.size
    require(offset == len(expected), f"{label}: reference field count differs")
    if first == finish:
        return 0, 0, 0, None
    path = directory / f"{first}-{finish}.bin"
    actual = path.read_bytes()
    compare_bytes(expected, actual, str(path))
    tail = (bytes(expected[-FIELD.size:]), actual[-FIELD.size:]) if finish == profile.output_count else None
    crossing = int(item_finish % 2 == 1 and chunk.finish < profile.records)
    return finish - first, len(actual), crossing, tail


def main():
    require(len(sys.argv) >= 6 and sys.argv[1] in PROFILES,
            "usage: check_piccs_prefix_fold.py <pad|matrix|fresh> <LeanQ0> <LeanQ1> "
            "<output-prefix-dir> <legacy-dir>...")
    family = sys.argv[1]
    profile = PROFILES[family]
    q0, q1 = (read_round(Path(path)) for path in sys.argv[2:4])
    require(q0[1:3] == q1[1:3] and q0[6] == q1[3] and q0[9] == q1[7],
            "saved Lean rounds do not form a consistent prefix")
    challenges = [q0[5], q1[5]]
    r0, r1 = map(tuple, challenges)
    directory = Path(sys.argv[4])
    require(directory.is_dir(), f"{directory}: not an output directory")
    chunks = legacy_chunks([Path(path) for path in sys.argv[5:]], profile, r0)
    ranges = check_manifest(directory, profile, chunks, challenges)
    jobs = [(profile, chunk, chunks[index + 1].first_words if index + 1 < len(chunks) else None,
             r0, r1, directory) for index, chunk in enumerate(chunks)]
    rows = byte_count = crossing_pairs = 0
    final_tail = None
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for checked_rows, checked_bytes, crossings, tail in executor.map(check_chunk, jobs):
            rows += checked_rows
            byte_count += checked_bytes
            crossing_pairs += crossings
            if tail is not None:
                require(final_tail is None, "duplicate final output range")
                final_tail = tail
    values = profile.output_count * profile.output_width
    require(rows == profile.output_count and byte_count == values * FIELD.size,
            "comparison did not cover every output field")
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
        "event": "piccs_prefix_fold_byte_match", "family": family,
        "input_chunks": len(chunks), "input_records": profile.records,
        "input_K_values": profile.records * profile.width, "output_chunks": len(ranges),
        "output_rows": rows, "output_width": profile.output_width,
        "compared_K_values": values, "compared_field_words": 2 * values,
        "compared_bytes": byte_count, "cross_file_pairs": crossing_pairs,
        "zero_tail_pairs": (profile.items_per_record * profile.records) % 2,
        "final_tail_mutation_rejected": True,
    }))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
