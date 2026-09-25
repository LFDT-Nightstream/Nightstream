#!/usr/bin/env python3
"""Partition one Lean R range for PiDEC input; copy coefficient lines verbatim.

Hashes identify transport bytes, not protocol authority. On failure, partial
outputs remain in the new directory and no completed manifest is written.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


# Read only the JSON block index. The coefficient payload remains opaque bytes.
BLOCK_INDEX = re.compile(rb"^[ \t]*\[[ \t]*(0|[1-9][0-9]*)[ \t]*,")


def read_header(raw):
    header = json.loads(raw)
    if not isinstance(header, list) or len(header) != 4 or any(type(word) is not int for word in header):
        raise ValueError("expected integer header [1,total,start,end]")
    schema, total, start, end = header
    if schema != 1 or total <= 0 or not 0 <= start < end <= total:
        raise ValueError("invalid Lean R range header")
    return total, start, end


def block_lines(source, digest, start, end):
    previous = start - 1
    for line_number, raw in enumerate(source, 2):
        digest.update(raw)
        if raw.strip() == b"[]":
            if source.read(1):
                raise ValueError(f"data after terminator at line {line_number}")
            return
        match = BLOCK_INDEX.match(raw)
        if match is None:
            raise ValueError(f"expected indexed block or [] at line {line_number}")
        block = int(match.group(1))
        if not start <= block < end or block <= previous:
            raise ValueError(f"duplicate, unordered, or out-of-range block {block} at line {line_number}")
        previous = block
        yield block, raw
    raise ValueError("missing [] source terminator")


def split_parent(source_path, output_directory, blocks_per_range):
    if blocks_per_range <= 0:
        raise ValueError("blocks-per-range must be positive")
    source_path = source_path.resolve(strict=True)
    output_directory = output_directory.absolute()
    outputs = []
    source_digest = hashlib.sha256()
    with source_path.open("rb") as source:
        header = source.readline()
        source_digest.update(header)
        total, start, end = read_header(header)
        # An existing directory, file, or symlink must fail; nothing is replaced.
        output_directory.mkdir()
        rows = block_lines(source, source_digest, start, end)
        pending = next(rows, None)
        digits = len(str(total))
        for range_start in range(start, end, blocks_per_range):
            range_end = min(range_start + blocks_per_range, end)
            path = output_directory / f"range-{range_start:0{digits}d}-{range_end:0{digits}d}.jsonl"
            output_digest = hashlib.sha256()
            with path.open("xb") as output:
                raw_header = json.dumps([1, total, range_start, range_end], separators=(",", ":")).encode() + b"\n"
                output.write(raw_header)
                output_digest.update(raw_header)
                while pending is not None and pending[0] < range_end:
                    raw = pending[1]
                    output.write(raw)
                    output_digest.update(raw)
                    pending = next(rows, None)
                output.write(b"[]\n")
                output_digest.update(b"[]\n")
            outputs.append({
                "path": str(path),
                "start": range_start,
                "end": range_end,
                "sha256": output_digest.hexdigest(),
            })
        if pending is not None:
            raise ValueError("source block remains outside the output ranges")
    manifest = {
        "schema": 1,
        "source": {
            "path": str(source_path),
            "sha256": source_digest.hexdigest(),
            "header": [1, total, start, end],
        },
        "blocks_per_range": blocks_per_range,
        "outputs": outputs,
    }
    manifest_path = output_directory / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(f"partitioned_range={start}..{end} files={len(outputs)} manifest={manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing Lean R range file")
    parser.add_argument("output_directory", type=Path, help="new output directory")
    parser.add_argument("blocks_per_range", type=int, help="measured block extent for each output range")
    arguments = parser.parse_args()
    try:
        split_parent(arguments.source, arguments.output_directory, arguments.blocks_per_range)
    except (OSError, ValueError) as error:
        print(f"split_pidec_parent: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
