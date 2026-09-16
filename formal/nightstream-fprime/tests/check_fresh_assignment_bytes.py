"""Compare Lean-produced assignment bytes with every selected Rust carrier cell."""
import argparse
import json
from pathlib import Path

WIDTH = 54
TABLE = tuple(
    bytes(1 if (code >> 8) & (1 << bit) else 255 if code & (1 << bit) else 0
          for bit in range(8))
    for code in range(1 << 16)
)

def equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: coefficient bytes differ")

class Target:
    def __init__(self, path):
        value = json.loads(Path(path).read_text())
        assert value["rows"] == WIDTH and value["data"] == []
        assert value["constant_hint"] is None
        packed = value["packed_signed_unit"]
        assert packed["cols"] == value["cols"]
        assert packed["values"] == [{"value": 0}, {"value": 1}, {"value": 18446744069414584320}]
        masks = packed["bits"]["ColumnMasks"]
        self.positive = masks["positive"]
        self.negative = masks["negative"]
        self.columns = value["cols"]
        assert len(self.positive) == len(self.negative) == self.columns
        for positive, negative in zip(self.positive, self.negative):
            assert 0 <= positive < 1 << WIDTH and 0 <= negative < 1 << WIDTH
            assert positive & negative == 0

    def interval(self, first, finish):
        assert 0 <= first <= finish <= self.columns * WIDTH
        result = bytearray()
        for column in range(first // WIDTH, (finish + WIDTH - 1) // WIDTH):
            positive, negative = self.positive[column], self.negative[column]
            result.extend(b"".join(
                TABLE[((positive >> shift) & 255) * 256 + ((negative >> shift) & 255)]
                for shift in range(0, WIDTH, 8)
            )[:WIDTH])
        start = first % WIDTH
        return result[start:start + finish - first]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    target = Target(args.target)
    records = []
    logical_width = None
    public_width = None
    for directory_name in args.directories:
        directory = Path(directory_name)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["schema"] == 1
        if logical_width is None:
            logical_width = manifest["logical_width"]
            public_width = manifest["public_width"]
        assert logical_width == manifest["logical_width"]
        assert public_width == manifest["public_width"] == 270
        public = (directory / "public.bin").read_bytes()
        assert len(public) == public_width
        equal(public, target.interval(0, public_width), "public prefix")
        local = manifest["blocks"]
        assert [row["ordinal"] for row in local] == list(
            range(manifest["first_block"], manifest["finish_block"]))
        assert {entry.name for entry in directory.iterdir()} == {
            "manifest.json", "public.bin", *(row["file"] for row in local)}
        for row in local:
            assert row["file"] == f'block-{row["ordinal"]}.bin'
            assert public_width <= row["first"] < row["finish"] <= logical_width
            actual = (directory / row["file"]).read_bytes()
            assert len(actual) == row["finish"] - row["first"]
            expected = target.interval(row["first"], row["finish"])
            equal(actual, expected, row["file"])
            records.append((row, directory))
    records.sort(key=lambda pair: pair[0]["ordinal"])
    assert len({row["ordinal"] for row, _ in records}) == len(records)
    compared = public_width + sum(row["finish"] - row["first"] for row, _ in records)
    tail = target.columns * WIDTH - logical_width
    if args.complete:
        assert [row["ordinal"] for row, _ in records] == list(range(30))
        end = public_width
        for row, _ in records:
            assert row["first"] == end
            end = row["finish"]
        assert end == logical_width
        equal(target.interval(logical_width, target.columns * WIDTH), bytes(tail), "zero tail")
        compared += tail
    row, directory = records[-1]
    actual = (directory / row["file"]).read_bytes()
    changed = bytearray(actual)
    changed[-1] = 1 if changed[-1] == 0 else 0
    try:
        equal(actual, changed, "changed final target coefficient")
    except ValueError:
        pass
    else:
        raise AssertionError("changed target accepted")
    print(json.dumps({
        "status": "passed", "complete": args.complete,
        "blocks": len(records), "coefficients": compared,
        "logical_width": logical_width, "tail_coefficients": tail if args.complete else 0,
        "changed_target_rejected": True,
    }))

if __name__ == "__main__":
    main()
