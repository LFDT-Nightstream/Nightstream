"""Assemble completed Lean assignment blocks; no Rust output is an input."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carrier_output")
    parser.add_argument("witness_output")
    parser.add_argument("directories", nargs="+")
    args = parser.parse_args()
    carrier_path, witness_path = Path(args.carrier_output), Path(args.witness_output)
    assert carrier_path.resolve() != witness_path.resolve()
    assert not carrier_path.exists() and not witness_path.exists()
    rows = []
    prefix = None
    logical_width = None
    physical_fields = None
    for name in args.directories:
        directory = Path(name)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["schema"] == 1 and manifest["public_width"] == 270
        public = (directory / "public.bin").read_bytes()
        assert len(public) == 270
        if prefix is None:
            prefix, logical_width, physical_fields = public, manifest["logical_width"], manifest["physical_fields"]
        assert prefix == public and logical_width == manifest["logical_width"]
        assert physical_fields == manifest["physical_fields"]
        local = manifest["blocks"]
        assert [row["ordinal"] for row in local] == list(
            range(manifest["first_block"], manifest["finish_block"]))
        assert {entry.name for entry in directory.iterdir()} == {
            "manifest.json", "public.bin", *(row["file"] for row in local)}
        rows.extend((row, directory) for row in local)
    rows.sort(key=lambda pair: pair[0]["ordinal"])
    assert [row["ordinal"] for row, _ in rows] == list(range(30))
    next_column = 270
    for row, directory in rows:
        assert row["file"] == f'block-{row["ordinal"]}.bin'
        assert row["first"] == next_column < row["finish"] <= logical_width
        assert (directory / row["file"]).stat().st_size == row["finish"] - row["first"]
        next_column = row["finish"]
    assert next_column == logical_width
    invalid = bytes(0 if value in (0, 1, 255) else 1 for value in range(256))
    assert 1 not in prefix.translate(invalid)
    with carrier_path.open("xb") as output:
        output.write(prefix)
        for row, directory in rows:
            data = (directory / row["file"]).read_bytes()
            assert 1 not in data.translate(invalid)
            output.write(data)
        output.write(bytes((-logical_width) % 54))
    carrier = carrier_path.read_bytes()
    positive, negative = [], []
    for column in range(len(carrier) // 54):
        pos = neg = 0
        for lane, value in enumerate(carrier[column * 54:(column + 1) * 54]):
            if value == 1:
                pos |= 1 << lane
            elif value == 255:
                neg |= 1 << lane
        positive.append(pos)
        negative.append(neg)
    columns = len(positive)
    witness = {
        "rows": 54, "cols": columns, "data": [], "constant_hint": None,
        "packed_signed_unit": {
            "bits": {"ColumnMasks": {"positive": positive, "negative": negative}},
            "values": [{"value": 0}, {"value": 1}, {"value": 18446744069414584320}],
            "cols": columns,
        },
    }
    with witness_path.open("x") as output:
        json.dump(witness, output, separators=(",", ":"))
    print(json.dumps({"status": "passed", "logical_width": logical_width,
                      "carrier_coefficients": len(carrier), "tail_coefficients": len(carrier) - logical_width,
                      "physical_fields": physical_fields, "blocks": columns}))

if __name__ == "__main__":
    main()
