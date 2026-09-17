#!/usr/bin/env python3
"""Project original or Lean-produced signed witnesses into the existing C/R input.

No C proof or expected output is read. This is a checked format projection,
not a substitute for source opening, state/public, or running-parent checks.
Bootstrap retains the existing check-owned-sources gate. Feedback takes the
independent Lean D claims and private ranges plus the Lean fresh witness/claim.
Constants are the selected Poseidon2HashChainV1 b=2, k_rho=16 profile.
"""
import argparse
from array import array
import json
import os
from pathlib import Path
import tempfile

P = 18446744069414584321
D, BLOCKS, LOGICAL, CHILDREN, PUBLIC, MATRICES = 54, 4685394, 253011231, 16, 270, 14
MASK = (1 << D) - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-JSON number: " + value)


def parse(text):
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=object_pairs, parse_constant=reject_constant)


def natural(value):
    require(type(value) is int and value >= 0, "expected a natural number")
    return value


def field(value):
    require(type(value) is int and 0 <= value < P, "noncanonical field")
    return value


def wrapped_field(value):
    require(isinstance(value, dict) and set(value) == {"value"}, "invalid field object")
    return field(value["value"])


def vector(value, count, decode):
    require(isinstance(value, list) and len(value) == count, "wrong vector width")
    return [decode(item) for item in value]


def extension(value):
    require(isinstance(value, dict) and set(value) == {"value", "_phantom"}
            and value["_phantom"] is None, "invalid extension object")
    return vector(value["value"], 2, wrapped_field)


def commitment(value):
    require(value["d"] == D and value["kappa"] == 22, "wrong commitment geometry")
    return vector(value["data"], 22 * D, wrapped_field)


def fresh_claim(path):
    value = read(path)
    require(value["m_in"] == PUBLIC and value["adv"] is None, "unexpected fresh claim")
    return commitment(value["c"]), vector(value["x"], PUBLIC, wrapped_field)


def public_matrix(value):
    require(value["rows"] == D and value["cols"] == 5, "wrong public matrix geometry")
    require(value["packed_signed_unit"] is None, "unexpected packed public matrix")
    if value["constant_hint"] is not None:
        require(value["data"] == [], "ambiguous constant public matrix")
        return [wrapped_field(value["constant_hint"])] * PUBLIC
    data = vector(value["data"], PUBLIC, wrapped_field)
    return [data[(index % D) * 5 + index // D] for index in range(PUBLIC)]


def padded_evaluation(value):
    values = vector(value, 64, extension)
    require(all(item == [0, 0] for item in values[D:]), "nonzero extension padding")
    return values[:D]


def original_running(envelope):
    require(natural(envelope["schema"]) == 1 and envelope["child_witness_count"] == CHILDREN,
            "wrong source envelope schema or child count")
    require(0 < natural(envelope["iteration"]) < P, "invalid active source counter")
    vector(envelope["z0"], 4, field)
    vector(envelope["current"], 4, field)
    claims = vector(envelope["running_claims"], CHILDREN, lambda item: item)
    require(envelope["running_parent"] is not None, "missing running parent")
    points, commitments, publics, pads, matrices = [], [], [], [], []
    for claim in claims:
        require(claim["m_in"] == PUBLIC and claim["adv"] is None, "unexpected running claim")
        points.append(vector(claim["r"], 28, extension))
        commitments.append(commitment(claim["c"]))
        publics.append(public_matrix(claim["X"]))
        pads.append(padded_evaluation(claim["eval_k"]))
        matrices.append(vector(claim["eval_a"], MATRICES, padded_evaluation))
    require(all(point == points[0] for point in points), "running points differ")
    return [points[0], commitments, publics, pads, matrices]


def numeric_running(path):
    value = vector(read(path), 5, lambda item: item)
    pair = lambda item: vector(item, 2, field)
    point = vector(value[0], 28, pair)
    commitments = vector(value[1], CHILDREN, lambda item: vector(item, 1188, field))
    publics = vector(value[2], CHILDREN, lambda item: vector(item, PUBLIC, field))
    pads = vector(value[3], CHILDREN, lambda item: vector(item, D, pair))
    matrices = vector(value[4], CHILDREN,
                      lambda item: vector(item, MATRICES, lambda row: vector(row, D, pair)))
    return [point, commitments, publics, pads, matrices]


def mask_pair(positive, negative):
    natural(positive)
    natural(negative)
    require((positive | negative) <= MASK and positive & negative == 0,
            "invalid signed-unit masks")
    return positive, negative


def witness(path, public, fresh=False):
    value = read(path)
    require(value["rows"] == D and value["cols"] == BLOCKS, "wrong witness geometry")
    require(value["data"] == [], "expected compact signed-unit witness")
    packed = value["packed_signed_unit"]
    if packed is None:
        require(value["constant_hint"] is not None
                and wrapped_field(value["constant_hint"]) == 0, "expected zero constant witness")
        masks = None
    else:
        require(value["constant_hint"] is None and packed["cols"] == BLOCKS,
                "ambiguous witness representation")
        require(vector(packed["values"], 3, wrapped_field) == [0, 1, P - 1],
                "wrong signed-unit alphabet")
        require(set(packed["bits"]) == {"ColumnMasks"}, "expected column masks")
        bits = packed["bits"]["ColumnMasks"]
        positive, negative = bits["positive"], bits["negative"]
        require(len(positive) == BLOCKS and len(negative) == BLOCKS, "wrong mask extent")
        for pos, neg in zip(positive, negative):
            mask_pair(pos, neg)
        masks = array("Q", positive), array("Q", negative)
        if fresh:
            require(((positive[-1] | negative[-1]) >> (LOGICAL % D)) == 0,
                    "nonzero fresh carrier tail")
    for index, expected in enumerate(public):
        pos, neg = (0, 0) if masks is None else (masks[0][index // D], masks[1][index // D])
        actual = 1 if (pos >> (index % D)) & 1 else P - 1 if (neg >> (index % D)) & 1 else 0
        require(actual == expected, "source witness/public mismatch")
    return masks


def emit(stream, block, entries):
    if entries:
        stream.write(json.dumps([block, entries], separators=(",", ":")) + "\n")


def original(source, stream):
    source = Path(source)
    c, x = fresh_claim(source / "fresh-claim.json")
    running = original_running(read(source / "envelope.json"))
    witnesses = [witness(source / "fresh-witness.json", x, fresh=True)]
    witnesses.extend(witness(source / f"digit-{child}.json", running[2][child])
                     for child in range(CHILDREN))
    for block in range(BLOCKS):
        entries = []
        for source_index, masks in enumerate(witnesses):
            if masks is not None:
                pos, neg = masks[0][block], masks[1][block]
                if pos | neg:
                    entries.append([source_index, pos, neg])
        emit(stream, block, entries)
    return [c, x, running]


def digit_record(stream, first, finish, previous):
    line = stream.readline()
    require(line != "", "missing D range terminator")
    value = parse(line)
    if value == []:
        require(stream.read(1) == "", "extra data after D range terminator")
        return None
    require(isinstance(value, list) and len(value) == 2, "invalid D block record")
    block = natural(value[0])
    require(first <= block < finish and block > previous, "unordered or out-of-range D block")
    require(isinstance(value[1], list) and value[1], "empty D block record")
    children = {}
    previous_child = -1
    for entry in value[1]:
        require(isinstance(entry, list) and len(entry) == 3, "invalid D child record")
        child = natural(entry[0])
        require(previous_child < child < CHILDREN, "unordered, duplicate, or invalid D child")
        previous_child = child
        pos, neg = mask_pair(entry[1], entry[2])
        require(pos | neg != 0, "zero D child record")
        children[child] = (pos, neg)
    return block, children


def feedback(args, stream):
    c, x = fresh_claim(args.fresh_claim)
    running = numeric_running(args.children)
    fresh = witness(args.fresh_witness, x, fresh=True)
    next_block = 0
    for path in args.ranges:
        with Path(path).open() as digits:
            header = vector(parse(digits.readline()), 6, natural)
            require(header[:4] == [1, D, CHILDREN, BLOCKS], "wrong D range header")
            first, finish = header[4:]
            require(first == next_block and first < finish <= BLOCKS, "D range gap/overlap")
            record = digit_record(digits, first, finish, first - 1)
            for block in range(first, finish):
                children = {}
                if record is not None and record[0] == block:
                    children = record[1]
                    record = digit_record(digits, first, finish, block)
                entries = []
                if fresh is not None:
                    pos, neg = fresh[0][block], fresh[1][block]
                    if pos | neg:
                        entries.append([0, pos, neg])
                for child in range(CHILDREN):
                    pos, neg = children.get(child, (0, 0))
                    if block < PUBLIC // D:
                        for lane in range(D):
                            actual = 1 if (pos >> lane) & 1 else P - 1 if (neg >> lane) & 1 else 0
                            require(actual == running[2][child][block * D + lane],
                                    "D witness/public mismatch")
                    if pos | neg:
                        entries.append([child + 1, pos, neg])
                emit(stream, block, entries)
            require(record is None, "unconsumed D block")
            next_block = finish
    require(next_block == BLOCKS, "incomplete D range coverage")
    return [c, x, running]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    bootstrap = modes.add_parser("original")
    bootstrap.add_argument("source_directory")
    bootstrap.add_argument("output_directory")
    next_fold = modes.add_parser("feedback")
    next_fold.add_argument("fresh_witness")
    next_fold.add_argument("fresh_claim")
    next_fold.add_argument("children")
    next_fold.add_argument("output_directory")
    next_fold.add_argument("ranges", nargs="+")
    args = parser.parse_args()
    output = Path(args.output_directory).absolute()
    require(not output.exists(), "output directory already exists")
    with tempfile.TemporaryDirectory(prefix="source-projection-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        with (temporary / "sources.jsonl").open("x") as stream:
            stream.write(json.dumps([1, D, CHILDREN + 1, BLOCKS], separators=(",", ":")) + "\n")
            public = original(args.source_directory, stream) if args.mode == "original" else feedback(args, stream)
            stream.write("[]\n")
        (temporary / "public.json").write_text(json.dumps(public, separators=(",", ":")) + "\n")
        require(not output.exists(), "output directory already exists")
        os.rename(temporary, output)
    print(json.dumps({"event": "source_only_projection_complete", "mode": args.mode,
                      "sources": CHILDREN + 1, "blocks": BLOCKS, "output": str(output)}))


if __name__ == "__main__":
    main()
