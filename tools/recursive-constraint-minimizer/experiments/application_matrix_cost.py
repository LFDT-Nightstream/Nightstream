"""Count normalized application matrix entries from Lean-exported operands.

The counter combines coefficients after every affine operation. It never
expands the semantic expression trees across Poseidon2 partial rounds.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

P = 18446744069414584321


def add(*forms):
    result = collections.defaultdict(int)
    for form in forms:
        for column, coefficient in form.items():
            result[column] = (result[column] + coefficient) % P
    return {column: value for column, value in result.items() if value}


def scale(coefficient, form):
    return {column: value * coefficient % P for column, value in form.items()
            if value * coefficient % P}


def entries(values):
    return add(*({column: coefficient} for column, coefficient in values))


def field(block, slot):
    kind, count, start = block
    assert kind == 2 and 0 <= slot < count
    return {start + slot * 41 + digit: pow(3, digit, P) for digit in range(41)}


def external(state):
    weights = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
    blocks = [add(*(scale(weight, state[base + index])
                    for index, weight in enumerate(weights[lane])))
              for base in (0, 4) for lane in range(4)]
    return [add(blocks[lane], blocks[lane % 4], blocks[lane % 4 + 4])
            for lane in range(8)]


def input_state(rules, invocation, one):
    state = [{} for _ in range(8)]
    for region, term in rules:
        first, count, first_lane, lanes = region
        if not first <= invocation < first + count:
            continue
        major = invocation - first
        for minor in range(lanes):
            lane = first_lane + minor
            if term[0] == 0:
                _, block, base, stride, lane_stride = term
                form = field(block, base + major * stride + minor * lane_stride)
            elif term[0] == 2:
                _, block, base, stride = term
                form = external([field(block, base + major * stride + index)
                                 for index in range(8)])[minor]
            elif term[0] == 4:
                _, values, lane_count = term
                value = values[major * lane_count + minor]
                form = {} if value == [0] else {one: value[1]}
            else:
                raise ValueError(f"Unsupported input term: {term[0]}")
            state[lane] = add(state[lane], form)
    return state


def measure(data):
    program, initial, internal, terminal, diagonal = data
    counts = [0] * 14
    rows = 0
    for opcode, block in program:
        if opcode == 1:
            one, values = block
            for value in values:
                counts[1] += 1
                counts[4] += len(entries(value))
                rows += 1
            continue
        if opcode != 2:
            raise ValueError(f"Unsupported matrix opcode: {opcode}")
        invocations, one, retained, rules = block
        for invocation in range(invocations):
            state = external(input_state(rules, invocation, one))
            next_slot = invocation * 86

            def sbox(value):
                nonlocal next_slot, rows
                output = field(retained, next_slot)
                counts[1] += 1
                counts[4] += len(output)
                counts[5] += len(value)
                next_slot += 1
                rows += 1
                return output

            for constants in initial:
                state = external([sbox(add(form, {one: constant}))
                                  for form, constant in zip(state, constants, strict=True)])
            for constant in internal:
                state[0] = sbox(add(state[0], {one: constant}))
                total = add(*state)
                state = [add(scale(coefficient, form), total)
                         for coefficient, form in zip(diagonal, state, strict=True)]
            for constants in terminal:
                state = external([sbox(add(form, {one: constant}))
                                  for form, constant in zip(state, constants, strict=True)])
            assert next_slot == (invocation + 1) * 86
    return {"logical_rows": rows, "matrix_nonzeros_by_port": counts,
            "matrix_nonzeros": sum(counts)}


def baseline_application(path):
    raw = path.read_bytes()
    package = json.loads(raw)
    first, count = package[3][7:9]
    candidates = [value for opcode, value in package[2]
                  if opcode == 0 and value[0] == [0, [[first, count]]]]
    assert len(candidates) == 1
    _, one, substitution, projection = candidates[0]
    ranges, grids = substitution
    assert not grids and projection == [0]

    def source(column):
        for start, length, block, offset in ranges:
            if start <= column < start + length:
                return field(block, offset + column - start)
        raise ValueError(f"Application source column {column} is not retained")

    def combination(value):
        constant, terms = value
        return add({one: constant}, *(scale(coefficient, source(column))
                                     for column, coefficient in terms))

    counts = [0] * 14
    covered = set()

    def row(ordinal, left, right, output):
        if not first <= ordinal < first + count:
            return
        assert ordinal not in covered
        covered.add(ordinal)
        counts[1] += 1
        for port, form in zip((2, 3, 4), (left, right, output), strict=True):
            counts[port] += len(combination(form))

    for ordinal, output, left, right in package[1][11]:
        row(ordinal, left, right, [0, [[output, 1]]])
    for ordinal, left, right, output in package[1][12]:
        row(ordinal, left, right, output)
    assert len(covered) == count
    return {"logical_rows": count, "matrix_nonzeros_by_port": counts,
            "matrix_nonzeros": sum(counts), "package_sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = measure(json.loads(raw))
    result["baseline_application"] = baseline_application(args.baseline)
    result["matrix_nonzero_reduction"] = result["baseline_application"]["matrix_nonzeros"] - result["matrix_nonzeros"]
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["scope"] = "Independent normalized sparse arithmetic over exact Lean application operands."
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
