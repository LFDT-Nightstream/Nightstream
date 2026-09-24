"""Count normalized entries of the Lean-emitted wide matrix program.

Field slots stay as 41-coordinate atoms during linear arithmetic. Counting
expands an atom only when it overlaps another atom. The baseline is checked
against the previously measured full matrix vector before reusing any block.
"""

import argparse
import collections
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path

P = 18446744069414584321
WIDTHS = (1, 1, 41)
EXPECTED = [229698543, 3483539, 272967007, 35832394, 813066947,
            1502256188, 0, 104652, 0, 0, 0, 0, 0, 0]
POWERS = [[pow(point, lane, P) for lane in range(54)] for point in range(108)]


def add(*forms):
    result = collections.defaultdict(int)
    for form in forms:
        for atom, coefficient in form.items():
            result[atom] = (result[atom] + coefficient) % P
    return {atom: value for atom, value in result.items() if value}


def scale(value, form):
    return {atom: weight * value % P for atom, weight in form.items() if weight * value % P}


def scalar(column, value):
    return {(column, 1): value % P} if value % P else {}


def field(block, slot):
    kind, count, start = block
    if kind not in (0, 1, 2) or not 0 <= slot < count:
        raise ValueError("invalid retained slot")
    width = WIDTHS[kind]
    return {(start + slot * width, width): 1}


def wire(entries):
    return add(*(scalar(column, value) for column, value in entries))


def nonzeros(form):
    ordered = sorted(form)
    if all(start + width <= next_start
           for (start, width), (next_start, _) in zip(ordered, ordered[1:])):
        return sum(width for _, width in ordered)
    expanded = collections.defaultdict(int)
    for (start, width), value in form.items():
        for digit in range(width):
            expanded[start + digit] = (expanded[start + digit] + value * pow(3, digit, P)) % P
    return sum(value != 0 for value in expanded.values())


def external(state):
    weights = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
    blocks = [add(*(scale(weight, state[base + index])
                    for index, weight in enumerate(weights[lane])))
              for base in (0, 4) for lane in range(4)]
    return [add(blocks[lane], blocks[lane % 4], blocks[lane % 4 + 4]) for lane in range(8)]


def project(projection, column):
    if projection == [0]:
        return column
    if projection[0] != 1:
        raise ValueError("unknown column projection")
    matches = [target + column - source for source, target, count in projection[1]
               if source <= column < source + count]
    if len(matches) != 1:
        raise ValueError(f"column {column} has {len(matches)} projections")
    return matches[0]


def injective(projection, source_width, target_width):
    if projection[0] != 1:
        raise ValueError("expected an explicit retained-column map")
    domains, images = [], []
    for source, target, count in projection[1]:
        if source < 0 or target < 0 or source + count > source_width or target + count > target_width:
            raise ValueError("projection range is out of bounds")
        if count:
            domains.append((source, source + count))
            images.append((target, target + count))
    for intervals in (domains, images):
        intervals.sort()
        if any(end > later for (_, end), (later, _) in zip(intervals, intervals[1:])):
            raise ValueError("projection is not injective")


def source(substitution, column):
    ranges, grids = substitution
    matches = []
    for start, count, block, slot in ranges:
        if start <= column < start + count:
            matches.append(field(block, slot + column - start))
    for grid in grids:
        start, major_count, major_stride, minor_count, minor_stride, count, block, mode, slot, major_slot, minor_slot = grid
        if column < start or major_stride == 0 or minor_stride == 0:
            continue
        major, offset = divmod(column - start, major_stride)
        minor, offset = divmod(offset, minor_stride)
        if major >= major_count or minor >= minor_count or offset >= count:
            continue
        base = slot + major * major_slot + minor * minor_slot
        if mode == 0:
            matches.append(field(block, base + offset))
        elif mode == 1 and offset < 8:
            matches.append(external([field(block, base + lane) for lane in range(8)])[offset])
        else:
            raise ValueError("unknown or invalid source grid")
    if len(matches) != 1:
        raise ValueError(f"source {column} has {len(matches)} owners")
    return matches[0]


def combination(value, substitution, one, projection=(0,)):
    constant, terms = value
    identity = list(projection) == [0]
    return add(scalar(one, constant), *(scale(coefficient, source(substitution,
        column if identity else project(projection, column))) for column, coefficient in terms))


def schedule(value):
    if value[0] == 0:
        return itertools.chain.from_iterable(range(first, first + count) for first, count in value[1])
    if value[0] == 1:
        return iter(value[1])
    raise ValueError("unknown row schedule")


def poseidon_input(rules, invocation, one):
    state = [{} for _ in range(8)]
    for region, term in rules:
        first, count, first_lane, lanes = region
        if not first <= invocation < first + count:
            continue
        major = invocation - first
        for minor in range(lanes):
            lane = first_lane + minor
            tag = term[0]
            if tag == 0:
                _, block, base, stride, lane_stride = term
                form = field(block, base + major * stride + minor * lane_stride)
            elif tag == 1:
                form = scalar(one, term[1])
            elif tag == 2:
                _, block, base, stride = term
                form = external([field(block, base + major * stride + i) for i in range(8)])[minor]
            elif tag == 3:
                _, block, tags, required, base, stride, lane_stride = term
                form = field(block, base + major * stride + minor * lane_stride) if tags[major] == required else {}
            elif tag == 4:
                _, values, lane_count = term
                value = values[major * lane_count + minor]
                form = {} if value == [0] else scalar(one, value[1])
            elif tag == 5:
                _, values, substitution, tags, required, lane_count = term
                form = combination(values[major * lane_count + minor], substitution, one) if tags[major] == required else {}
            elif tag == 6:
                _, values, lane_count = term
                form = wire(values[major * lane_count + minor])
            else:
                raise ValueError(f"unknown Poseidon input term {tag}")
            state[lane] = add(state[lane], form)
    return state


def affine_grid(rules, coordinates, one):
    result = {}
    for region, term in rules:
        offsets = tuple(value - start for value, start in zip(coordinates, region[::2]))
        if not all(0 <= offset < count for offset, count in zip(offsets, region[1::2])):
            continue
        if term[0] == 0:
            _, block, base, major, middle, minor, coefficient = term
            form = scale(coefficient, field(block, base + sum(x * y for x, y in zip(offsets, (major, middle, minor)))))
        elif term[0] == 1:
            form = scalar(one, term[1])
        else:
            raise ValueError("unknown affine-grid term")
        result = add(result, form)
    return result


class Counter:
    def __init__(self, package, operands):
        self.rows = {}
        for row, output, left, right in package[1][11]:
            self.put(row, (left, right, [0, [[output, 1]]]))
        for row, left, right, output in package[1][12]:
            self.put(row, (left, right, output))
        self.initial, self.internal, self.terminal, self.diagonal = operands[5:9]

    def put(self, row, value):
        if row in self.rows:
            raise ValueError("duplicate physical source row")
        self.rows[row] = value

    def count(self, instruction):
        tag, block = instruction[:2]
        counts = [0] * 14
        rows = 0
        if tag in (0, 6):
            indices, one, substitution, projection = block
            template = instruction[2] if tag == 6 else None
            for index in schedule(indices):
                triple = template[3 * index:3 * index + 3] if template is not None else self.rows[index]
                if len(triple) != 3:
                    raise ValueError("incomplete ordinary template")
                counts[1] += 1
                for port, value in zip((2, 3, 4), triple):
                    counts[port] += nonzeros(combination(value, substitution, one, projection))
                rows += 1
        elif tag == 1:
            one, values = block
            for value in values:
                counts[1] += 1
                counts[4] += nonzeros(wire(value))
                rows += 1
        elif tag == 2:
            invocations, one, retained, rules = block
            for invocation in range(invocations):
                state = external(poseidon_input(rules, invocation, one))
                slot = invocation * 86

                def sbox(value):
                    nonlocal slot, rows
                    output = field(retained, slot)
                    counts[1] += 1
                    counts[4] += nonzeros(output)
                    counts[5] += nonzeros(value)
                    slot += 1
                    rows += 1
                    return output

                for constants in self.initial:
                    state = external([sbox(add(form, scalar(one, constant))) for form, constant in zip(state, constants, strict=True)])
                for constant in self.internal:
                    state[0] = sbox(add(state[0], scalar(one, constant)))
                    total = add(*state)
                    state = [add(scale(coefficient, form), total) for coefficient, form in zip(self.diagonal, state, strict=True)]
                for constants in self.terminal:
                    state = external([sbox(add(form, scalar(one, constant))) for form, constant in zip(state, constants, strict=True)])
                if slot != (invocation + 1) * 86:
                    raise ValueError("wrong S-box count")
        elif tag == 3:
            if len(block) == 8:
                families, one, challenges, base, stride, inputs, output, quotient = block
                challenge = lambda src: [add(field(challenges, base + src * stride + lane), scalar(one, -2)) for lane in range(54)]
            elif len(block) == 7:
                families, one, challenges, stride, inputs, output, quotient = block
                challenge = lambda src: [wire(challenges[src * stride + lane]) for lane in range(54)]
            else:
                raise ValueError("invalid product block")
            offset = 0
            for sources, blocks, cells in families:
                size = blocks * 54 * cells
                for src in range(sources):
                    left = challenge(src)
                    for segment in range(blocks):
                        for cell in range(cells):
                            slots = [offset + src * size + (segment * 54 + lane) * cells + cell for lane in range(54)]
                            right = [source(inputs, slot) for slot in slots]
                            out = [field(output, slot) for slot in slots]
                            group = [field(quotient, slot) for slot in slots]
                            prior = [{} for _ in slots] if src == 0 else [field(output, slot - size) for slot in slots]
                            for point, powers in enumerate(POWERS):
                                def evaluate(state):
                                    return add(*(scale(weight, form) for weight, form in zip(powers, state)))
                                counts[0] += nonzeros(evaluate(left))
                                counts[2] += nonzeros(evaluate(right))
                                counts[4] += nonzeros(add(evaluate(out), scale(-1, evaluate(prior)),
                                    scale((pow(point, 54, P) + pow(point, 27, P) + 1) % P, evaluate(group))))
                                counts[7] += 1
                                rows += 1
                offset += sources * size
        elif tag == 4:
            shape, one, left, right, output = block
            for coordinates in itertools.product(*(range(count) for count in shape)):
                counts[1] += 1
                for port, rules in zip((2, 3, 4), (left, right, output)):
                    counts[port] += nonzeros(affine_grid(rules, coordinates, one))
                rows += 1
        else:
            raise ValueError(f"unsupported matrix opcode {tag}")
        return {"rows": rows, "nonzeros_by_port": counts, "nonzeros": sum(counts)}


COUNTER = None


def worker(job):
    index, instruction = job
    return index, COUNTER.count(instruction)


def run_parallel(instructions):
    results = [None] * len(instructions)
    workers = min(os.cpu_count() or 1, len(instructions))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for index, result in pool.imap_unordered(worker, enumerate(instructions), chunksize=1):
            results[index] = result
            print(json.dumps({"block": index, "opcode": instructions[index][0], **result}), flush=True)
    return results


def totals(results):
    counts = [sum(result["nonzeros_by_port"][port] for result in results) for port in range(14)]
    return {"rows": sum(result["rows"] for result in results), "nonzeros_by_port": counts, "nonzeros": sum(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operands", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    operands_raw, baseline_raw = args.operands.read_bytes(), args.baseline.read_bytes()
    operands, package = json.loads(operands_raw), json.loads(baseline_raw)
    if operands[0] != 1 or operands[2] != package[2]:
        raise ValueError("Lean baseline program differs from the selected source archive")
    global COUNTER
    COUNTER = Counter(package, operands)
    print(json.dumps({"status": "baseline", "workers": min(os.cpu_count() or 1, len(package[2]))}), flush=True)
    baseline = run_parallel(package[2])
    baseline_total = totals(baseline)
    if baseline_total["nonzeros_by_port"] != EXPECTED or baseline_total["rows"] != 3588191:
        raise ValueError(f"baseline control failed: {baseline_total}")
    lookup = {json.dumps(block, separators=(",", ":")): result for block, result in zip(package[2], baseline)}
    candidate, changed, positions = [], [], []
    for instruction in operands[1]:
        if instruction[0] == 5:
            _, source_width, projection, child = instruction
            injective(projection, source_width, operands[3])
            key = json.dumps(child, separators=(",", ":"))
            if key not in lookup:
                raise ValueError("mapped block has no byte-equal baseline control")
            candidate.append(lookup[key])
        else:
            positions.append(len(candidate))
            changed.append(instruction)
            candidate.append(None)
    print(json.dumps({"status": "candidate", "changed_blocks": len(changed)}), flush=True)
    for position, result in zip(positions, run_parallel(changed)):
        candidate[position] = result
    candidate_total = totals(candidate)
    if candidate_total["rows"] != operands[4]:
        raise ValueError("candidate row count differs from Lean")
    result = {"baseline": baseline_total, "candidate": candidate_total,
              "baseline_blocks": baseline, "candidate_blocks": candidate,
              "operands_sha256": hashlib.sha256(operands_raw).hexdigest(),
              "baseline_sha256": hashlib.sha256(baseline_raw).hexdigest(),
              "scope": "Exact sparse arithmetic. Reused byte-equal blocks preserve counts under checked injective maps; their complete read support is proved by Wide.MatrixProgram.fixedPoint_exact and phase support."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "baseline": baseline_total, "candidate": candidate_total}), flush=True)


if __name__ == "__main__":
    main()
