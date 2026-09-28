#!/usr/bin/env python3
"""Create the retained C/D mutation classes from fresh numeric checker inputs.

The case definitions come from NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip.
Run PiDECActualMutations through validate.sh to check these files. Generating
them establishes no rejection result and no witness-opening validity.
"""

import argparse
import copy
import json
from pathlib import Path


# PiCCSInputCheck's selected Nightstream Goldilocks b=2, k_rho=16 schema.
MODULUS = 18446744069414584321
CHILDREN, MATRICES, DEGREE = 16, 14, 54


def numeric_json(value):
    return json.dumps(value, separators=(",", ":")) + "\n"


def vector(value, dimensions):
    if not dimensions:
        if type(value) is not int or not 0 <= value < MODULUS:
            raise ValueError("expected a canonical Goldilocks integer")
        return
    if type(value) is not list or len(value) != dimensions[0]:
        raise ValueError(f"expected vector width {dimensions[0]}")
    for item in value:
        vector(item, dimensions[1:])


def running(value):
    if type(value) is not list or len(value) != 5:
        raise ValueError("expected five running-claim fields")
    for item, shape in zip(value, (
        (28, 2), (CHILDREN, 22 * DEGREE), (CHILDREN, 270),
        (CHILDREN, DEGREE, 2), (CHILDREN, MATRICES, DEGREE, 2),
    )):
        vector(item, shape)


def read_numeric(path):
    text = path.read_bytes().decode("utf-8")
    value = json.loads(text)
    canonical = numeric_json(value)
    if text not in (canonical, canonical[:-1]):
        raise ValueError("expected canonical numeric JSON with an optional final newline")
    return value


def generate(input_path, children_path, output):
    ccs, children = read_numeric(input_path), read_numeric(children_path)
    if type(ccs) is not list or len(ccs) != 7 or type(ccs[0]) is not int or ccs[0] != 2:
        raise ValueError("expected PiCCS input schema 2")
    for item, shape in zip(ccs[1:6], (
        (22 * DEGREE,), (270,), (28, 10, 2),
        (CHILDREN + 1, DEGREE, 2), (CHILDREN + 1, MATRICES, DEGREE, 2),
    )):
        vector(item, shape)
    running(ccs[6])
    running(children)
    if any(word not in (0, 1, MODULUS - 1) for public in children[2] for word in public):
        raise ValueError("expected signed binary child public inputs")

    output.mkdir(exist_ok=False)
    mutations = output / "mutations"
    mutations.mkdir()
    cases = []

    def write(name, value, owner, text=None):
        path = mutations / f"{name}.json"
        with path.open("x") as destination:
            destination.write(numeric_json(value) if text is None else text)
        cases.append({"case": name, "file": str(path.relative_to(output)), "expected_owner": owner})

    def change(name, indices, action, owner="public_check"):
        value = copy.deepcopy(children)
        parent = value
        for index in indices[:-1]:
            parent = parent[index]
        parent[indices[-1]] = action(parent[indices[-1]])
        write(name, value, owner)

    def bump(word):
        return (word + 1) % MODULUS

    for child in range(CHILDREN):
        change(f"public_child_{child}_commitment", (1, child, 0), bump)
    change("public_child_public", (2, 0, 0), bump)
    change("public_shared_point", (0, 0, 0), bump)
    change("public_child_eval_K", (3, 0, 0, 0), bump)
    for matrix in range(MATRICES):
        change(f"public_child_eval_A{matrix}", (4, 0, matrix, 0, 0), bump)
    change("public_child_digit_range", (2, 0, 0), lambda _word: 2)

    for name, indices in (
        ("missing_child", (1,)), ("short_commitment", (1, 0)),
        ("short_public", (2, 0)), ("short_point", (0,)),
        ("short_eval_K", (3, 0)), ("missing_matrix", (4, 0)),
        ("short_eval_A", (4, 0, 0)),
    ):
        change(f"encoding_{name}", indices, lambda values: values[:-1], "decoder")
    change("encoding_noncanonical_field", (1, 0, 0), lambda _word: MODULUS, "decoder")
    write("encoding_extra_field", children + [0], "decoder")
    # Preserve the retained case name. Its alternate bytes are whitespace,
    # not a changed arithmetic value; the canonical parser must reject it.
    canonical = numeric_json(children)
    write("encoding_alternate_number", children, "decoder", canonical[:1] + " " + canonical[1:])

    changed_ccs = copy.deepcopy(ccs)
    changed_ccs[3][0][0][0] = bump(changed_ccs[3][0][0][0])
    bad_input = output / "invalid_first_round_constant.json"
    with bad_input.open("x") as destination:
        destination.write(numeric_json(changed_ccs))
    cases.append({
        "case": "invalid_first_round_constant", "file": bad_input.name,
        "expected_owner": "upstream_pi_ccs",
    })
    manifest = {
        "schema": 1,
        "source_input": str(input_path.resolve()), "source_children": str(children_path.resolve()),
        "mutation_directory": mutations.name, "changed_ccs_input": bad_input.name,
        "checker": "formal/nightstream-fprime/tests/PiDECActualMutations.lean",
        "cases": cases,
        "checker_internal_cases": ["unbounded_parent", "rejected_C_stops_D"],
        "scope": "Generated cases only. Run the Lean checker to establish rejection; Rust results are separate.",
    }
    with (output / "manifest.json").open("x") as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pi_ccs_input", type=Path)
    parser.add_argument("children", type=Path)
    parser.add_argument("new_output_directory", type=Path)
    args = parser.parse_args()
    manifest = generate(args.pi_ccs_input, args.children, args.new_output_directory)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
