#!/usr/bin/env python3
"""Exact arithmetic controls for two restricted Poseidon reduction candidates.

Run under `timeout --signal=KILL 300`, per the root AGENTS.md test cap.
No solver result is a Lean proof or a global Poseidon lower bound.
"""

import json
import math
from pathlib import Path
import shutil
import subprocess
import time


P = 18446744069414584321
CURRENT_TRITS = 41
SHORT_TRITS = CURRENT_TRITS - 1
SBOX_DEGREE = 7
COMPOSED_DEGREE = SBOX_DEGREE**2
SELECTED_DEGREE_BOUND = 9
CORE_DEGREE_BOUND = SELECTED_DEGREE_BOUND - 1


def field(value):
    return f"(as ff{value % P} F)"


def field_header():
    return ["(set-logic QF_FF)", f"(define-sort F () (_ FiniteField {P}))"]


def power(term, exponent):
    return field(1) if exponent == 0 else f"(ff.mul {' '.join([term] * exponent)})"


def lift_query(width, target, witness=None):
    lines = ["(set-logic QF_LIA)", "(declare-const quotient Int)"]
    for index in range(width):
        lines.extend([
            f"(declare-const d{index} Int)",
            f"(assert (and (<= (- 1) d{index}) (<= d{index} 1)))",
        ])
    terms = " ".join(f"(* {3**index} d{index})" for index in range(width))
    lines.append(f"(assert (= (+ {terms}) (+ {target} (* {P} quotient))))")
    if witness is not None:
        assert len(witness) == width
        lines.append("(assert (= quotient 0))")
        for index, value in enumerate(witness):
            integer = f"(- {-value})" if value < 0 else str(value)
            lines.append(f"(assert (= d{index} {integer}))")
    return "\n".join(lines + ["(check-sat)"]) + "\n"


def coefficient_query(monomials, normalize_first=False):
    """Coefficient equations after substituting Y = X^49, over F_P."""
    buckets = {}
    lines = field_header()
    for index, (x_degree, y_degree) in enumerate(monomials):
        lines.append(f"(declare-const c{index} F)")
        exponent = x_degree + COMPOSED_DEGREE * y_degree
        buckets.setdefault(exponent, []).append(f"c{index}")
    for coefficients in buckets.values():
        total = coefficients[0] if len(coefficients) == 1 else f"(ff.add {' '.join(coefficients)})"
        lines.append(f"(assert (= {total} {field(0)}))")
    nonzero = " ".join(f"(not (= c{i} {field(0)}))" for i in range(len(monomials)))
    lines.append(f"(assert (or {nonzero}))")
    if normalize_first:
        lines.append(f"(assert (= c0 {field(1)}))")
    return "\n".join(lines + ["(check-sat)"]) + "\n"


def solve(solver, name, smt2, expected):
    started = time.monotonic()
    run = subprocess.run(
        [solver, "--lang=smt2", "--ff-solver=gb"],
        input=smt2, text=True, capture_output=True, check=True,
    )
    status = run.stdout.strip().splitlines()[0]
    assert status == expected, (name, expected, run.stdout, run.stderr)
    result = {
        "status": status,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "stdout": run.stdout,
        "stderr": run.stderr,
        "smt2": smt2,
    }
    print(f"{name}: {status}", flush=True)
    return result


def main():
    solver = shutil.which("cvc5")
    assert solver, "cvc5 with finite-field support must be on PATH"
    version = subprocess.check_output([solver, "--version"], text=True).splitlines()[0]

    capacity = 3**SHORT_TRITS
    radius = (capacity - 1) // 2
    missing = radius + 1
    assert capacity < P and 0 < missing < P
    # Every signed sum lies in [-radius, radius]. No congruent representative
    # of `missing` is in that interval: q >= 0 is too high, q <= -1 too low.
    assert missing > radius and missing - P < -radius
    inverse_seven = pow(SBOX_DEGREE, -1, P - 1)
    assert math.gcd(SBOX_DEGREE, P - 1) == 1
    root = pow(missing, inverse_seven, P)
    assert pow(root, SBOX_DEGREE, P) == missing
    witness = [-1] * SHORT_TRITS + [1]
    reconstructed = sum(digit * 3**index for index, digit in enumerate(witness))
    assert reconstructed == missing
    truncated = sum(digit * 3**index for index, digit in enumerate(witness[:-1])) % P
    assert truncated != missing

    runs = {}
    runs["forty_trits_missing_value"] = solve(
        solver, "40-trit missing value", lift_query(SHORT_TRITS, missing), "unsat",
    )
    runs["forty_one_trits_positive_control"] = solve(
        solver, "41-trit explicit witness", lift_query(CURRENT_TRITS, missing, witness), "sat",
    )
    root_query = field_header() + [
        "(declare-const x F)", f"(assert (= x {field(root)}))",
        f"(assert (= {power('x', SBOX_DEGREE)} {field(missing)}))", "(check-sat)",
    ]
    runs["missing_value_is_sbox_output"] = solve(
        solver, "missing value has seventh root", "\n".join(root_query) + "\n", "sat",
    )

    monomials = [
        (i, j)
        for j in range(CORE_DEGREE_BOUND + 1)
        for i in range(CORE_DEGREE_BOUND - j + 1)
    ]
    exponents = [i + COMPOSED_DEGREE * j for i, j in monomials]
    expanded_degree = max(exponents)
    assert len(monomials) == (CORE_DEGREE_BOUND + 1) * (CORE_DEGREE_BOUND + 2) // 2
    assert len(set(exponents)) == len(monomials)
    assert expanded_degree == CORE_DEGREE_BOUND * COMPOSED_DEGREE < P
    # Independent coefficient certificate: each column is a distinct standard
    # basis vector, so selecting those coefficient rows gives the identity.
    certificate = [
        [int(row_exponent == column_exponent) for column_exponent in exponents]
        for row_exponent in exponents
    ]
    assert all(
        value == int(row == column)
        for row, entries in enumerate(certificate)
        for column, value in enumerate(entries)
    )
    runs["no_degree_eight_bivariate_relation"] = solve(
        solver, "degree-eight coefficient search", coefficient_query(monomials), "unsat",
    )
    positive_support = [(0, 1), (COMPOSED_DEGREE, 0)]
    assert [i + COMPOSED_DEGREE * j for i, j in positive_support] == [49, 49]
    assert (1 + (P - 1)) % P == 0
    runs["degree_forty_nine_positive_control"] = solve(
        solver, "degree-49 relation Y-X^49",
        coefficient_query(positive_support, normalize_first=True), "sat",
    )

    wrong_output = pow(2, SBOX_DEGREE, P)
    correct_output = pow(2, COMPOSED_DEGREE, P)
    assert wrong_output != correct_output
    shortcut = field_header() + [
        "(declare-const x F)", "(declare-const y F)",
        f"(assert (= x {field(2)}))", f"(assert (= y {power('x', SBOX_DEGREE)}))",
        f"(assert (not (= y {power('x', COMPOSED_DEGREE)})))", "(check-sat)",
    ]
    runs["wrong_seventh_power_fusion"] = solve(
        solver, "incorrect fusion counterexample", "\n".join(shortcut) + "\n", "sat",
    )
    dropped_link = field_header() + [
        "(declare-const x F)", "(declare-const z F)", "(declare-const y F)",
        f"(assert (= x {field(0)}))", f"(assert (= z {field(1)}))",
        f"(assert (= y {power('z', SBOX_DEGREE)}))",
        f"(assert (not (= y {power('x', COMPOSED_DEGREE)})))", "(check-sat)",
    ]
    assert pow(1, SBOX_DEGREE, P) != pow(0, COMPOSED_DEGREE, P)
    runs["dropped_intermediate_link"] = solve(
        solver, "dropped first S-box counterexample", "\n".join(dropped_link) + "\n", "sat",
    )

    report = {
        "scope": "Untrusted solver evidence for two restricted candidates, with exact integer/coefficient replay. No global Poseidon lower bound or production change.",
        "checkpoint": "8b7c07d8", "solver": version, "field_modulus": P,
        "encoding": {
            "current_trits": CURRENT_TRITS, "candidate_trits": SHORT_TRITS,
            "candidate_capacity": capacity, "signed_radius": radius,
            "missing_field_value": missing, "seventh_root": root,
            "inverse_seven_mod_p_minus_one": inverse_seven,
            "seventh_power_replay": pow(root, SBOX_DEGREE, P),
            "forty_one_trit_witness": witness, "truncated_value": truncated,
            "translation_obligation": "Lean must equate signed-trit field recomposition with the exact integer sum modulo p, prove its interval bound, and connect the arbitrary S-box input to any selected protocol context. No bitvectors are used.",
            "generality": "The cardinality 3^40<p also excludes any total independent encoding of all field values with 40 signed-unit coordinates. The explicit missing value is specific to radix-three reconstruction.",
        },
        "pair_elimination": {
            "candidate": "A nonzero polynomial R(X,Y) of total degree at most 8 that vanishes for every Y=X^49 over Goldilocks, with no additional witness variables.",
            "degree_authority": "The existing selected relation has strict degree bound nine, so this candidate class has total degree at most eight; no profile change.",
            "monomial_count": len(monomials), "expanded_degree": expanded_degree,
            "enough_distinct_nodes_if_evaluated": expanded_degree + 1,
            "used_coefficient_certificate_instead_of_nodes": True,
            "coefficient_rank": len(monomials),
            "monomial_exponent_map": [
                {"x_degree": i, "y_degree": j, "substituted_exponent": i + COMPOSED_DEGREE * j}
                for i, j in monomials
            ],
            "proof_obligation": "Lean must justify substitution, degree<field-cardinality, polynomial equality from universal field vanishing, and the coefficient-map injection. cvc5 checks only the displayed coefficient equations.",
            "positive_control": "Y-X^49; coefficients 1 and -1; total degree 49, outside the selected degree bound.",
            "witness_maps": "Dropping the intermediate z projects(x,z,y) to(x,y). Reconstruction z=x^7 is efficient for the exact relation Y=X^49; its degree violates the fixed profile. Endpoint-language checks alone do not justify other witness maps.",
            "not_covered": "Full Poseidon rounds with affine mixing, other witness representations, extra variables, disjunctions, lookups, or equations coupling other state coordinates.",
        },
        "counterexamples": {
            "incorrect_fusion": {"x": 2, "shortcut_y": wrong_output, "valid_y": correct_output},
            "dropped_first_link": {"x": 0, "z": 1, "y": 1},
        },
        "runs": runs,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
