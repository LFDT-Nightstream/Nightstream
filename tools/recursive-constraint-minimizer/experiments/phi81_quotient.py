#!/usr/bin/env python3
"""Check the exact Goldilocks evaluation contract for the Phi81 quotient plan.

Run with `timeout --signal=KILL 300 python3 phi81_quotient.py` as required by
the root AGENTS.md test cap. cvc5 results are search evidence, not proof authority.
The coefficient certificate and counterexample replay use Python integers only.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import time


MODULUS = 18446744069414584321
RING_DEGREE = 54
NODE_COUNT = 2 * RING_DEGREE
PHI81 = [1] + [0] * 26 + [1] + [0] * 26 + [1]


def evaluate(coefficients, value):
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * value + coefficient) % MODULUS
    return result


def roots_polynomial(nodes):
    coefficients = [1]
    for node in nodes:
        extended = [0] * (len(coefficients) + 1)
        for index, coefficient in enumerate(coefficients):
            extended[index] = (extended[index] - node * coefficient) % MODULUS
            extended[index + 1] = (extended[index + 1] + coefficient) % MODULUS
        coefficients = extended
    return coefficients


def divide_monic(dividend, divisor):
    assert divisor[-1] == 1
    remainder = list(dividend)
    quotient = [0] * max(0, len(dividend) - len(divisor) + 1)
    for degree in range(len(dividend) - 1, len(divisor) - 2, -1):
        coefficient = remainder[degree]
        shift = degree - len(divisor) + 1
        quotient[shift] = coefficient
        for index, value in enumerate(divisor):
            remainder[shift + index] = (
                remainder[shift + index] - coefficient * value
            ) % MODULUS
    assert all(value == 0 for value in remainder[len(divisor) - 1 :])
    return quotient, remainder[: len(divisor) - 1]


def coefficient_certificate():
    """Construct Lagrange polynomials and check inverse * Vandermonde = I."""
    nodes = range(NODE_COUNT)
    product = roots_polynomial(nodes)
    lagrange = []
    for node in nodes:
        numerator, remainder = divide_monic(product, [(-node) % MODULUS, 1])
        assert remainder == [0]
        denominator = evaluate(numerator, node)
        assert denominator != 0
        inverse = pow(denominator, -1, MODULUS)
        lagrange.append([coefficient * inverse % MODULUS for coefficient in numerator])
    for degree in nodes:
        reconstructed = [
            sum(pow(node, degree, MODULUS) * lagrange[node][coefficient] for node in nodes)
            % MODULUS
            for coefficient in nodes
        ]
        assert reconstructed == [int(index == degree) for index in nodes]
    return {"dimension": NODE_COUNT, "inverse_times_vandermonde_is_identity": True}


def field_constant(value):
    return f"(as ff{value % MODULUS} F)"


def query(node_count, monic):
    lines = [
        "(set-logic QF_FF)",
        "(set-option :produce-models true)",
        f"(define-sort F () (_ FiniteField {MODULUS}))",
    ]
    lines.extend(f"(declare-const d{degree} F)" for degree in range(NODE_COUNT))
    for node in range(node_count):
        terms = []
        for degree in range(NODE_COUNT):
            coefficient = pow(node, degree, MODULUS)
            if coefficient:
                terms.append(
                    f"d{degree}" if coefficient == 1
                    else f"(ff.mul {field_constant(coefficient)} d{degree})"
                )
        expression = terms[0] if len(terms) == 1 else f"(ff.add {' '.join(terms)})"
        lines.append(f"(assert (= {expression} {field_constant(0)}))")
    if monic:
        lines.append(f"(assert (= d{NODE_COUNT - 1} {field_constant(1)}))")
    else:
        nonzero = " ".join(
            f"(not (= d{degree} {field_constant(0)}))" for degree in range(NODE_COUNT)
        )
        lines.append(f"(assert (or {nonzero}))")
    lines.append("(check-sat)")
    if monic:
        lines.append(f"(get-value ({' '.join(f'd{i}' for i in range(NODE_COUNT))}))")
    return "\n".join(lines) + "\n"


def solve(solver, smt2):
    start = time.monotonic()
    # The outer process enforces the root policy's 300-second invocation cap.
    run = subprocess.run(
        [solver, "--lang=smt2", "--ff-solver=gb"],
        input=smt2, text=True, capture_output=True, check=True,
    )
    status = run.stdout.splitlines()[0]
    assert status in ("sat", "unsat"), (status, run.stderr)
    return {
        "status": status,
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "stdout": run.stdout,
        "stderr": run.stderr,
    }


def replay_attack(coefficients):
    assert len(coefficients) == NODE_COUNT
    assert coefficients == roots_polynomial(range(NODE_COUNT - 1))
    quotient, remainder = divide_monic(coefficients, PHI81)
    h = [(-value) % MODULUS for value in remainder]
    q = [(-value) % MODULUS for value in quotient]
    assert len(h) == RING_DEGREE and len(q) == RING_DEGREE
    assert any(h)
    residual = []
    for degree in range(NODE_COUNT):
        phi_q = sum(
            PHI81[phi_degree] * q[degree - phi_degree]
            for phi_degree in (0, 27, 54)
            if 0 <= degree - phi_degree < len(q)
        )
        residual.append((-(h[degree] if degree < len(h) else 0) - phi_q) % MODULUS)
    assert residual == coefficients
    assert all(evaluate(residual, node) == 0 for node in range(NODE_COUNT - 1))
    omitted_value = evaluate(residual, NODE_COUNT - 1)
    assert omitted_value != 0
    return {
        "a": "zero polynomial", "b": "zero polynomial",
        "h_coefficients": h, "q_coefficients": q,
        "checked_nodes": NODE_COUNT - 1,
        "false_ring_product": any(h),
        "residual_at_omitted_node": omitted_value,
        "coefficient_replay_matches_model": True,
    }


def main():
    solver = shutil.which("cvc5")
    assert solver, "cvc5 with finite-field support must be on PATH"
    version = subprocess.check_output([solver, "--version"], text=True).splitlines()[0]
    certificate = coefficient_certificate()
    print("Coefficient certificate checked.", flush=True)
    complete = solve(solver, query(NODE_COUNT, monic=False))
    assert complete["status"] == "unsat"
    print("108-node query: UNSAT.", flush=True)
    incomplete = solve(solver, query(NODE_COUNT - 1, monic=True))
    assert incomplete["status"] == "sat"
    pairs = re.findall(r"\(d(\d+)\s+#f(\d+)m\d+\)", incomplete["stdout"])
    if not pairs:
        pairs = re.findall(r"\(d(\d+)\s+\(as ff(\d+)\s", incomplete["stdout"])
    values = {int(index): int(value) for index, value in pairs}
    assert set(values) == set(range(NODE_COUNT)), incomplete["stdout"]
    attack = replay_attack([values[index] for index in range(NODE_COUNT)])
    print("107-node query: SAT; false ring product replayed.", flush=True)
    report = {
        "scope": "Untrusted cvc5 search and exact integer replay; not Lean proof or package parity.",
        "solver": version, "field_modulus": MODULUS,
        "ring_degree": RING_DEGREE, "quotient_coefficient_count": RING_DEGREE,
        "residual_degree_bound_exclusive": NODE_COUNT,
        "coefficient_certificate": certificate,
        "complete_nodes": complete, "omitted_node_control": incomplete,
        "attack_replay": attack,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
