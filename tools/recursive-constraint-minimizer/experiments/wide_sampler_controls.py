#!/usr/bin/env python3
"""Review controls for sampler commit 919cd12b4; Lean remains proof authority.

Run under the project test cap: timeout --signal=KILL 300 python3 <this file>.
"""

from fractions import Fraction
import json
from math import gcd, log2, prod
from pathlib import Path
import re
import shutil
import subprocess


REVISION = "919cd12b4"
P = 18446744069414584321


def solve(solver, logic, body, expected):
    query = f"(set-logic {logic})\n" + "\n".join(body) + "\n(check-sat)\n"
    command = [solver, "--lang=smt2"]
    if logic == "QF_FF":
        command.append("--ff-solver=gb")
    result = subprocess.run(command, input=query, text=True, capture_output=True, check=True)
    actual = result.stdout.strip()
    if actual != expected:
        raise AssertionError(f"expected {expected}, got {actual}: {result.stderr}")
    return actual


def main():
    solver = shutil.which("cvc5")
    if solver is None:
        raise RuntimeError("cvc5 is required")
    source = subprocess.run(
        ["git", "show", f"{REVISION}:formal/nightstream-fprime/NightstreamFPrime/"
         "Gadgets/Sampling/WideReduction.lean"],
        text=True, capture_output=True, check=True,
    ).stdout
    moduli = [int(x) for x in re.search(r"!\[([^]]+)\]", source, re.S)[1].split(",")]

    def parameter(name):
        return int(re.search(rf"def {name} : Nat := (\d+)", source)[1])

    q_bits = parameter("quotientBitCount")
    digits = parameter("digitCount")
    d_bits = parameter("digitBitCount")
    k_bits = parameter("checkBitCount")
    bias = q_bits + digits * d_bits
    n, draws = 5 ** digits, P ** 4
    remainder = draws % n
    distance = Fraction(remainder * (n - remainder), n * draws)
    assert all(gcd(a, b) == 1 for i, a in enumerate(moduli) for b in moduli[i + 1:])
    assert max(draws, n * 2 ** q_bits) <= prod(moduli)
    assert (draws - 1) // n < 2 ** q_bits
    assert distance < Fraction(1, 2 ** 132)

    checks = {}
    field = [f"(define-sort F () (_ FiniteField {P}))"]
    field += [f"(declare-const b{i} F)" for i in range(3)]
    field += [f"(assert (= (ff.mul b{i} (ff.add b{i} (as ff{P-1} F))) (as ff0 F)))"
              for i in range(3)]
    field += ["(define-fun digit () F (ff.add b0 (ff.mul (as ff2 F) b1) "
              "(ff.mul (as ff4 F) b2)))",
              "(assert (or (= digit (as ff5 F)) (= digit (as ff6 F)) (= digit (as ff7 F))))"]
    checks["digit_range_rejects_5_to_7"] = solve(solver, "QF_FF", field + [
        "(assert (= (ff.mul b2 (ff.add b0 b1)) (as ff0 F)))"], "unsat")
    checks["missing_digit_range_accepts_5"] = solve(solver, "QF_FF", field + [
        "(assert (= b0 (as ff1 F)))", "(assert (= b1 (as ff0 F)))",
        "(assert (= b2 (as ff1 F)))"], "sat")

    # The largest modulus bounds all six rows. These are integer range
    # consequences of the bit proofs, with an explicit field-wrap variable.
    m = max(moduli)
    lift = [f"(declare-const {name} Int)" for name in ("a", "b", "k", "wrap")]
    lift += [f"(assert (and (<= 0 a) (<= a {256*(m-1)})))",
             f"(assert (and (<= 0 b) (<= b {bias*(m-1)})))",
             "(assert (<= 0 k))",
             f"(define-fun left () Int (+ a {bias*m}))",
             f"(define-fun right () Int (+ b (* {m} k)))",
             f"(assert (= left (+ right (* {P} wrap))))",
             "(assert (distinct left right))"]
    checks["bounded_check_row_cannot_wrap"] = solve(
        solver, "QF_LIA", lift + [f"(assert (< k {2**k_bits}))"], "unsat")
    checks["missing_check_range_allows_wrap"] = solve(solver, "QF_LIA", lift + [
        "(assert (= a 0))", "(assert (= b 0))", f"(assert (= k {bias+P}))"], "sat")

    # Check the range consequence after the separately proved CRT lemma.
    crt = [f"(declare-const {name} Int)" for name in ("x", "y", "multiple")]
    crt += [f"(assert (and (<= 0 x) (< x {draws})))",
            f"(assert (and (<= 0 y) (< y {n*2**q_bits})))",
            f"(assert (= (- x y) (* {prod(moduli)} multiple)))",
            "(assert (distinct x y))"]
    checks["crt_range_forces_equality"] = solve(solver, "QF_LIA", crt, "unsat")

    # Actual bounded false witness when the sixth check row is deleted.
    forged = prod(moduli[:-1])
    q, r = divmod(forged, n)
    weights = [2**b*n for b in range(q_bits)] + [
        2**b*5**d for d in range(digits) for b in range(d_bits)]
    bits = [q >> b & 1 for b in range(q_bits)] + [
        (r // 5**d % 5) >> b & 1 for d in range(digits) for b in range(d_bits)]
    assert q < 2**q_bits and r < n and r != 0
    rows, quotients, residuals = [], [], []
    for i, modulus in enumerate(moduli):
        reduced = sum(a % modulus * bit for a, bit in zip(weights, bits))
        k, residual = divmod(bias * modulus - reduced, modulus)
        assert 0 <= k < 2**k_bits
        quotients.append(k)
        residuals.append(residual)
        rows.append(f"(assert (= {bias*modulus} (+ {reduced} (* {modulus} {k}))))")
    assert residuals[:-1] == [0] * (len(moduli) - 1) and residuals[-1] != 0
    checks["missing_sixth_check_accepts_false_scalar"] = solve(solver, "QF_LIA", rows[:-1], "sat")
    checks["sixth_check_rejects_same_false_scalar"] = solve(solver, "QF_LIA", rows, "unsat")

    result = {
        "reviewed_commit": REVISION,
        "moduli": moduli,
        "exact_bias": {"numerator": distance.numerator, "denominator": distance.denominator},
        "bias_log2": log2(distance),
        "seventeen_draws_bound_log2": log2(17 * distance),
        "checks": checks,
        "omitted_sixth_row_witness": {"X": 0, "Q": q, "R": r, "K": quotients,
                                      "check_residuals": residuals},
        "scope": "Local range checks and replayed omission attacks; not a production or transcript proof.",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
