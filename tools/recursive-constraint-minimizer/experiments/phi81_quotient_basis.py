#!/usr/bin/env python3
"""Exact research check for storing Q(0)..Q(53) in the existing 54 slots.

Run under `timeout --signal=KILL 300`, per the root AGENTS.md test cap.
This reads the current formula artifact but never changes production files.
The finite matrices are checked exhaustively, not on random assignments.
"""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from phi81_quotient import (
    MODULUS, NODE_COUNT, PHI81, RING_DEGREE, divide_monic, evaluate,
    field_constant, roots_polynomial,
)


ROOT = Path(__file__).resolve().parents[3]
LEAN = ROOT / "formal/nightstream-fprime/NightstreamFPrime"
DOCS = [
    "https://doc.sagemath.org/html/en/reference/polynomial_rings/sage/rings/polynomial/polynomial_ring.html#sage.rings.polynomial.polynomial_ring.PolynomialRing_field.lagrange_polynomial",
    "https://doc.sagemath.org/html/en/reference/finite_rings/sage/rings/finite_rings/finite_field_constructor.html",
    "https://cvc5.github.io/docs/cvc5-1.3.4/theories/finite_field.html",
]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right)) % MODULUS


def multiply(left, right):
    columns = list(zip(*right))
    return [[dot(row, column) for column in columns] for row in left]


def linear_form(weights, prefix):
    terms = [
        f"{prefix}{index}" if value == 1 else
        f"(ff.mul {field_constant(value)} {prefix}{index})"
        for index, value in enumerate(weights) if value
    ]
    return terms[0] if len(terms) == 1 else f"(ff.add {' '.join(terms)})"


def lagrange_matrices():
    product = roots_polynomial(range(RING_DEGREE))
    basis = []
    for node in range(RING_DEGREE):
        numerator, remainder = divide_monic(product, [(-node) % MODULUS, 1])
        assert remainder == [0]
        denominator = evaluate(numerator, node)
        assert denominator != 0
        basis.append([x * pow(denominator, -1, MODULUS) % MODULUS for x in numerator])
    # V sends coefficients to values. I sends values to coefficients.
    vandermonde = [[pow(node, power, MODULUS) for power in range(RING_DEGREE)]
                   for node in range(RING_DEGREE)]
    inverse = [list(column) for column in zip(*basis)]
    evaluation = [[pow(node, power, MODULUS) for power in range(RING_DEGREE)]
                  for node in range(NODE_COUNT)]
    weights = [[evaluate(polynomial, node) for polynomial in basis]
               for node in range(NODE_COUNT)]
    identity = [[int(i == j) for j in range(RING_DEGREE)] for i in range(RING_DEGREE)]
    assert multiply(vandermonde, inverse) == identity
    assert multiply(inverse, vandermonde) == identity
    assert weights[:RING_DEGREE] == identity
    assert multiply(weights, vandermonde) == evaluation
    assert multiply(evaluation, inverse) == weights
    return vandermonde, inverse, evaluation, weights


def source_cost(evaluation, weights):
    source = (LEAN / "Export/Stage1/PiRLCProductMatrixProgram.lean").read_text()
    families = [tuple(map(int, fields)) for fields in re.findall(
        r"\{ sourceCount := (\d+), blockCount := (\d+), cellCount := (\d+) \}", source)]
    assert families == [(17, 22, 1), (17, 5, 1), (17, 1, 2), (17, 14, 2)]
    width_source = (LEAN / "Layout/BalancedTernary.lean").read_text()
    field_width = int(re.search(r"def width : Nat := (\d+)", width_source).group(1))
    assert field_width == 41
    library_path = ROOT / "crates/nightstream-fprime/artifacts/shared-formulas-v1.json"
    library = json.loads(library_path.read_text())
    assert library["profile"] == [MODULUS, 2, 16, 65536, 54, 28, 14, 13]
    component = next(c for c in library["components"] if c["id"] == "phi81-product-v1")
    assert component["input_count"] == 271 and len(component["variants"]) == 1
    rows = component["variants"][0]["rows"]
    assert len(rows) == NODE_COUNT
    quotient_start, output_port = 109, 4
    old_counts, new_counts, raw_counts = [], [], []
    for node, row in enumerate(rows):
        actual = [0] * RING_DEGREE
        raw = 0
        for port, form in enumerate(row):
            for column, coefficient in form:
                if quotient_start <= column < quotient_start + RING_DEGREE:
                    assert port == output_port
                    actual[column - quotient_start] = (
                        actual[column - quotient_start] + coefficient) % MODULUS
                    raw += 1
        phi = evaluate(PHI81, node)
        assert phi != 0
        expected = [phi * coefficient % MODULUS for coefficient in evaluation[node]]
        assert actual == expected
        candidate = [phi * coefficient % MODULUS for coefficient in weights[node]]
        old_counts.append(sum(value != 0 for value in actual))
        new_counts.append(sum(value != 0 for value in candidate))
        raw_counts.append(raw)
        for coefficient in actual + candidate:
            if coefficient:
                assert all(coefficient * pow(3, digit, MODULUS) % MODULUS
                           for digit in range(field_width))
    # Match descriptor.invocationAtLane: family/source/block/lane/cell order.
    rings, slots, offset = 0, set(), 0
    for source_count, block_count, cell_count in families:
        for source_index in range(source_count):
            for block in range(block_count):
                for cell in range(cell_count):
                    indices = [offset + source_index * block_count * RING_DEGREE * cell_count
                               + (block * RING_DEGREE + lane) * cell_count + cell
                               for lane in range(RING_DEGREE)]
                    assert len(set(indices)) == RING_DEGREE and not slots.intersection(indices)
                    slots.update(indices)
                    rings += 1
        offset += source_count * block_count * RING_DEGREE * cell_count
    assert slots == set(range(offset)) and rings == 969 and offset == 52326
    old, new = sum(old_counts), sum(new_counts)
    saving = (old - new) * field_width * rings
    assert (old, new, old - new, saving) == (5779, 2970, 2809, 111598761)
    metrics = json.loads(Path(__file__).with_name("checkpoint-metrics.json").read_text())
    checkpoint = metrics["checkpoint"]
    predicted = checkpoint["matrix_nonzero_total"] - saving
    return {
        "source_families": families,
        "ring_products": rings,
        "field_slot_coordinates": field_width,
        "quotient_slots": offset,
        "lane_cell_slot_map_is_disjoint_bijection": True,
        "formula_artifact_read_only": str(library_path.relative_to(ROOT)),
        "all_108_current_quotient_forms_match_source_equation": True,
        "quotient_output_port": output_port,
        "all_108_modulus_values_nonzero": True,
        "all_nonzero_weights_expand_to_41_nonzero_trit_weights": True,
        "raw_current_template_quotient_terms_per_ring": sum(raw_counts),
        "current_normalized_weights_per_node": old_counts,
        "candidate_normalized_weights_per_node": new_counts,
        "current_normalized_quotient_weights_per_ring": old,
        "candidate_normalized_quotient_weights_per_ring": new,
        "saved_normalized_weights_per_ring": old - new,
        "saved_matrix_nonzeros_per_ring": (old - new) * field_width,
        "saved_matrix_nonzeros": saving,
        "checkpoint_matrix_nonzeros": checkpoint["matrix_nonzero_total"],
        "predicted_matrix_nonzeros": predicted,
        "original_baseline_matrix_nonzeros": metrics["baseline"]["matrix_nonzero_total"],
        "predicted_excess_over_original_baseline": predicted - metrics["baseline"]["matrix_nonzero_total"],
        "committed_coordinates_before_and_after": checkpoint["committed_coordinates"],
        "logical_rows_before_and_after": checkpoint["logical_rows"],
        "product_rows_before_and_after": rings * NODE_COUNT,
        "checkpoint_total_is_saved_conformance_evidence_not_remeasured": True,
        "runtime_or_memory_saving_measured": False,
    }


def rank_query(node_count, monic):
    lines = ["(set-logic QF_FF)", "(set-option :produce-models true)",
             f"(define-sort F () (_ FiniteField {MODULUS}))"]
    lines += [f"(declare-const q{i} F)" for i in range(RING_DEGREE)]
    for node in range(node_count):
        form = linear_form([pow(node, i, MODULUS) for i in range(RING_DEGREE)], "q")
        lines.append(f"(assert (= {form} {field_constant(0)}))")
    if monic:
        lines.append(f"(assert (= q53 {field_constant(1)}))")
    else:
        nonzero = " ".join(f"(not (= q{i} {field_constant(0)}))" for i in range(RING_DEGREE))
        lines.append(f"(assert (or {nonzero}))")
    lines.append("(check-sat)")
    if monic:
        lines.append(f"(get-value ({' '.join(f'q{i}' for i in range(RING_DEGREE))}))")
    return "\n".join(lines) + "\n"


def solve(solver, query):
    start = time.monotonic()
    result = subprocess.run([solver, "--lang=smt2", "--ff-solver=gb"], input=query,
                            text=True, capture_output=True, check=True, timeout=300)
    status = result.stdout.splitlines()[0]
    assert status in ("sat", "unsat"), result.stderr
    return {"status": status, "elapsed_seconds": round(time.monotonic() - start, 6),
            "query": query, "stdout": result.stdout, "stderr": result.stderr}


def main():
    started = time.monotonic()
    vandermonde, inverse, evaluation, weights = lagrange_matrices()
    cost = source_cost(evaluation, weights)
    # A=X^53, B=X gives Q=1 and H=-1-X^27. The final VALUE slot is 1, not 0.
    product = [0] * 54 + [1]
    quotient, remainder = divide_monic(product, PHI81)
    coefficients = quotient + [0] * (RING_DEGREE - len(quotient))
    values = [dot(row, coefficients) for row in vandermonde]
    recovered = [dot(row, values) for row in inverse]
    assert recovered == coefficients and coefficients[53] == 0 and values[53] == 1
    for node in range(NODE_COUNT):
        assert pow(node, 54, MODULUS) == (
            evaluate(remainder, node) + evaluate(PHI81, node) * dot(weights[node], values)
        ) % MODULUS
    exact_seconds = round(time.monotonic() - started, 6)
    print(f"Exact matrices, formula replay and NNZ counts passed in {exact_seconds}s.", flush=True)
    solver = shutil.which("cvc5")
    assert solver, "The installed cvc5 CLI is required for the two rank controls."
    complete = solve(solver, rank_query(RING_DEGREE, False))
    assert complete["status"] == "unsat"
    print("54-value basis injectivity: UNSAT counterexample query.", flush=True)
    incomplete = solve(solver, rank_query(RING_DEGREE - 1, True))
    assert incomplete["status"] == "sat"
    pairs = re.findall(r"\(q(\d+)\s+#f(\d+)m\d+\)", incomplete["stdout"])
    model = {int(index): int(value) for index, value in pairs}
    assert set(model) == set(range(RING_DEGREE)), incomplete["stdout"]
    attack = [model[index] for index in range(RING_DEGREE)]
    assert attack == roots_polynomial(range(RING_DEGREE - 1))
    assert all(evaluate(attack, node) == 0 for node in range(RING_DEGREE - 1))
    assert evaluate(attack, RING_DEGREE - 1) != 0
    report = {
        "scope": "Research exact arithmetic and cvc5 controls; not a Lean proof, selected package change, or runtime benchmark.",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "field_modulus": MODULUS,
        "quotient_slots": RING_DEGREE,
        "equation_nodes": list(range(NODE_COUNT)),
        "basis_nodes": list(range(RING_DEGREE)),
        "tool_availability": {
            "sage_executable": shutil.which("sage"),
            "sage_python": importlib.util.find_spec("sage") is not None,
            "sympy_python": importlib.util.find_spec("sympy") is not None,
            "galois_python": importlib.util.find_spec("galois") is not None,
            "cvc5_python": importlib.util.find_spec("cvc5") is not None,
            "cvc5_cli": subprocess.check_output([solver, "--version"], text=True).splitlines()[0],
            "exact_engine": "Python standard-library integers modulo the Goldilocks prime",
        },
        "exact_checks": {
            "forward_map": "v_j = sum_i j^i q_i for j=0..53",
            "inverse_map": "q_i = sum_j [X^i] product_(m != j)(X-m)/(j-m) v_j",
            "V_times_inverse_equals_identity": True,
            "inverse_times_V_equals_identity": True,
            "all_108_Lagrange_rows_times_V_equal_monomial_evaluation_rows": True,
            "all_108_monomial_rows_times_inverse_equal_Lagrange_rows": True,
            "checks_cover_all_vectors_by_exact_matrix_coefficient_equality": True,
            "basis_padded_coefficient_is_not_last_value": {"q53": coefficients[53], "Q_at_53": values[53]},
            "honest_X53_times_X_product_replays_all_108_equations": True,
            "elapsed_seconds": exact_seconds,
        },
        "cost_axes": cost,
        "rank_control_54_values": complete,
        "rank_control_53_values": incomplete,
        "rank_control_replay": {"matches_roots_polynomial": True, "value_at_missing_node_53": evaluate(attack, 53)},
        "primary_docs": DOCS,
        "source_queries": [
            "command -v sage; command -v cvc5; python3 importlib.util.find_spec for sage/cvc5/sympy/galois",
            "PiRLCProductMatrixProgram.lean: four family dimensions and families_ringCount",
            "Phi81ProductPlan.lean: evaluateForm, outputForm, productRow",
            "SharedFormulas.lean: phi81Interface quotient inputs 109..162",
            "ProductSumRow.lean: output at meaningful port 4",
            "BalancedTernary.lean: width; RetainedSlot.lean: recomposeForms",
            "Layout/MatrixProgram/Phi81Product.lean: invocationAtLane and quotientState?",
            "Rust package/matrix_program/{phi81,template,form}.rs: formula substitution, retained radix 3, zero scaling",
            "Read-only shared-formulas-v1.json: every quotient coefficient in all 108 rows",
        ],
        "non_lean_timeout_seconds": 300,
        "timeout_authority": "Root AGENTS.md non-Lean test cap",
        "total_elapsed_seconds": round(time.monotonic() - started, 6),
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {cost['saved_matrix_nonzeros']:,} predicted NNZ; result: {output}", flush=True)


if __name__ == "__main__":
    main()
