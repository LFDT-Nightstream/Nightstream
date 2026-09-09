"""Compute the selected setup's distribution budget and SIS model estimates.

Run with a current `validate.sh ajtai-setup-v1-parity` JSON file as argv[1].
This is a CPython evaluation of the Euclidean formulas in lattice-estimator
53da5982597709ba0fdf94ea37a84d822310fd84, not a Sage or full-estimator run.
It does not prove MSIS hardness or the public-seed setup assumption.
"""

import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path


REVISION = "53da5982597709ba0fdf94ea37a84d822310fd84"
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "formal/nightstream-fprime/NightstreamFPrime/Spec"
AUTHORITY = ROOT / "formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1SetupAuthority.lean"


def nat_definition(text, name):
    return int(re.search(r"^def " + name + r" : Nat := (\d+)$", text, re.M)[1])


def chen_log_delta(beta):
    """Chen's beta>40 expression, evaluated in the log domain."""
    return (math.log(beta / (2 * math.pi * math.e)) + math.log(math.pi * beta) / beta) / (2 * (beta - 1))


def block_size(log_delta, dimension):
    # This path is deliberately restricted to the beta>40 branch used here.
    if log_delta >= math.log(1.01295):
        raise ValueError("The selected inputs do not use the reviewed Chen branch")
    for beta in range(41, dimension + 1):
        if chen_log_delta(beta) <= log_delta:
            return beta
    raise ValueError("No feasible BKZ block size at this lattice dimension")


def matzov_bits(beta, dimension, slope, intercept):
    # Kyber.__call__, inherited by MATZOV; B=None gives LLL cost d^3.
    free = max(beta * math.log(4 / 3) / math.log(beta / (2 * math.pi * math.e)), 0)
    overhead = 1 / (1 - 2 ** (-slope))
    cost = dimension**3 + overhead**2 * max(dimension - beta, 1) * 2 ** (slope * (beta - free) + intercept)
    return math.log2(cost)


def sis_models(q, n, m, infinity_bound):
    length = infinity_bound * math.sqrt(m)
    if length >= q:
        raise ValueError("The reviewed Euclidean SIS branch requires length < q")
    dimension = min(math.floor(2 * n * math.log2(q) / math.log2(length)), m)
    log_delta = (math.log(length) - n * math.log(q) / dimension) / (dimension - 1)
    beta = block_size(log_delta, dimension)
    return {
        "n": n, "m": m, "q": q, "infinity_bound": infinity_bound,
        "euclidean_bound": length, "lattice_dimension": dimension,
        "beta": beta, "log_delta": log_delta,
        "chen_bracket": [chen_log_delta(beta - 1), chen_log_delta(beta)],
        "log2_cost": {
            "MATZOV_classical_gates": matzov_bits(beta, dimension, 0.29613500308205365, 20.387885985467914),
            "MATZOV_quantum_depth_times_width": matzov_bits(beta, dimension, 0.2663676536352464, 25.299541499216627),
            "ADPS16_classical_Core_SVP": 0.292 * beta,
            "ADPS16_quantum_Core_SVP": 0.265 * beta,
        },
    }


def main():
    vectors = json.loads(Path(sys.argv[1]).read_text())
    schema, setup_id, _, _, seed, _, descriptor = vectors
    if schema != 3 or descriptor != [len(setup_id)] + setup_id + descriptor[38:40] + [len(seed)] + seed:
        raise ValueError("Unexpected setup parity framing")
    if bytes(setup_id).decode() != "nightstream-ajtai-chacha20-wide256-v1" or len(seed) != 32:
        raise ValueError("Unexpected setup ID or seed length")
    q = nat_definition((SPEC / "Algebra.lean").read_text(), "goldilocksModulus")
    degree = nat_definition((SPEC / "Algebra.lean").read_text(), "ringDegree")
    rows, columns = descriptor[38:40]
    authority = AUTHORITY.read_text()
    if [rows, columns] != [nat_definition(authority, "verifierRows"), nat_definition(authority, "messageColumns")]:
        raise ValueError("Current Lean vectors and selected setup dimensions differ")
    profile = (SPEC / "Profile.lean").read_text().split("def productionGlobalParams", 1)[1].split("structure ProductionProfile", 1)[0]
    values = {name: int(re.search(r"^  " + name + r" := (\d+)$", profile, re.M)[1]) for name in ["b", "k", "expansionT"]}
    big_b = values["b"] ** values["k"]
    samples = 2**256
    remainder = samples % q
    coefficient_count = rows * columns * degree
    single_exact = Fraction(remainder * (q - remainder), q * samples)
    single_bound = Fraction(remainder, samples)
    # Published lattice-estimator delta examples check the inversion branch.
    examples = [(1.0121, 50), (1.0093, 100), (1.0024, 808)]
    for delta, expected in examples:
        assert block_size(math.log(delta), expected) == expected
    print(json.dumps({
        "source": "current Lean setup vectors and working-tree profile",
        "setup_descriptor": descriptor,
        "profile": {"b": values["b"], "k_rho": values["k"], "B": big_b, "T": values["expansionT"], "d": degree, "kappa": rows, "ring_columns": columns},
        "distribution": {
            "premise": "Independent uniform 256-bit inputs; this is not a ChaCha20 security assertion",
            "samples": samples, "remainder": remainder, "coefficient_count": coefficient_count,
            "single_exact_tv": str(single_exact), "single_proved_bound": str(single_bound),
            "whole_key_exact_hybrid_bound": str(coefficient_count * single_exact),
            "whole_key_proved_hybrid_bound": str(coefficient_count * single_bound),
            "proved_bound_negative_log2": -math.log2(coefficient_count * single_bound),
        },
        "estimator_source_revision": REVISION,
        "execution": "CPython Euclidean formula evaluation; not the full Sage estimator",
        "upstream_delta_examples": examples,
        "selected": sis_models(q, rows * degree, columns * degree, 8 * values["expansionT"] * big_b),
        "paper_B6_reference_only": sis_models(q, 18 * 54, 19884107 * 54, 8 * 216 * 2**14),
        "limits": ["Heuristic generic lattice attack model", "No proof for structured Module-SIS or this fixed public seed", "Cost units differ between models; these are not interchangeable security-bit guarantees"],
    }, indent=2))


if __name__ == "__main__":
    main()
