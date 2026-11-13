#!/usr/bin/env python3
"""Re-validate params.json and vectors.json without derive.py.

Run `python3 scripts/poseidon2/check.py`. Each check prints one PASS or
FAIL line. The exit code is nonzero when a check fails or when a negative
control is accepted. The algorithms differ from derive.py on purpose:
Lucas primality and trial division instead of Miller-Rabin, the
characteristic polynomial by interpolation of det(xI - A) instead of
Faddeev-LeVerrier, Berlekamp's rank test instead of Rabin's test, a round
number search without the reference script's loop quirk, and the p3-style
light linear layers instead of dense matrices.
"""

import glob
import hashlib
import json
import math
import re
import sys
from itertools import combinations, islice
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDILOCKS = 2**64 - 2**32 + 1
MERSENNE31 = 2**31 - 1
PLONKY3_ROUNDS_128 = {
    MERSENNE31: {
        (16, 3): (8, 20), (16, 5): (8, 14), (16, 7): (8, 13), (16, 9): (8, 13), (16, 11): (8, 13),
        (24, 3): (8, 23), (24, 5): (8, 22), (24, 7): (8, 21), (24, 9): (8, 21), (24, 11): (8, 21),
        (32, 3): (8, 31), (32, 5): (8, 30), (32, 7): (8, 30), (32, 9): (8, 30), (32, 11): (8, 30),
    },
    GOLDILOCKS: {
        (8, 3): (8, 41), (8, 5): (8, 27), (8, 7): (8, 22), (8, 9): (8, 19), (8, 11): (8, 17),
        (12, 3): (8, 42), (12, 5): (8, 27), (12, 7): (8, 22), (12, 9): (8, 20), (12, 11): (8, 18),
        (16, 3): (8, 42), (16, 5): (8, 27), (16, 7): (8, 22), (16, 9): (8, 20), (16, 11): (8, 18),
    },
}
PLONKY3_DIAGONALS = (
    (GOLDILOCKS, "p3 MATRIX_DIAG_12_GOLDILOCKS",
     ["-2", "1", "2", "1/2", "3", "4", "-1/2", "-3", "-4", "1/2^2", "-1/2^2", "1/2^3"]),
    (GOLDILOCKS, "p3 MATRIX_DIAG_16_GOLDILOCKS",
     ["-2", "1", "2", "1/2", "3", "4", "-1/2", "-3", "-4", "1/2^3", "1/2^4", "1/2^5",
      "-1/2^3", "-1/2^4", "-1/2^5", "1/2^32"]),
    (MERSENNE31, "p3 internal.rs Mersenne31 example",
     ["-2", "1", "2", "4", "8", "16", "32", "64", "128", "256", "1024", "4096", "8192", "16384",
      "32768", "65536"]),
)
NON_MDS_M4 = [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]

failures = 0


def report(ok, text):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'} {text}")


def negative_control(rejected, text):
    report(rejected, f"negative control: {text} -> check returns FAIL as required")


def trial_division_prime(n):
    return n > 1 and all(n % k for k in range(2, math.isqrt(n) + 1))


def lucas_prime(p, factors):
    """p is prime if some a has order p - 1 modulo p; needs every factor of p - 1 prime."""
    primes = set(factors)
    return any(
        pow(a, p - 1, p) == 1 and all(pow(a, (p - 1) // r, p) != 1 for r in primes)
        for a in range(2, 200)
    )


def ceil_log(d, x):
    return next(k for k in range(x.bit_length() + 1) if d**k >= x)


def secure(p, t, d, rf, rp, m=128):
    """The reference inequalities rewritten in integers: rf >= ceil(log_d(X) - rp)
    holds exactly when d^(rf + rp) >= X, and likewise for the others."""
    statistical = 6 if m <= (p.bit_length() - 1 - (d - 1) // 2) * (t + 1) else 10
    interpolation = 1 + ceil_log(d, 2 ** min(m, p.bit_length())) + ceil_log(d, t) - rp
    k1, k2, k3 = rf + rp, rf + rp - t + 1, rf * (t - 1) - t + 2 + rp
    groebner1 = k1 >= 0 and (d**k1 >= 2**m or d**k1 >= p)
    groebner2 = k2 >= 0 and (d ** (k2 * (t + 1)) >= 2**m or d ** (2 * k2) >= p)
    groebner3 = k3 >= 0 and d ** (2 * k3) >= 2**m
    if not (rf >= max(statistical, interpolation) and groebner1 and groebner2 and groebner3):
        return False
    r = t // 3
    n, k = (rf - 1) * t + rp + r + r * (rf // 2) + rp + d, r * (rf // 2) + rp + d
    log2_binom = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)
    if abs(2 * log2_binom - (m - 1)) > 1:
        return 2 * log2_binom > m - 1
    return math.comb(n, k) ** 2 > 2 ** (m - 1)


def round_numbers(p, t, d):
    return min(
        (t * (rf + 2) + -(-43 * rp // 40), rf + 2, -(-43 * rp // 40))
        for rp in range(1, 500)
        for rf in range(4, 100, 2)
        if secure(p, t, d, rf, rp)
    )[1:]


def laplace_det(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1) ** j * a[0][j] * laplace_det([row[:j] + row[j + 1:] for row in a[1:]])
               for j in range(len(a)))


def is_mds(m4, p):
    return all(
        laplace_det([[m4[i][j] for j in cols] for i in rows]) % p != 0
        for k in range(1, 5)
        for rows in combinations(range(4), k)
        for cols in combinations(range(4), k)
    )


def eliminate(a, p):
    """Row reduction modulo p; returns (rank, determinant)."""
    a = [[v % p for v in row] for row in a]
    rank, det, n_cols = 0, 1, len(a[0])
    for col in range(n_cols):
        pivot = next((r for r in range(rank, len(a)) if a[r][col]), None)
        if pivot is None:
            det = 0
            continue
        if pivot != rank:
            a[rank], a[pivot] = a[pivot], a[rank]
            det = -det
        det = det * a[rank][col] % p
        inv = pow(a[rank][col], p - 2, p)
        for r in range(rank + 1, len(a)):
            if a[r][col]:
                factor = a[r][col] * inv % p
                a[r] = [(x - factor * y) % p for x, y in zip(a[r], a[rank])]
        rank += 1
    return rank, det % p


def matmul(a, b, p):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) % p for j in range(len(b[0]))]
            for i in range(len(a))]


def charpoly(a, p):
    """Interpolates det(xI - A) at x = 0..t; coefficients from x^0 to x^t."""
    t = len(a)
    xs = list(range(t + 1))
    ys = [eliminate([[(x * (i == j) - a[i][j]) for j in range(t)] for i in range(t)], p)[1] for x in xs]
    coeffs = [0] * (t + 1)
    for k, (xk, yk) in enumerate(zip(xs, ys)):
        basis = [1]
        denominator = 1
        for j, xj in enumerate(xs):
            if j != k:
                basis = [((basis[i - 1] if i else 0) - xj * (basis[i] if i < len(basis) else 0)) % p
                         for i in range(len(basis) + 1)]
                denominator = denominator * (xk - xj) % p
        scale = yk * pow(denominator, p - 2, p) % p
        coeffs = [(c + scale * b) % p for c, b in zip(coeffs, basis)]
    assert coeffs[-1] == 1, "characteristic polynomial is not monic"
    return coeffs


def poly_rem(a, f, p):
    a = [v % p for v in a]
    inv = pow(f[-1], p - 2, p)
    while len(a) >= len(f):
        c = a[-1] * inv % p
        shift = len(a) - len(f)
        for i, v in enumerate(f):
            a[shift + i] = (a[shift + i] - c * v) % p
        a.pop()
        while a and a[-1] == 0:
            a.pop()
    return a


def poly_mulmod(a, b, f, p):
    product = [0] * (len(a) + len(b) - 1) if a and b else []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] += x * y
    return poly_rem(product, f, p)


def poly_gcd_degree(a, b, p):
    while b:
        a, b = b, poly_rem(a, b, p)
    return len(a) - 1


def is_irreducible(f, p):
    """Berlekamp: a squarefree f has n - rank(Q - I) irreducible factors."""
    n = len(f) - 1
    derivative = [i * c % p for i, c in enumerate(f)][1:]
    if poly_gcd_degree(f, derivative, p) != 0:
        return False
    xp, base, e = [1], [0, 1], p
    while e:
        if e & 1:
            xp = poly_mulmod(xp, base, f, p)
        base = poly_mulmod(base, base, f, p)
        e >>= 1
    rows, row = [], [1]
    for i in range(n):
        dense = row + [0] * (n - len(row))
        dense[i] -= 1
        rows.append(dense)
        row = poly_mulmod(row, xp, f, p)
    return eliminate(rows, p)[0] == n - 1


def internal_matrix_failure(diag, p):
    t = len(diag)
    if any(v % p in (0, p - 1) for v in diag):
        return "an entry of D or of M_I is zero"
    m = [[(1 + diag[i]) * (i == j) + (i != j) for j in range(t)] for i in range(t)]
    if eliminate(m, p)[1] == 0:
        return "M_I is singular"
    power = m
    for i in range(1, 2 * t + 1):
        if not is_irreducible(charpoly(power, p), p):
            return f"charpoly of M_I^{i} is reducible"
        power = matmul(power, m, p)
    return None


def field_elements(label, count, q):
    data = hashlib.shake_128(label.encode("ascii")).digest(8 * (count + 64))
    words = (int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8))
    values = list(islice((w for w in words if w < q), count))
    assert len(values) == count
    return values


def light_external(x, m4, p):
    y = []
    for c in range(0, len(x), 4):
        y += [sum(m4[i][j] * x[c + j] for j in range(4)) for i in range(4)]
    sums = [sum(y[k::4]) for k in range(4)]
    return [(v + sums[i % 4]) % p for i, v in enumerate(y)]


def light_permute(state, m4, diag, initial, internal, terminal, d, p):
    x = light_external(state, m4, p)
    for row in initial:
        x = light_external([pow((v + c) % p, d, p) for v, c in zip(x, row)], m4, p)
    for c in internal:
        x[0] = pow((x[0] + c) % p, d, p)
        total = sum(x)
        x = [(diag[i] * v + total) % p for i, v in enumerate(x)]
    for row in terminal:
        x = light_external([pow((v + c) % p, d, p) for v, c in zip(x, row)], m4, p)
    return x


def parse_fraction(text, p):
    sign = -1 if text.startswith("-") else 1
    numerator, _, denominator = text.lstrip("-").partition("/")
    base, _, exponent = (denominator or "1").partition("^")
    return sign * int(numerator) * pow(int(base) ** int(exponent or 1), p - 2, p) % p


def plonky3_goldilocks_kat(m4):
    paths = glob.glob(str(Path.home() / ".cargo/registry/src/*/p3-goldilocks-0.5.3/src/poseidon2.rs"))
    if not paths:
        print("SKIP p3-goldilocks 0.5.3 source not found; width-12 Plonky3 known-answer test not run")
        return
    src = Path(paths[0]).read_text()

    def block(name):
        start = src.index("=", src.index(f"pub const {name}"))
        end = re.compile(r"^\]\)?;", re.M).search(src, start).start()
        body = re.sub(r"//[^\n]*", "", src[start + 1:end])
        return [int(v, 0) for v in re.findall(r"\b(0x[0-9a-fA-F]+|\d+)\b", body)]

    rows = lambda values: [values[i:i + 12] for i in range(0, len(values), 12)]
    test = src[src.index("fn test_default_goldilocks_poseidon2_width_12"):]
    expected = [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+", test[:test.index("assert_eq")])]
    got = light_permute(list(range(12)), m4, block("MATRIX_DIAG_12_GOLDILOCKS"),
                        rows(block("GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL")),
                        block("GOLDILOCKS_POSEIDON2_RC_12_INTERNAL"),
                        rows(block("GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL")), 7, GOLDILOCKS)
    report(len(expected) == 12 and got == expected,
           "light permutation with this M4 reproduces the p3-goldilocks 0.5.3 width-12 known answer")


def check_instance(inst, m4, q, vectors, suffix):
    t, d, rf, rp, label = inst["t"], inst["d"], inst["rf"], inst["rp"], inst["label"]
    tag = f"t={t}:"
    report(int(inst["q"]) == q, f"{tag} instance q matches the field")
    report(math.gcd(d, q - 1) == 1, f"{tag} gcd(d={d}, q - 1) = 1, so x^{d} is a permutation")
    report(round_numbers(q, t, d) == (rf, rp),
           f"{tag} (R_F, R_P) = ({rf}, {rp}) is the cheapest 128-bit choice with margin")
    report(label == f"Nightstream/Poseidon2/q=2^64-59/t={t}/d={d}/rf={rf}/rp={rp}/v1",
           f"{tag} label binds q, t, d, R_F, R_P")

    stream = field_elements(label, rf * t + rp, q)
    initial = [stream[i * t:(i + 1) * t] for i in range(rf // 2)]
    terminal = [stream[(rf // 2 + i) * t:(rf // 2 + i + 1) * t] for i in range(rf // 2)]
    internal = stream[rf * t:]
    stored = ([[int(v) for v in row] for row in inst["external_initial"]],
              [int(v) for v in inst["internal"]],
              [[int(v) for v in row] for row in inst["external_terminal"]])
    report(stored == (initial, internal, terminal),
           f"{tag} {rf * t + rp} round constants regenerate from SHAKE128(label)")
    tampered = json.loads(json.dumps(stored))
    tampered[1][0] = (tampered[1][0] + 1) % q
    negative_control(tuple(tampered) != (initial, internal, terminal),
                     f"{tag} one internal constant changed by +1")

    report(eliminate(m4_blocks(m4, t), q)[1] != 0, f"{tag} M_E = circ(2 M4, M4, ...) is invertible")

    diag = [int(v) for v in inst["internal_diagonal"]]
    signed = [int(v) for v in inst["internal_diagonal_signed"]]
    report([v % q for v in signed] == diag, f"{tag} internal diagonal residues equal the signed values {signed}")
    failure = internal_matrix_failure(diag, q)
    report(failure is None,
           f"{tag} M_I = J + diag(D): entries nonzero, invertible, charpoly(M_I^i) irreducible "
           f"for i = 1..{2 * t}" + (f" ({failure})" if failure else ""))

    pool = [1] + [s * v for v in range(2, 4 * t) for s in (1, -1)]
    pool = pool[:4 * t]
    candidates = (
        [pool[i] for i in head] + [pool[n - 1]]
        for n in range(t, len(pool) + 1)
        for head in combinations(range(n - 1), t - 1)
    )
    tried = inst["diagonal_candidates_tried"]
    first = list(islice(candidates, tried))
    earlier_rejected = all(internal_matrix_failure(c, q) is not None for c in first[:-1])
    report(first[-1] == signed and earlier_rejected,
           f"{tag} D is candidate {tried} of the stated rule; all {tried - 1} earlier candidates FAIL")
    all_ones = internal_matrix_failure([1] * t, q)
    negative_control(all_ones is not None, f"{tag} M_I check on the all-ones diagonal ({all_ones})")

    test_states = field_elements(label + suffix, 3 * t, q)
    expected_inputs = [list(range(t)), [0] * t] + [test_states[i * t:(i + 1) * t] for i in range(3)]
    cases = [v for v in vectors if v["t"] == t]
    inputs = [[int(x) for x in v["input"]] for v in cases]
    report(inputs == expected_inputs, f"{tag} {len(cases)} test vector inputs are [0..t-1], zero, 3 SHAKE states")
    outputs = [light_permute(s, m4, diag, initial, internal, terminal, d, q) for s in inputs]
    stored_outputs = [[int(x) for x in v["output"]] for v in cases]
    report(outputs == stored_outputs, f"{tag} test vector outputs match the light-layer permutation")
    stored_outputs[0][0] = (stored_outputs[0][0] + 1) % q
    negative_control(outputs != stored_outputs, f"{tag} one test vector output changed by +1")


def m4_blocks(m4, t):
    return [[(2 if i // 4 == j // 4 else 1) * m4[i % 4][j % 4] for j in range(t)] for i in range(t)]


def main():
    params = json.loads((HERE / "params.json").read_text())
    vectors = json.loads((HERE / "vectors.json").read_text())["vectors"]
    q = int(params["q"])
    factors = [int(f) for f in params["q_minus_1_factors"]]
    m4 = params["m4"]

    report(q == 2**64 - 59, "q = 2^64 - 59")
    report(math.prod(factors) == q - 1, f"q - 1 = {' * '.join(map(str, factors))}")
    report(all(trial_division_prime(f) for f in factors), "each factor of q - 1 is prime (trial division)")
    report(lucas_prime(q, factors), "q is prime (Lucas test with the factorization of q - 1)")

    mismatches = [(p.bit_length(), t, d) for p, table in PLONKY3_ROUNDS_128.items()
                  for (t, d), expected in table.items() if round_numbers(p, t, d) != expected]
    report(not mismatches, f"round-number search reproduces all 30 Plonky3 poseidon2_round_numbers_128 "
                           f"entries (31-bit and 64-bit) {mismatches or ''}")
    report(all(round_numbers(q, t, d) == v for (t, d), v in PLONKY3_ROUNDS_128[GOLDILOCKS].items()),
           "over q, every 64-bit (t, d) entry equals the Plonky3 64-bit table")

    report(is_mds(m4, q), f"M4 = {m4} is MDS over F_q (all 69 square minors nonzero)")
    negative_control(not is_mds(NON_MDS_M4, q), f"MDS check on {NON_MDS_M4}")

    for p, name, values in PLONKY3_DIAGONALS:
        diag = [parse_fraction(v, p) for v in values]
        failure = internal_matrix_failure(diag, p)
        report(failure is None, f"positive control: M_I check accepts {name} over its own field")

    plonky3_goldilocks_kat(m4)

    for inst in params["instances"]:
        check_instance(inst, m4, q, vectors, params["test_state_label_suffix"])

    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
