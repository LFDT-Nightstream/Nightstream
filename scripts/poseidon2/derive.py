#!/usr/bin/env python3
"""Derive Poseidon2 parameters over F_q, q = 2^64 - 59, for widths 16 and 12.

Run `python3 scripts/poseidon2/derive.py`. It writes params.json and
vectors.json next to this file. Python 3 standard library only. Rerun
`check.py` to re-validate the output independently.

Layer conventions match p3-poseidon2 0.5.3 as p3-goldilocks 0.5.3 uses it:

- permute: x <- M_E x, then R_F/2 full rounds, R_P partial rounds,
  R_F/2 full rounds.
- full round: x_i <- (x_i + c_i)^d for each i, then x <- M_E x.
- partial round: x_0 <- (x_0 + c)^d, then x <- M_I x.
- M_E = circ(2 M4, M4, ..., M4) in 4x4 blocks, with M4 = MDSMat4.
- M_I = J + diag(D), J the all-ones matrix: (M_I x)_i = D_i x_i + sum(x),
  as in p3 `matmul_internal`.
- Round constants come from one SHAKE128 stream in the draw order of p3
  `Poseidon2::new_from_rng`: initial full rounds (row by row), terminal
  full rounds, then partial rounds.
"""

import hashlib
import json
import math
from decimal import ROUND_CEILING, Decimal, getcontext
from functools import lru_cache
from itertools import combinations, islice
from pathlib import Path

Q = 2**64 - 59
Q_MINUS_1_FACTORS = (2, 2, 11, 137, 547, 5594472617641)
GOLDILOCKS = 2**64 - 2**32 + 1
MERSENNE31 = 2**31 - 1
SECURITY_BITS = 128
WIDTHS = (16, 12)
CANDIDATE_DEGREES = (3, 5, 7)
SBOX_DEGREE = 7
SBOX_MULTIPLICATIONS = {3: 2, 5: 3, 7: 4}

MDS_MAT4 = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
HL_MAT4 = ((5, 7, 1, 3), (4, 6, 1, 1), (1, 3, 5, 7), (1, 1, 4, 6))
M4 = MDS_MAT4

PLONKY3_ROUNDS_128 = {
    31: {
        (16, 3): (8, 20), (16, 5): (8, 14), (16, 7): (8, 13), (16, 9): (8, 13), (16, 11): (8, 13),
        (24, 3): (8, 23), (24, 5): (8, 22), (24, 7): (8, 21), (24, 9): (8, 21), (24, 11): (8, 21),
        (32, 3): (8, 31), (32, 5): (8, 30), (32, 7): (8, 30), (32, 9): (8, 30), (32, 11): (8, 30),
    },
    64: {
        (8, 3): (8, 41), (8, 5): (8, 27), (8, 7): (8, 22), (8, 9): (8, 19), (8, 11): (8, 17),
        (12, 3): (8, 42), (12, 5): (8, 27), (12, 7): (8, 22), (12, 9): (8, 20), (12, 11): (8, 18),
        (16, 3): (8, 42), (16, 5): (8, 27), (16, 7): (8, 22), (16, 9): (8, 20), (16, 11): (8, 18),
    },
}
PLONKY3_PRIME = {31: MERSENNE31, 64: GOLDILOCKS}

LABEL = "Nightstream/Poseidon2/q=2^64-59/t={t}/d={d}/rf={rf}/rp={rp}/v1"
TEST_STATE_SUFFIX = "/test-states"
TEST_STATE_COUNT = 3
DIAGONAL_RULE = (
    "pool P = (1, 2, -2, 3, -3, 4, -4, ...), all integers except 0 and -1, ordered by |v| then sign; "
    "for n = t, t+1, ...: for each (t-1)-subset S of indices 0..n-2 in itertools.combinations order: "
    "candidate D = [P[i] for i in S] + [P[n-1]]; the first candidate that passes all M_I checks is chosen"
)

HERE = Path(__file__).resolve().parent

getcontext().prec = 60


def is_probable_prime(n):
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ceil(x):
    return int(Decimal(x).to_integral_value(rounding=ROUND_CEILING))


def binomial_squared_exceeds(n, k, bits):
    """C(n, k)^2 > 2^bits; exact near the boundary. The reference test
    ceil(2 log2 C(n, k)) >= m is this with bits = m - 1."""
    estimate = 2 * (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)
    if abs(estimate - bits) > 1:
        return estimate > bits
    binom = math.comb(n, k)
    return binom * binom > 2**bits


def round_numbers(p, t, d, m=SECURITY_BITS):
    """`find_FD_round_numbers` and `sat_inequiv_alpha` of the Poseidon2
    reference generator (HorizenLabs poseidon2_rust_params.sage), with the
    security margin on (R_F + 2, ceil(1.075 R_P)) and cost t R_F + R_P.
    No copy of that script is in this tree; main() and check.py compare the
    result with the Plonky3 poseidon2_round_numbers_128 table."""
    n = p.bit_length()
    ln2, lnd = Decimal(2).ln(), Decimal(d).ln()
    log2p = Decimal(p).ln() / ln2
    log_d_2 = ln2 / lnd
    statistical = 6 if m <= (n - 1 - (d - 1) // 2) * (t + 1) else 10
    interpolation = 1 + ceil(log_d_2 * min(m, n)) + ceil(Decimal(t).ln() / lnd)
    groebner1 = log_d_2 * min(m, log2p)
    groebner2 = t - 1 + log_d_2 * min(Decimal(m) / (t + 1), log2p / 2)
    groebner3 = t - 2 + Decimal(m) / (2 * lnd / ln2)
    r = t // 3

    @lru_cache(maxsize=None)
    def rf_min(rp):
        return max(statistical, interpolation - rp, ceil(groebner1 - rp),
                   ceil(groebner2 - rp), ceil((groebner3 - rp) / (t - 1)))

    def secure(rf, rp):
        if rf < rf_min(rp):
            return False
        over = (rf - 1) * t + rp + r + r * (rf // 2) + rp + d
        under = r * (rf // 2) + rp + d
        return binomial_squared_exceeds(over, under, m - 1)

    best = None
    for rp in range(1, 500):
        for rf in range(4, 100, 2):
            if secure(rf, rp):
                # The reference overwrites its loop variable R_P_t here, so later R_F values
                # in this row are tested against the margined R_P. check.py searches without it.
                rf_margin, rp = rf + 2, (43 * rp + 39) // 40
                cost = t * rf_margin + rp
                if best is None or cost < best[0] or (cost == best[0] and rf_margin < best[1]):
                    best = (cost, rf_margin, rp)
    return best[1], best[2]


def det(matrix):
    a = [[v % Q for v in row] for row in matrix]
    n, result = len(a), 1
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col]), None)
        if pivot is None:
            return 0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            result = -result
        result = result * a[col][col] % Q
        inv = pow(a[col][col], -1, Q)
        for r in range(col + 1, n):
            factor = a[r][col] * inv % Q
            if factor:
                a[r] = [(x - factor * y) % Q for x, y in zip(a[r], a[col])]
    return result % Q


def is_mds(matrix):
    n = len(matrix)
    return all(
        det([[matrix[i][j] for j in cols] for i in rows]) != 0
        for k in range(1, n + 1)
        for rows in combinations(range(n), k)
        for cols in combinations(range(n), k)
    )


def matmul(a, b):
    columns = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) % Q for col in columns] for row in a]


def matvec(a, v):
    return [sum(x * y for x, y in zip(row, v)) % Q for row in a]


def charpoly(a):
    """Faddeev-LeVerrier; coefficients from x^0 up to the monic x^t."""
    t = len(a)
    coeffs = [0] * t + [1]
    mk = [[0] * t for _ in range(t)]
    for k in range(1, t + 1):
        mk = matmul(a, mk)
        for i in range(t):
            mk[i][i] = (mk[i][i] + coeffs[t - k + 1]) % Q
        trace = sum(row[i] for i, row in enumerate(matmul(a, mk)))
        coeffs[t - k] = -trace * pow(k, -1, Q) % Q
    return coeffs


def trim(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def poly_mod(a, f):
    a = [x % Q for x in a]
    n = len(f) - 1
    for i in range(len(a) - 1, n - 1, -1):
        c = a[i]
        if c:
            for j in range(n):
                a[i - n + j] = (a[i - n + j] - c * f[j]) % Q
    return trim(a[:n] or [0])


def poly_mulmod(a, b, f):
    product = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] += x * y
    return poly_mod(product, f)


def poly_gcd(a, b):
    a, b = trim([x % Q for x in a]), trim([x % Q for x in b])
    while b != [0]:
        inv = pow(b[-1], -1, Q)
        b = [x * inv % Q for x in b]
        a, b = b, poly_mod(a, b)
    return [x * pow(a[-1], -1, Q) % Q for x in a]


def is_irreducible(f):
    """Rabin's test for a monic f of degree n over F_q: x^(q^n) = x mod f and
    gcd(x^(q^(n/r)) - x, f) = 1 for each prime r dividing n."""
    n = len(f) - 1
    xq, base, e = [1], [0, 1], Q
    while e:
        if e & 1:
            xq = poly_mulmod(xq, base, f)
        base, e = poly_mulmod(base, base, f), e >> 1
    frobenius_rows = [[1]]
    for _ in range(1, n):
        frobenius_rows.append(poly_mulmod(frobenius_rows[-1], xq, f))

    def frobenius(g):
        out = [0] * n
        for c, row in zip(g, frobenius_rows):
            for j, v in enumerate(row):
                out[j] = (out[j] + c * v) % Q
        return trim(out)

    images = [[0, 1]]
    for _ in range(n):
        images.append(frobenius(images[-1]))
    if images[n] != [0, 1]:
        return False
    for r in {r for r in range(2, n + 1) if n % r == 0 and all(r % s for s in range(2, r))}:
        g = images[n // r] + [0] * 2
        g[1] -= 1
        if poly_gcd(trim(g), f) != [1]:
            return False
    return True


def external_matrix(t):
    return [[(2 if i // 4 == j // 4 else 1) * M4[i % 4][j % 4] for j in range(t)] for i in range(t)]


def internal_matrix(diag):
    t = len(diag)
    return [[(1 + diag[i] if i == j else 1) % Q for j in range(t)] for i in range(t)]


def internal_matrix_failure(diag):
    """Poseidon2 (ePrint 2023/323) section 5.2 and 5.3, as checked by
    p3-poseidon2 0.5.3 internal.rs: every entry of M_I and of D nonzero, M_I
    invertible, and the characteristic polynomial of M_I^i irreducible for
    i = 1..2t. Returns None when the diagonal passes."""
    if any(v % Q in (0, Q - 1) for v in diag):
        return "an entry of D or of M_I is zero"
    m = internal_matrix(diag)
    if det(m) == 0:
        return "M_I is singular"
    power = m
    for i in range(1, 2 * len(diag) + 1):
        if not is_irreducible(charpoly(power)):
            return f"charpoly of M_I^{i} is reducible"
        power = matmul(power, m)
    return None


def diagonal_pool():
    yield 1
    v = 2
    while True:
        yield v
        yield -v
        v += 1


def find_diagonal(t):
    pool = list(islice(diagonal_pool(), 4 * t))
    tried = 0
    for n in range(t, len(pool) + 1):
        for head in combinations(range(n - 1), t - 1):
            diag = [pool[i] for i in head] + [pool[n - 1]]
            tried += 1
            failure = internal_matrix_failure(diag)
            if failure is None:
                return diag, tried
            print(f"    candidate {tried}: {diag} rejected, {failure}")
    raise RuntimeError("no diagonal found")


def field_stream(label):
    xof = hashlib.shake_128(label.encode("ascii"))
    offset, size = 0, 1024
    while True:
        buffer = xof.digest(size)
        while offset + 8 <= size:
            word = int.from_bytes(buffer[offset:offset + 8], "little")
            offset += 8
            if word < Q:
                yield word
        size *= 2


def round_constants(label, t, rf, rp):
    stream = field_stream(label)
    take = lambda count: list(islice(stream, count))
    initial = [take(t) for _ in range(rf // 2)]
    terminal = [take(t) for _ in range(rf // 2)]
    return initial, terminal, take(rp)


def permute(instance, state):
    """Reference permutation with dense M_E and M_I, straight from the definition."""
    me, mi, d = instance["me"], instance["mi"], instance["d"]
    x = matvec(me, state)
    for c in instance["initial"]:
        x = matvec(me, [pow(v + k, d, Q) for v, k in zip(x, c)])
    for c in instance["internal"]:
        x[0] = pow(x[0] + c, d, Q)
        x = matvec(mi, x)
    for c in instance["terminal"]:
        x = matvec(me, [pow(v + k, d, Q) for v, k in zip(x, c)])
    return x


def decimal_strings(values):
    return [str(v % Q) for v in values]


def write_json(path, value):
    text = json.dumps(value, indent=1) + "\n"
    path.write_text(text)
    print(f"wrote {path.name} sha256={hashlib.sha256(text.encode()).hexdigest()}")


def main():
    product = math.prod(Q_MINUS_1_FACTORS)
    print(f"q = {Q} prime: {is_probable_prime(Q)}")
    print(f"q - 1 = {' * '.join(map(str, Q_MINUS_1_FACTORS))}: product ok {product == Q - 1}, "
          f"factors prime {all(is_probable_prime(f) for f in Q_MINUS_1_FACTORS)}")
    valid = [d for d in range(3, 12) if math.gcd(d, Q - 1) == 1]
    print(f"S-box degrees d < 12 with gcd(d, q - 1) = 1: {valid}")

    print("Plonky3 poseidon2_round_numbers_128 cross-check (computed vs table):")
    mismatches = 0
    for bits, table in PLONKY3_ROUNDS_128.items():
        for (t, d), expected in table.items():
            got = round_numbers(PLONKY3_PRIME[bits], t, d)
            mismatches += got != expected
            if (bits, t, d) in ((64, 16, 7), (64, 12, 7)) or got != expected:
                print(f"  {bits}-bit (t={t}, d={d}): computed {got}, Plonky3 {expected}")
            if bits == 64 and round_numbers(Q, t, d) != expected:
                mismatches += 1
                print(f"  q (t={t}, d={d}): differs from the 64-bit table")
    print(f"  {sum(map(len, PLONKY3_ROUNDS_128.values()))} table entries, {mismatches} mismatches "
          "(64-bit rows also recomputed for q)")
    assert mismatches == 0

    print("Round numbers over F_q, 128-bit security, with margin:")
    rounds = {}
    for t in WIDTHS:
        for d in CANDIDATE_DEGREES:
            rf, rp = rounds[t, d] = round_numbers(Q, t, d)
            sboxes = t * rf + rp
            print(f"  t={t} d={d}: R_F={rf} R_P={rp} S-boxes={sboxes} "
                  f"multiplications={sboxes * SBOX_MULTIPLICATIONS[d]}")

    print(f"M4 = MDSMat4 {M4} MDS over F_q: {is_mds(M4)}; HLMDSMat4 MDS over F_q: {is_mds(HL_MAT4)}")
    assert is_mds(M4) and math.gcd(SBOX_DEGREE, Q - 1) == 1

    instances, vectors = [], []
    for t in WIDTHS:
        d = SBOX_DEGREE
        rf, rp = rounds[t, d]
        label = LABEL.format(t=t, d=d, rf=rf, rp=rp)
        print(f"t={t}: searching the internal diagonal")
        diag, tried = find_diagonal(t)
        print(f"  chosen D = {diag} after {tried} candidates")
        me = external_matrix(t)
        assert det(me) != 0
        initial, terminal, internal = round_constants(label, t, rf, rp)
        instances.append({
            "q": str(Q), "t": t, "d": d, "rf": rf, "rp": rp, "label": label,
            "internal_diagonal": decimal_strings(diag),
            "internal_diagonal_signed": [str(v) for v in diag],
            "diagonal_candidates_tried": tried,
            "external_initial": [decimal_strings(row) for row in initial],
            "external_terminal": [decimal_strings(row) for row in terminal],
            "internal": decimal_strings(internal),
        })
        instance = {"me": me, "mi": internal_matrix(diag), "d": d,
                    "initial": initial, "terminal": terminal, "internal": internal}
        random = field_stream(label + TEST_STATE_SUFFIX)
        cases = [("iota", list(range(t))), ("zero", [0] * t)]
        cases += [(f"shake{i}", list(islice(random, t))) for i in range(TEST_STATE_COUNT)]
        for name, state in cases:
            vectors.append({"t": t, "name": name, "input": decimal_strings(state),
                            "output": decimal_strings(permute(instance, state))})

    write_json(HERE / "params.json", {
        "q": str(Q),
        "q_minus_1_factors": [str(f) for f in Q_MINUS_1_FACTORS],
        "m4": [list(row) for row in M4],
        "external_matrix": "circ(2 M4, M4, ..., M4) in 4x4 blocks (p3 mds_light_permutation)",
        "internal_matrix": "J + diag(internal_diagonal), (M_I x)_i = D_i x_i + sum(x) (p3 matmul_internal)",
        "constant_order": "SHAKE128(label) as 8-byte little-endian words, words >= q rejected; "
                          "external_initial row by row, then external_terminal, then internal",
        "permutation": "x <- M_E x; rf/2 full rounds; rp partial rounds (S-box on lane 0); rf/2 full rounds",
        "diagonal_rule": DIAGONAL_RULE,
        "test_state_label_suffix": TEST_STATE_SUFFIX,
        "instances": instances,
    })
    write_json(HERE / "vectors.json", {"q": str(Q), "vectors": vectors})


if __name__ == "__main__":
    main()
