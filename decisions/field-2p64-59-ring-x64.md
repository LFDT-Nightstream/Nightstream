# Field `2^64 − 59` and Ring `X^64 + 1`

## Status

Accepted by the owner on 2026-11-13.

## Problem

The final proof was compressed with Spartan over WHIR, then with a shrink
layer. The owner found this path too slow. WHIR and the shrink layer were the
larger part of the compression work.

Akita (ePrint 2026/1983) is a lattice polynomial commitment scheme with a
faster prover and a smaller proof. Akita is not secure over Goldilocks: the
ring `X^d + 1` splits fully modulo the Goldilocks prime, so short challenges
are not invertible. Akita needs a prime `q ≡ 2s + 1 (mod 4s)` for a small
power of two `s`. Its 64-bit profile uses `q = 2^64 − 59`, which has `s = 2`.

The folding scheme and the commitment scheme must use one field. So the
project must leave Goldilocks.

## SuperNeo

SuperNeo v1.2 is generic in the field and in the cyclotomic ring. The ring
must give Module-SIS hardness and a strong sampling set (Theorem 4 and
Definition 6). Section 8 gives three parameter sets for fields of 64 bits or
less. None of them uses `2^64 − 59`. Section 8.1 uses the ring `X^64 + 1` with
a different prime.

## Decision

Nightstream uses one profile, "Nightstream P64, k_rho = 16":

| Parameter | Value |
|---|---|
| Prime `q` | `2^64 − 59 = 18446744073709551557` |
| Ring | `F_q[X]/(X^64 + 1)`, `η = 128`, `d = 64` |
| Sum-check field `K` | `F_q[u]/(u² − 7)` |
| Challenge alphabet | `{−2, −1, 0, 1, 2}` |
| Expansion factor `T` | `128` |
| Decomposition | `b = 2`, `k_rho = 16`, `B = 2^16` |
| Main commitment rank `κ` | `18` |

This profile replaces Goldilocks, the ring `X^54 + X^27 + 1`, and rank 22.
There is no second profile and no compatibility path.

### Why these values hold

- `q` is prime and `q ≡ 5 (mod 8)`. The order of `q` modulo 128 is 32. So
  Theorem 4 holds with `z = 4`: `X^64 + 1` has two irreducible factors of
  degree 32, and each element with infinity norm below `q^(1/2)/√2`, about
  `2^31.5`, is invertible.
- Two elements of the alphabet differ by at most 4, so the challenge set is a
  strong sampling set. It has `5^64` elements, about `2^148.6`.
- For a power-of-two cyclotomic ring, Theorem 5 gives `T = φ(η) · 2 = 128`.
- 7 is not a square modulo `q`, so `K` is a field.
- `gcd(7, q − 1) = 1`, so the Poseidon2 S-box `x^7` is a permutation.

### Security calculation

The model is that of
[`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md):
width `2^28` coefficients, norm `8TB`, the lattice estimator with the ADPS16
quantum cost. The owner ran it at estimator commit `38618db` on 2026-11-13.
The control run gave the recorded value for the prior profile.

| Rank | Coefficients `κ·d` | BKZ block size | ADPS16 quantum |
|---|---|---|---|
| 22, prior profile | 1188 | 429 | 113.685 bits |
| 18, this profile | 1152 | 433 | 114.745 bits |

After the allowance of 3 bits for eight targets, rank 18 gives 111.7 bits.
The owner selected this level. It does not meet a 128-bit post-allowance
target. Rank 20 gives 130.3 bits after the allowance.

The estimate is for generic lattice attacks on a uniform matrix. It is not a
proof for structured Module-SIS or for a fixed public seed. The chain norm of
the bounded-depth fold, `8T(B + cap·T·(b − 1))`, was not run in the estimator
for this profile.

### Consequences

- The Spartan, WHIR, and shrink compression path is removed. A compression
  layer over Akita replaces it and has its own decision record.
- `q − 1` has 2-adicity 2. A scheme that needs a large power-of-two
  multiplicative subgroup of `F_q` cannot be used.
- Poseidon2 has a new instance over `F_q`, with its own derivation and checks.
- The sampler of a ring challenge reads 5 field lanes, because `5^64` is
  larger than `5^54`.
- The auxiliary rank-two maps of
  [`rank-two-sis-security.md`](./rank-two-sis-security.md) keep their limit
  as a count of coefficients, `50,371 · 54 = 2,720,034`. The rank in
  coefficients grows from 108 to 128 at the same width and norm, so the
  instance is not easier. The limit is then 42,500 ring columns.

### Hash rule

Poseidon2 makes each Fiat–Shamir challenge of the folding scheme and of the
step circuits, because the step circuits compute these challenges again.
SHAKE128 expands the Ajtai setup. The owner also accepts SHAKE for the
challenges of the Akita commitment scheme, which no step circuit computes.

## Scope and supersession

This decision supersedes the rank `κ = 22` and the field of
[`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md).
The SHAKE128 setup rules of
[`fprime-ajtai-shake128-setup.md`](./fprime-ajtai-shake128-setup.md) stay in
force with the new modulus.

It does not change `b = 2`, `k_rho = 16`, `B = 2^16`, the `2^28` domain, the
selected application, or the recursion.

The final security theorem must name these external assumptions:

- Module-SIS hardness under the pinned estimator model at rank 18;
- the low-norm invertibility bound of Theorem 4 for `X^64 + 1`;
- Poseidon2 over `F_q` as a random oracle, with the derived instance; and
- the security of Akita, when the compression layer uses it.
