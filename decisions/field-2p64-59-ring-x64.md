# Field `2^64 − 59` and Ring `X^64 + 1`

## Status

Accepted by the owner on 2026-11-13: the prime, the ring, rank 18, and SHAKE
for the Akita challenges. The other entries (the 5-lane sampler, the limit of
the auxiliary maps, the profile name, the wording of the hash rule) follow
from these and wait for the owner's review. The owner approved `k_rho = 14`
and the Module-SIS method of section Security calculation on 2026-11-14. The owner
agreed with the sister project ProtogaLattice on `K = F_q[u]/(u² − 2)` on
2026-11-14.

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

Nightstream uses one profile, "Nightstream P64, k_rho = 14":

| Parameter | Value |
|---|---|
| Prime `q` | `2^64 − 59 = 18446744073709551557` |
| Ring | `F_q[X]/(X^64 + 1)`, `η = 128`, `d = 64` |
| Sum-check field `K` | `F_q[u]/(u² − 2)` |
| Challenge alphabet | `{−2, −1, 0, 1, 2}` |
| Expansion factor `T` | `128` |
| Decomposition | `b = 2`, `k_rho = 14`, `B = 2^14` |
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
- `q ≡ 5 (mod 8)`, so 2 is not a square modulo `q`, and `K` is a field
  (`Spec.FieldExtension.twoProjectiveNonresidue`). Akita's 64-bit profile and
  the sister project ProtogaLattice use the same `K`, so no map between two
  forms of the field is necessary.
- `gcd(7, q − 1) = 1`, so the Poseidon2 S-box `x^7` is a permutation.
- Why 14: at `T = 128` and `cap = 64` the deepest accumulator is
  `(1 + cap + k_rho)·T = 79·128 = 10,112`, which is below `B = 2^14`
  (`Spec.GlobalParams.production_accumulatorBound_lt_bigB`).

### Security calculation

The owner approved this method. The norm is the largest extraction norm,
the chain norm `8T(B + cap·T·(b − 1)) = 8T(2^14 + 64T) = 25,165,824`. There
is no 3-bit allowance for more targets, because the main key is one SHAKE128
matrix, so there is one target. The width is `2^28` coefficients.

| Rank `κ` | Norm | ADPS16 quantum | BKZ block size | ADPS16 classical | Default model |
|---|---|---|---|---|---|
| 18 | 25,165,824 | 127.20 bits | 480 | 140.16 bits | 166.84 bits |

The sister project ProtogaLattice computed these values with
lattice-estimator commit `3e48ef4` and Sage 10.9. Nightstream did not run
them again, and the files of the run are not in the repository
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
item 2).

The estimate is for generic lattice attacks on a uniform matrix. It is not a
proof for structured Module-SIS or for a fixed public seed.

### Consequences

- The Spartan, WHIR, and shrink compression path is removed. The compression
  over Akita of [`akita-compression.md`](./akita-compression.md) replaces
  it.
- `q − 1` has 2-adicity 2. A scheme that needs a large power-of-two
  multiplicative subgroup of `F_q` cannot be used.
- Poseidon2 has an instance over `F_q`
  ([`poseidon2-p64-instance.md`](./poseidon2-p64-instance.md)).
- The sampler of a ring challenge reads 5 field lanes and reduces them
  modulo `5^64` ([SPEC.md](../SPEC.md) section 5.3).
- The auxiliary rank-two maps of
  [`rank-two-sis-security.md`](./rank-two-sis-security.md) keep their limit
  as a count of coefficients, `50,371 · 54 = 2,720,034`. The rank in
  coefficients grows from 108 to 128 at the same width and norm, so the
  instance is not easier. The limit is then 42,500 ring columns. No map of
  the current tree uses the rank two.

### Hash rule

Poseidon2 makes each Fiat–Shamir challenge of the folding scheme and of the
step circuits, because the step circuits compute these challenges again.
SHAKE128 expands the Ajtai setup. The owner also accepts SHAKE for the
challenges of the Akita commitment scheme, which no step circuit computes.

### Scope

This decision supersedes the rank `κ = 22` and the field of
[`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md).
The SHAKE128 expander of
[`fprime-ajtai-shake128-setup.md`](./fprime-ajtai-shake128-setup.md) stays in
force with this modulus, with the sampling rule of
[`fprime-ajtai-rejection-sampling.md`](./fprime-ajtai-rejection-sampling.md).

It does not change `b = 2`, the `2^28` domain, the selected application, or
the recursion.

This decision selects the objects of these external assumptions
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md),
Assumptions):

- Module-SIS hardness at rank 18 and the chain norm, in the model of section
  Security calculation (`[MSIS]`);
- Poseidon2 over `F_q`, with the derived instance (`[P2-PERM]`,
  `[SPONGE]`); and
- the security of Akita for the compression (`[AKITA]`).

The low-norm invertibility bound of Theorem 4 for `X^64 + 1` is not an
assumption, because Lean proves it (`Spec.RingStrongSet.lowNormInvertibility`).
