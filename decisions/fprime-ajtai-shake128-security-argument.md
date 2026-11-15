# Ajtai key expander: security argument for review

Status: draft for external review, 2026-09-29. The owner selected SHAKE128
([decision](fprime-ajtai-shake128-setup.md)), and Lean and
Rust now implement the construction of section 4. Lean proves the
ideal-model steps of section 5 (section 7). Premises P1 and P2 stay external,
and no external review has checked the argument. The seed and domain policy
(section 6) is still an owner choice.

Changed on 2026-11-14: [`fprime-ajtai-rejection-sampling.md`](./fprime-ajtai-rejection-sampling.md) replaces the 256-bit wide reduction with 8-byte rejection sampling and a fallback. Sections 4, 5 and 7 state the new rule.

Superseded in part on 2026-11-13: [`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) replaces the Goldilocks prime, the ring `X^54 + X^27 + 1` and the 22 rows of this argument with `q = 2^64 − 59`, the ring `X^64 + 1` and 18 rows. The numbers below are those of the prior profile.

## 1. Purpose

The current setup rests on a premise about one fixed matrix
(`fprime-ajtai-public-seed-msis-assumption.md`). This document proposes standard premises
and a reduction from them, and it selects an expander that supports that
reduction.

## 2. Current state

- SuperNeo, Definition 8 and Theorem 6: `Setup` samples the matrix `M`
  uniformly at random. Binding follows from MSIS for a uniform matrix. The
  paper does not discuss seeds.
- Until 2026-09-29, Nightstream expanded `M` with
  `nightstream-ajtai-chacha20-wide256-v1` from one fixed public seed.
  `fprime-ajtai-public-seed-msis-assumption.md` assumes that MSIS is hard for that specific
  matrix. The approval does not cover the SHAKE128 matrix.
- Lean proves that a binding collision gives a short kernel vector for the
  same key: below `2B` for ordinary binding (`Binding.lean`) and below
  `8TB = 113,246,208` for relaxed binding (`RelaxedBinding.lean`). The
  random-oracle knowledge theorem computes such a kernel vector from each
  counted rerun of its binding reduction (`RandomOracleBinding.rerunKernel`),
  and with the key drawn from uniform streams inside its game it bounds the
  binding term by an MSIS solver's success on a uniform matrix
  (`RandomOracleSetup.knowledge_error_le_setup`, through
  `Programming.expect_le_programmed`). Hardness of the kernel problem is a
  premise. No Lean result proves it.
- Pseudorandomness of the expander does not help. The seed is public, so any
  attacker can calculate `M` (`FOUNDATION_SECURITY_REVIEW.md` in `docs/reviews/nightstream-fprime-requirements` at commit `f7e093d8f`).

## 3. Premises

- **P1, ideal expander.** SHAKE128 is a random oracle: for each distinct
  input, its output is an independent uniform byte string.
- **P2, uniform MSIS.** MSIS is hard for a uniformly random matrix with
  22 rows, 4,708,530 ring columns over `F[X]/(X^54 + X^27 + 1)`, and
  coefficient bound 113,246,208. This is the paper's premise at the Nightstream
  parameters. The lattice estimator gives a cost estimate, not a proof.
- **P3.** The existing Poseidon2 premise for the package identity. It does not
  change.

Kyber uses the same form of premise: its proofs model SHAKE-128, which
expands its matrix, as a random oracle (specification round 3, §5.4). FrodoKEM
proves the expansion step in the ideal model and states that it covers an
attacker who knows the seed (specification 2021, §5.1.3). The indexed XOF
structure and the rejection sampling follow Kyber. This construction reads a
fixed number of words for each element and has a fallback.

## 4. Construction

```text
domain = setup_id ‖ seed        (38 + 32 bytes)
setup_id = "nightstream-ajtai-shake128-reject64-v1"
input(row, column) = domain ‖ row_u32_le ‖ column_u64_le        (82 bytes)
words(row, column) = the first 84 little-endian 64-bit words of
    SHAKE128(input(row, column))
coefficient(row, column, lane) =
    the lane-th word of words(row, column) below q,
    or 0 when fewer than lane + 1 words are below q
```

- A word of `q` or more is rejected. The 0 is the fallback.
- Every input has the same length, so the map from `(row, column)` to the input
  is injective for rows below `2^32` and columns below `2^64`. Lean proves this
  (`elementInput_injective`). Each key element has its own XOF input and reads
  the 84 words of one 672-byte output (`elementWords_length`).
- `seed` is the existing 32-byte verifier-owned seed. A package domain
  (section 6) would replace it.
- Generation stays indexed and lazy: one XOF call gives one key element.

## 5. Argument

The attacker receives the setup description, can evaluate SHAKE128 on any
input, and wins if it outputs a binding collision for the expanded key.

**Game 0.** The real game.

**Game 1.** Replace SHAKE128 by a random oracle `H`. The sponge construction
is indifferentiable from a random oracle in the ideal-permutation model, with
an advantage of about `N² / 2^(c+1)`, where `N` counts all permutation calls
and `c = 256` for SHAKE128 (Bertoni, Daemen, Peeters, Van Assche, 2008). `N`
includes the setup expansion itself: at most 22 × 4,708,530 elements times 4
calls, below `2^29`.

**Game 2.** The reduction receives a uniform MSIS matrix `A` (P2). It answers
the oracle as follows:

- For a setup input `x(row, column)`, it samples the 84 words uniformly from
  the streams whose kept words give the ring element `A[row][column]`. It
  does this lazily and stores the answer.
- For every other input, it returns fresh uniform bytes and stores them.
- It answers repeated inputs from storage.

The attacker's own queries to setup inputs therefore return the same bytes
that define the key, so its view stays consistent. The reduction recognizes a
setup input by its fixed 82-byte format and its 70-byte domain prefix.

**Distance between Game 1 and Game 2.** In both games a stream is a uniform
preimage of its ring element. Only the law of the element differs. It is
uniform in Game 2. In Game 1 it is exactly uniform unless the fallback gives
a coefficient, which needs 21 or more rejected words of the 84. With
`ε = C(84, 21) · 59^21 / 2^1344`, below `2^-1155`, the two laws differ by at
most `2ε` in total absolute difference. For `N` independent key elements,
the probability of any event of the attacker's view changes by at most
`2Nε`. The bounded-depth key has `N = 18 · 2^20`, so the change is below
`2^-1129`.

**Result.** A collision in Game 2 gives a short kernel vector for `A` through
the existing Lean reduction. So

```text
Adv_binding ≤ Adv_MSIS(uniform) + N² / 2^257 + 2^-1129
```

in the classical random-oracle model, with the attacker's work, including any
work done before the proof, counted against P2.

## 6. Seed and domain policy

- **Reproducibility.** A specified derivation of `domain` lets anyone
  recompute the key. It does not prove that nobody searched: a designer can
  try many seeds or labels (NewHope, §2).
- **Search.** In the ideal model, a search over `S` candidate domains raises
  the chance of a weak matrix by at most a factor `S` over one uniform matrix.
  P2 must hold with that factor. `S` is the designer's search budget; this
  document does not fix a value.
- **Separation.** A different domain for each package gives a different matrix.
  Under P1 and P2, a solution for one matrix does not help with another. A
  weakness in the expander or in the parameters would still affect all of them.
- **Shared exposure.** With one global domain, all proofs depend on one
  matrix. A large one-time computation against that matrix is worth more. It
  is not an automatic forgery: a false proof needs more than a kernel vector.
  FrodoKEM and NewHope make a new matrix for each key for this reason.
- **No circular derivation.** The final package identity hashes the setup
  (`packageIdentityPreimage` contains `verifierContextDescriptor fits setup`).
  A package domain must come from an identifier fixed before the setup, for
  example `structuralPackageIdentity program fits`. The final identity then
  binds the generated setup, as it does now.

The owner must select a global domain or a domain for each package.

## 7. Lean obligations

1. Specify the expander: the input encoding, its injectivity and the disjoint
   output slices. Done: `Spec/AjtaiSetupV1.lean`,
   `Spec/AjtaiSetupV1/Shake128.lean` and `Spec/AjtaiSetupV1/IndexEncoding.lean`.
2. Done. The law of one key element: `Rejection.card_element` (each ring
   element is the element of the same number of streams with no fallback),
   `Rejection.card_fallback_le` and `Rejection.fallbackBound_lt` (the
   probability of the fallback).
3. Done. The programming step: `Programming.real_sub_programmed_le` bounds
   the change of any event of the attacker's complete view, all setup streams
   and every other value it reads (`Extra`), when each stream becomes a
   uniform preimage of a uniform matrix element. The bound for one element is
   `Programming.element_bias_le`, which uses step 2; the product bound is
   `product_difference_le`.
4. Done. `Programming.binding_le_solver` composes step 3 with
   `IsCollision.shortKernel`, the existing collision-to-kernel reduction.
   `RandomOracleSetup.programmingError_lt` bounds the error at the
   bounded-depth key below `2^-1129`, and `Programming.verifierKey_eq` shows
   that the key is the element key of its SHAKE128 streams.

Not proved in Lean: P1 and P2; that P1 applied to the distinct 82-byte inputs
gives independent uniform 64-bit words (each word encodes its bytes
bijectively); the running time of the simulation; and the indifferentiability
term `N² / 2^257`. The Lean result stays conditional on P1 and P2.

## 8. Expander choice

Measured cost of one production key pass
(`crates/neo-ajtai/tests/evidence/key-expander-20260929/` at commit `f7e093d8f`):

| Expander | CPU | Metal |
|---|---:|---:|
| ChaCha20, current | 7.11 s | 0.604 s |
| ChaCha20, both halves of each block | 4.51 s | 0.322 s |
| SHAKE128 | 7.50 s | 1.360 s |
| SHAKE256 | 8.67 s | 1.607 s |

CPU times use the ARMv8 SHA3 instructions. The Metal Keccak kernel is not
tuned.

- **SHAKE128 (selected).** The random-oracle model for SHAKE128 is the
  standard basis for public-matrix expansion (Kyber, FrodoKEM). The sponge
  has a published indifferentiability proof. In the integrated prover, the CPU
  lifecycle cost stayed within run-to-run variation. On Metal, the third
  lifecycle step took 18.5 s instead of 17.6 s with ChaCha20.
- **SHAKE256.** A capacity of 512 bits instead of 256. At the Nightstream
  security target, the generic term for SHAKE128 is already far smaller than
  the MSIS term. It costs about 16% more than SHAKE128 on CPU and 18% more on
  Metal.
- **ChaCha20.** It is cheaper, but I found no published argument for ChaCha20
  as a public-key matrix expander. An argument must use the random-permutation
  model on the input domain with fixed constants: the bare permutation was
  designed to be ideal only up to certain symmetries (Barbero, Bellini,
  Makarim, 2020). The multi-user proof for ChaCha20-Poly1305 models the
  permutation as random, but it assumes a secret key (Degabriele et al.,
  CCS 2021).

## 9. Questions for the reviewer

1. Is the random-oracle model for SHAKE128 acceptable as the key expander of
   a proof system with one fixed domain, when KEMs make a new matrix for each
   key?
2. Is the programming argument in section 5 complete, including the
   attacker's direct queries and the preimage sampling?
3. Is a bound on the attacker's total work sufficient for precomputation
   against a fixed domain, or is an auxiliary-input model necessary?
4. Should each package have its own domain?
5. The Fiat–Shamir path uses the classical random-oracle model
   (`NSD-THREAT-MODEL-001`). Is a quantum random-oracle version of section 5
   necessary?

## Sources

- SuperNeo v1.2, Definitions 5 and 8, Theorem 6 (`docs/superneo-paper-v1_2/`).
- CRYSTALS-Kyber specification, round 3, §5.4:
  <https://pq-crystals.org/kyber/data/kyber-specification-round3-20210131.pdf>
- FrodoKEM specification, 2021, §1 and §5.1.3:
  <https://frodokem.org/files/FrodoKEM-specification-20210604.pdf>
- Alkim, Ducas, Pöppelmann, Schwabe, Post-quantum key exchange — a new hope,
  §2: <https://cryptojedi.org/papers/newhope-20160803.pdf>
- Bertoni, Daemen, Peeters, Van Assche, On the indifferentiability of the
  sponge construction, EUROCRYPT 2008.
- Degabriele, Govinden, Günther, Paterson, The security of ChaCha20-Poly1305 in
  the multi-user setting, CCS 2021: <https://eprint.iacr.org/2023/085>
- Barbero, Bellini, Makarim, Rotational analysis of ChaCha permutation, 2020:
  <https://arxiv.org/abs/2008.13406>
- Nightstream: `fprime-ajtai-public-seed-msis-assumption.md`,
  `FOUNDATION_SECURITY_REVIEW.md` in `docs/reviews/nightstream-fprime-requirements` at commit `f7e093d8f`,
  `decisions/fprime-stage1-main-ajtai-setup.md`.
