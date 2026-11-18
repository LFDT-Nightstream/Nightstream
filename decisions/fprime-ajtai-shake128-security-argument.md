# Ajtai Key Expander: Security Argument

## Status

Proposed. This record is an argument for external review, and no external
review has checked it. The construction is accepted
([`fprime-ajtai-shake128-setup.md`](./fprime-ajtai-shake128-setup.md),
[`fprime-ajtai-rejection-sampling.md`](./fprime-ajtai-rejection-sampling.md)).
Lean proves the ideal-model steps. Premises P1 and P2 stay external. The
seed and domain policy below is an open owner choice.

## Problem

The seed is public, so each attacker can compute the key `M`.
Pseudorandomness of the expander does not help. A premise about one fixed
matrix
([superseded](fprime-ajtai-public-seed-msis-assumption.md)) is not a
standard premise. Binding of the expanded key needs standard premises and a
reduction from them.

## SuperNeo

SuperNeo v1.2 `Setup` samples the matrix `M` uniformly at random
(Definition 8). Binding follows from MSIS for a uniform matrix (Theorem 6).
The paper does not discuss seeds.

## Decision

### Premises

- **P1, ideal expander (`[KECCAK]`).** SHAKE128 is a random oracle: for each
  distinct input, its output is an independent uniform byte string.
- **P2, uniform MSIS (`[MSIS]`).** MSIS is hard for a uniformly random
  matrix of 18 rows over `F_q[X]/(X^64 + 1)` at the chain norm 25,165,824
  (`Spec.Folding.Schedule.production_msisChainBound`). This is the premise
  of the paper at the profile Nightstream P64, k_rho = 14. The estimate of
  [`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) is a cost
  estimate, not a proof.
- **P3.** The Poseidon2 premise for the package identity. It does not
  change.

Kyber uses the same form of premise: its proofs model SHAKE128, which
expands its matrix, as a random oracle (specification round 3, §5.4).
FrodoKEM proves the expansion step in the ideal model and states that it
covers an attacker who knows the seed (specification 2021, §5.1.3). The
indexed XOF structure and the rejection sampling follow Kyber. This
construction reads a fixed number of words for each element and has a
fallback.

### Construction

[SPEC.md](../SPEC.md) Algorithm 1 gives the construction. Each key element
`(row, column)` has its own 82-byte input `domain ‖ row ‖ column`, where
`domain` is the 38-byte setup ID and the 32-byte seed. Each input has the
same length, so the map from `(row, column)` to the input is injective for
rows below `2^32` and columns below `2^64`. Each key element reads the 84
words of one 672-byte output (`Spec.AjtaiSetupV1.elementWords_length`). A
word of `q` or more is rejected. When fewer than `lane + 1` words are kept,
the fallback sets coefficient `lane` to 0.

### Argument

The attacker receives the setup description, can evaluate SHAKE128 on each
input, and wins if it outputs a binding collision for the expanded key.

**Game 0.** The real game.

**Game 1.** Replace SHAKE128 by a random oracle `H`. The sponge construction
is indifferentiable from a random oracle in the ideal-permutation model, with
an advantage of about `N² / 2^(c+1)`, where `N` counts all permutation calls
and `c = 256` for SHAKE128 (Bertoni, Daemen, Peeters, Van Assche, 2008). `N`
includes the setup expansion: the bounded-depth key has `18 · 2^20` elements
of 4 calls each, which is below `2^27` calls (computed).

**Game 2.** The reduction receives a uniform MSIS matrix `A` (P2). It answers
the oracle as follows:

- For a setup input `x(row, column)`, it samples the 84 words uniformly from
  the streams whose kept words give the ring element `A[row][column]`. It
  does this lazily and stores the answer.
- For each other input, it returns fresh uniform bytes and stores them.
- It answers repeated inputs from storage.

So the attacker's own queries to setup inputs return the same bytes that
define the key, and its view stays consistent. The reduction recognizes a
setup input by its fixed 82-byte format and its 70-byte domain prefix.

**Distance between Game 1 and Game 2.** In both games a stream is a uniform
preimage of its ring element. Only the law of the element differs. It is
uniform in Game 2. In Game 1 it is exactly uniform unless the fallback gives
a coefficient, which needs 21 or more rejected words of the 84. With
`ε = C(84, 21) · 59^21 / 2^1344`, below `2^−1155`, the two laws differ by at
most `2ε` in total absolute difference. For `N_key` independent key
elements, the probability of each event of the attacker's view changes by at
most `2 · N_key · ε`. The bounded-depth key has `N_key = 18 · 2^20`, so the
change is below `2^−1129`.

**Result.** A collision in Game 2 gives a short kernel vector for `A`
through the Lean reduction. So

```text
Adv_binding ≤ Adv_MSIS(uniform) + N² / 2^257 + 2^−1129
```

in the classical random-oracle model. The work of the attacker, including
work before the proof, counts against P2.

### What Lean proves

- The input encoding (`Spec.AjtaiSetupV1.elementInput`). The file
  `Spec/AjtaiSetupV1/IndexEncoding.lean` states its injectivity, but no
  module imports that file, so the build does not check it.
- The law of one key element: each ring element is the element of the same
  number of streams with no fallback
  (`Spec.AjtaiSetupV1.Rejection.card_element`), and the bound of the
  fallback (`Spec.AjtaiSetupV1.Rejection.card_fallback_le`,
  `Spec.AjtaiSetupV1.Rejection.fallbackBound_lt`).
- The programming step: the change of each event of the attacker's complete
  view when each stream becomes a uniform preimage of a uniform matrix
  element (`Spec.AjtaiSetupV1.Programming.real_sub_programmed_le`,
  `Spec.AjtaiSetupV1.Programming.element_bias_le`,
  `Spec.AjtaiSetupV1.Programming.product_difference_le`).
- The composition with the collision-to-kernel reduction
  (`Spec.AjtaiSetupV1.Programming.binding_le_solver`,
  `Spec.AjtaiSetupV1.Programming.IsCollision.shortKernel`), the error below
  `2^−1129` at the bounded-depth key
  (`Export.BoundedDepth.RandomOracleSetup.programmingError_lt`), and that the
  key is the element key of its SHAKE128 streams
  (`Spec.AjtaiSetupV1.Programming.verifierKey_eq`).
- A relaxed binding collision gives a kernel vector of the same key with a
  norm below the chain norm
  (`Spec.RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel`).
  The knowledge bounds of both folds, averaged over the setup, have the MSIS
  advantage for a uniform matrix and the programming error as terms
  (`Export.BoundedDepth.RandomOracleSetup.light_knowledge_error_le_setup`,
  `Export.BoundedDepth.RandomOracleSetup.refresh_knowledge_error_le_setup`).

Lean does not prove: P1 and P2; that P1 on the distinct 82-byte inputs gives
independent uniform 64-bit words; the running time of the simulation; and
the indifferentiability term `N² / 2^257`. The Lean result stays conditional
on P1 and P2.

### Why SHAKE128

- **SHAKE128 (selected).** The random-oracle model for SHAKE128 is the
  standard basis for public-matrix expansion (Kyber, FrodoKEM). The sponge
  has a published indifferentiability proof.
- **SHAKE256.** It has a capacity of 512 bits in place of 256. The generic
  term for SHAKE128 is already far smaller than the MSIS term, and SHAKE256
  costs more prover time.
- **ChaCha20.** We found no published argument for ChaCha20 as a
  public-matrix expander. The bare permutation was designed to be ideal only
  up to certain symmetries (Barbero, Bellini, Makarim, 2020). The multi-user
  proof for ChaCha20-Poly1305 assumes a secret key (Degabriele et al.,
  CCS 2021).

### Seed and domain policy (open)

The current key has one global domain: the setup ID and the fixed seed. The
owner must select a global domain or a domain for each package.

- **Reproducibility.** A specified derivation of `domain` lets anyone
  compute the key again. It does not prove that nobody searched: a designer
  can try many seeds or labels (NewHope, §2).
- **Search.** In the ideal model, a search over `S` candidate domains raises
  the chance of a weak matrix by at most a factor `S` over one uniform
  matrix. P2 must hold with that factor.
- **Separation.** A different domain for each package gives a different
  matrix. Under P1 and P2, a solution for one matrix does not help with
  another. A weakness in the expander or in the parameters would still
  affect all of them.
- **Shared exposure.** With one global domain, all proofs depend on one
  matrix, so a large one-time computation against that matrix is worth
  more. It is not an automatic forgery: a false proof needs more than a
  kernel vector.
- **No circular derivation.** A package domain must come from an identifier
  that is fixed before the setup.

### Questions for the reviewer

1. Is the random-oracle model for SHAKE128 acceptable as the key expander of
   a proof system with one fixed domain, when KEMs make a new matrix for
   each key?
2. Is the programming argument complete, including the attacker's direct
   queries and the preimage sampling?
3. Is a bound on the attacker's total work sufficient for precomputation
   against a fixed domain, or is an auxiliary-input model necessary?
4. Should each package have its own domain?
5. The results use the classical random-oracle model
   ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), Out of
   scope). Is a quantum random-oracle version of the argument necessary?

### Sources

- SuperNeo v1.2, Definitions 5 and 8, Theorem 6.
- CRYSTALS-Kyber specification, round 3, §5.4:
  <https://pq-crystals.org/kyber/data/kyber-specification-round3-20210131.pdf>
- FrodoKEM specification, 2021, §1 and §5.1.3:
  <https://frodokem.org/files/FrodoKEM-specification-20210604.pdf>
- Alkim, Ducas, Pöppelmann, Schwabe, Post-quantum key exchange — a new hope,
  §2: <https://cryptojedi.org/papers/newhope-20160803.pdf>
- Bertoni, Daemen, Peeters, Van Assche, On the indifferentiability of the
  sponge construction, EUROCRYPT 2008.
- Degabriele, Govinden, Günther, Paterson, The security of ChaCha20-Poly1305
  in the multi-user setting, CCS 2021: <https://eprint.iacr.org/2023/085>
- Barbero, Bellini, Makarim, Rotational analysis of ChaCha permutation, 2020:
  <https://arxiv.org/abs/2008.13406>
