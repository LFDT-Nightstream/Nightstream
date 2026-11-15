# F′ Ajtai Key: 8-Byte Rejection Sampling

## Status

Accepted by the owner on 2026-11-14. It supersedes the setup ID and the
sampling rule of
[`fprime-ajtai-shake128-setup.md`](./fprime-ajtai-shake128-setup.md). The
seed, the SHAKE128 input layout, the rank, the dimensions and the authority
layout of that record stay in force.

## Problem

The 256-bit wide reduction reads 32 bytes of SHAKE128 output for each
coefficient. One key element reads 2,048 bytes, which is 13 Keccak-f
permutations. The prover expands key elements each time it commits, and the
owner puts prover speed first.

The sister project ProtogaLattice has the same field and ring and shares
parameters and methods with Nightstream where it can. It samples its key from
the same SHAKE128 input layout with 8-byte rejection sampling, at 4
permutations for one element (its record DR-011).

An earlier Nightstream setup used 64-bit rejection sampling with no fallback.
It needed a proof that sampling succeeded at each key index, which was not
practical
([`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md)).

## SuperNeo

SuperNeo `Setup` samples the commitment matrix uniformly at random
(Definition 8). The paper does not specify how to derive the matrix from a
seed.

## Decision

Sample each key element with the rule of ProtogaLattice:

- setup ID `nightstream-ajtai-shake128-reject64-v1` (38 bytes);
- key element `(row, block)` reads the first 84 words of SHAKE128 (FIPS 202)
  of the 82-byte input `setup_id || seed || row_u32_le || block_u64_le`. A
  word is 8 output bytes as one little-endian integer, and 84 words are four
  blocks of the rate;
- a word below `q = 2^64 − 59` is kept, and a word of `q` or more is
  rejected;
- coefficient `lane` is kept word `lane`; and
- the fallback: when fewer than `lane + 1` words are kept, coefficient `lane`
  is 0.

The fallback makes each key element a total function of its index, so no
proof of success at each index is necessary. ProtogaLattice uses the same
words, rejection and fallback under its own setup ID, so the two keys are
different matrices with one law.

Given SHAKE128 as a random oracle (`[KECCAK]`), a key element is exactly
uniform unless the fallback gives one of its coefficients: each ring element
is the element of the same number of streams that keep 64 words or more
(`Spec.AjtaiSetupV1.Rejection.card_element`). The fallback needs 21 or more
rejected words of the 84. Its probability for one element is at most
`C(84, 21) · 59^21 / 2^1344`, below `2^−1155`
(`Spec.AjtaiSetupV1.Rejection.card_fallback_le`,
`Spec.AjtaiSetupV1.Rejection.fallbackBound_lt`). The programming step of
[the security argument](fprime-ajtai-shake128-security-argument.md) costs
twice this bound for each key element
(`Spec.AjtaiSetupV1.Programming.element_bias_le`), which is below `2^−1129`
for the `18 · 2^20` elements of the bounded-depth key
(`Export.BoundedDepth.RandomOracleSetup.programmingError_lt`). The
wide-reduction bias bound is no longer in use.

Lean defines the rule (`Spec.AjtaiSetupV1.keptCoefficientNat`) and emits the
parity vectors that Rust checks. No index of the production key gives a
rejected word in practice, so the vectors include synthetic streams with
rejected words and with the fallback.
